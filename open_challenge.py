"""
WRO Future Engineers 2026 - OPEN CHALLENGE
Raspberry Pi | Camera-only | Servo steering + DC drive motor

WHAT THIS DOES
--------------
1. LANE CENTERING (camera-only, no distance sensors):
   Looks at a strip near the bottom of the frame, finds the left and right
   wall edges (by brightness), computes how far the track's center is from
   the frame's center, and steers proportionally (PID) to correct it.

2. LAP COUNTING:
   Reuses the orange/blue line-crossing logic from your lap counter to
   know when a lap is completed.

3. AUTOMATIC STOP:
   Drives forward until TARGET_LAPS is reached, then stops the motor.

BEFORE YOU RUN THIS - YOU MUST TUNE THESE FOR YOUR ROBOT
-----------------------------------------------------------
1. GPIO PIN NUMBERS (see CONFIG section) -- must match your actual wiring.
2. WALL_BRIGHTNESS_THRESHOLD -- run once with SHOW_DEBUG=True, look at the
   "Wall Mask" window, and adjust this number until the walls show white
   and the floor shows black in that window.
3. STEERING_MIN_ANGLE / STEERING_MAX_ANGLE -- your servo's safe mechanical
   steering limits (test on blocks first, wheels off the ground!).
4. DRIVE_SPEED -- start LOW (e.g. 0.3) for your first test run.
5. PID gains (KP, KI, KD) -- start with KP only (KI=KD=0), increase KP
   until the car reacts promptly without violently oscillating, then add
   a small KD to dampen any wobble.

SAFETY: Always test with the drive wheels off the ground first, and keep
a hand near the power switch for your first few runs.
"""

import cv2
import numpy as np
import json
import os
import time
from collections import deque
from gpiozero import AngularServo, Motor

# ============================================================
# CONFIG -- CHANGE THESE FOR YOUR ROBOT
# ============================================================

# ---- GPIO pins (BCM numbering) ----
STEERING_SERVO_PIN = 18          # PWM-capable pin
MOTOR_FORWARD_PIN = 23
MOTOR_BACKWARD_PIN = 24
MOTOR_ENABLE_PIN = 25            # PWM speed pin on your motor driver (None if not used)

# ---- Steering servo limits (degrees) ----
STEERING_MIN_ANGLE = -35         # full left
STEERING_MAX_ANGLE = 35          # full right
STEERING_CENTER_TRIM = 0         # nudge if car doesn't drive straight at angle 0

# ---- Drive settings ----
DRIVE_SPEED = 0.35               # 0.0 - 1.0, START LOW
TARGET_LAPS = 3                  # WRO Open Challenge = 3 laps

# ---- Camera ----
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
SHOW_DEBUG = True                # set False for max speed once tuned

# ---- Lane centering ----
LANE_ROI_TOP = int(HEIGHT * 0.55)     # region used to find wall edges
LANE_ROI_BOTTOM = int(HEIGHT * 0.85)
WALL_BRIGHTNESS_THRESHOLD = 170       # TUNE THIS -- pixels brighter than this = wall
SCAN_ROWS = 5                         # how many rows within the ROI to sample

# ---- Steering PID (error is in pixels, output is degrees) ----
KP = 0.12
KI = 0.0
KD = 0.05
MAX_INTEGRAL = 200

# ---- Lap line detection (reused from lap counter) ----
LINE_ROI_START = int(HEIGHT * 0.85)
MIN_AREA_LINE = 1500
line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

REFERENCE_COLORS = {
    "BLUE":   {"a": 120, "b": 90},
    "ORANGE": {"a": 155, "b": 185},
}
MAX_COLOR_DISTANCE = 35
STABLE_FRAMES_REQUIRED = 3


# ============================================================
# HARDWARE SETUP
# ============================================================
steering_servo = AngularServo(
    STEERING_SERVO_PIN,
    min_angle=STEERING_MIN_ANGLE,
    max_angle=STEERING_MAX_ANGLE,
    min_pulse_width=0.5 / 1000,
    max_pulse_width=2.5 / 1000,
)

drive_motor = Motor(
    forward=MOTOR_FORWARD_PIN,
    backward=MOTOR_BACKWARD_PIN,
    enable=MOTOR_ENABLE_PIN,
    pwm=True,
)


def set_steering(angle_deg):
    angle_deg = max(STEERING_MIN_ANGLE, min(STEERING_MAX_ANGLE, angle_deg))
    steering_servo.angle = angle_deg + STEERING_CENTER_TRIM


def drive_forward(speed):
    speed = max(0.0, min(1.0, speed))
    drive_motor.forward(speed)


def stop_motor():
    drive_motor.stop()


# ============================================================
# COLOR CALIBRATION (for lap line colors)
# ============================================================
def load_color_calibration():
    global REFERENCE_COLORS
    if os.path.exists("calibration.json"):
        with open("calibration.json", "r") as f:
            data = json.load(f)
        for color in ("BLUE", "ORANGE"):
            if color in data:
                REFERENCE_COLORS[color] = {"a": data[color]["a"], "b": data[color]["b"]}
        print("Loaded color calibration.json")
    else:
        print("No calibration.json found -- using fallback BLUE/ORANGE references. "
              "Run color_calibration_tool.py for better accuracy.")


def classify_lab(a_val, b_val, allowed_colors):
    best_color, best_dist = None, None
    for color in allowed_colors:
        ref = REFERENCE_COLORS[color]
        dist = ((a_val - ref["a"]) ** 2 + (b_val - ref["b"]) ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            best_dist, best_color = dist, color
    if best_dist is not None and best_dist <= MAX_COLOR_DISTANCE:
        return best_color
    return None


def white_balance(img):
    img = img.astype(np.float32)
    b, g, r = cv2.split(img)
    avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    avg_b, avg_g, avg_r = max(avg_b, 1), max(avg_g, 1), max(avg_r, 1)
    b = b * (avg_gray / avg_b)
    g = g * (avg_gray / avg_g)
    r = r * (avg_gray / avg_r)
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


# ============================================================
# LANE CENTERING (camera-only wall detection)
# ============================================================
def get_lane_error(frame):
    """Returns pixel error: negative = track center is LEFT of frame
    center (steer left), positive = track center is RIGHT (steer right).
    Returns None if walls can't be found on both sides."""

    roi = frame[LANE_ROI_TOP:LANE_ROI_BOTTOM, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    wall_mask = cv2.inRange(gray, WALL_BRIGHTNESS_THRESHOLD, 255)

    row_step = max(1, wall_mask.shape[0] // SCAN_ROWS)
    center_x = WIDTH // 2

    left_edges = []
    right_edges = []

    for row in range(0, wall_mask.shape[0], row_step):
        line = wall_mask[row]

        # scan left from center to find left wall
        left_wall = None
        for x in range(center_x, 0, -1):
            if line[x] > 0:
                left_wall = x
                break

        # scan right from center to find right wall
        right_wall = None
        for x in range(center_x, WIDTH):
            if line[x] > 0:
                right_wall = x
                break

        if left_wall is not None:
            left_edges.append(left_wall)
        if right_wall is not None:
            right_edges.append(right_wall)

    if not left_edges or not right_edges:
        return None, wall_mask  # lost a wall -- caller should handle (e.g. hold last steering)

    avg_left = sum(left_edges) / len(left_edges)
    avg_right = sum(right_edges) / len(right_edges)

    track_center = (avg_left + avg_right) / 2.0
    error = track_center - center_x

    return error, wall_mask


class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = max(1e-3, now - self.prev_time)

        self.integral += error * dt
        self.integral = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self.integral))

        derivative = (error - self.prev_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.prev_error = error
        self.prev_time = now
        return output


steering_pid = PID(KP, KI, KD)


# ============================================================
# LAP LINE DETECTION (reused logic)
# ============================================================
direction = "UNKNOWN"
first_line = False
expected_next = None
mark = 0
lap = 0
line_active = False
line_color = None
line_color_count = {"BLUE": 0, "ORANGE": 0, "NONE": 0}
last_seen_line_color = "NONE"


def process_cross(color):
    global direction, first_line, expected_next, mark, lap

    if not first_line:
        first_line = True
        if color == "ORANGE":
            direction, expected_next = "CLOCKWISE", "BLUE"
        else:
            direction, expected_next = "ANTICLOCKWISE", "ORANGE"
        print(f"Direction: {direction} | Waiting: {expected_next}")
        return

    if color != expected_next:
        return

    if direction == "CLOCKWISE":
        if color == "BLUE":
            mark += 1
            expected_next = "ORANGE"
        else:
            expected_next = "BLUE"
    else:
        if color == "ORANGE":
            mark += 1
            expected_next = "BLUE"
        else:
            expected_next = "ORANGE"

    if mark >= 4:
        lap += 1
        mark = 0
        print(f"===== LAP {lap} =====")


def detect_line_cross(frame):
    global line_active, line_color, last_seen_line_color, line_color_count

    roi = frame[LINE_ROI_START:HEIGHT, :]
    balanced = white_balance(roi)
    _, A, B = cv2.split(cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB))

    hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    mask = cv2.inRange(S, 70, 255)
    mask = cv2.bitwise_and(mask, cv2.inRange(V, 60, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, line_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_color, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_LINE or area <= best_area:
            continue
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [c], -1, 255, -1)
        a_mean = float(cv2.mean(A, mask=blob_mask)[0])
        b_mean = float(cv2.mean(B, mask=blob_mask)[0])
        color = classify_lab(a_mean, b_mean, ["BLUE", "ORANGE"])
        if color is not None:
            detected_color, best_area = color, area

    current = detected_color if detected_color is not None else "NONE"

    if current == last_seen_line_color:
        line_color_count[current] = line_color_count.get(current, 0) + 1
    else:
        last_seen_line_color = current
        line_color_count = {"BLUE": 0, "ORANGE": 0, "NONE": 0}
        line_color_count[current] = 1

    is_stable = line_color_count.get(current, 0) >= STABLE_FRAMES_REQUIRED
    stable_current = current if is_stable else None

    if stable_current not in (None, "NONE") and not line_active:
        line_active = True
        line_color = stable_current
    elif stable_current == "NONE" and line_active and is_stable:
        line_active = False
        process_cross(line_color)
        line_color = None

    return mask


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    load_color_calibration()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    last_error = 0.0

    print("Starting in 3 seconds... place robot on the track.")
    time.sleep(3)

    try:
        drive_forward(DRIVE_SPEED)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            # --- Lane centering ---
            error, wall_mask = get_lane_error(frame)

            if error is None:
                error = last_error  # lost a wall this frame -- hold last correction
            else:
                last_error = error

            steer_output = steering_pid.update(error)
            set_steering(steer_output)

            # --- Lap counting ---
            line_mask = detect_line_cross(frame)

            # --- Stop condition ---
            if lap >= TARGET_LAPS:
                print(f"Target of {TARGET_LAPS} laps reached. Stopping.")
                break

            if SHOW_DEBUG:
                cv2.line(frame, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), (255, 255, 0), 1)
                cv2.putText(frame, f"Lap: {lap}/{TARGET_LAPS}  Mark: {mark}/4",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Error: {error:.1f}  Steer: {steer_output:.1f} deg",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Open Challenge - Camera", frame)
                cv2.imshow("Wall Mask", wall_mask)
                cv2.imshow("Line Mask", line_mask)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    finally:
        stop_motor()
        set_steering(0)
        cap.release()
        cv2.destroyAllWindows()
        print("Motor stopped, cleaned up.")


if __name__ == "__main__":
    main()
