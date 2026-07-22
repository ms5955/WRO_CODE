"""
main.py
-------
WRO-style "Open Challenge" driver for the Pi 5 build:
  Pi Camera 3 (wide) -> wall-gap steering + orange/blue lap counting
  TB6612 + JGB37     -> drive
  MG90S              -> steering
  TOF050C (VL6180X)  -> parking distance sensing after lap 3

State machine: DRIVE -> PARKING -> STOPPED

Run:
    python3 main.py

Stop any time with Ctrl+C - motors are cut in the `finally` block.
"""

import time
import sys

from camera import PiCamera
from vision import WallGapSteering, LapLineCounter
from tof_sensor import TofSensor
from motor_control import Steering, DriveMotorTB6612

# ---------------- Tuning knobs ----------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_HFLIP = False   # flip these to match how the camera is physically mounted
CAMERA_VFLIP = False

TARGET_LAPS = 3

BASE_SPEED = 0.45      # 0.0-1.0 cruise speed - tune to your gearing/battery voltage
TURN_SPEED = 0.32      # speed while actively steering hard
PARK_SPEED = 0.25      # slow crawl speed during the parking approach

MAX_STEER_ANGLE = 35
STEER_GAIN = MAX_STEER_ANGLE / (FRAME_WIDTH / 2)   # deg per pixel of offset
STEER_SIGN = 1          # flip to -1 if the servo turns the wrong way
TURN_OFFSET_THRESHOLD = 80

# Parking: once TARGET_LAPS is reached, keep camera-steering but crawl
# forward until the TOF sensor sees the wall closer than this, then stop.
PARK_STOP_DISTANCE_CM = 10.0
PARK_TIMEOUT_S = 8.0    # safety fallback if the TOF never triggers

STATUS_PRINT_EVERY_S = 0.5


def main():
    print("Starting camera...")
    cam = PiCamera(width=FRAME_WIDTH, height=FRAME_HEIGHT, hflip=CAMERA_HFLIP, vflip=CAMERA_VFLIP)

    print("Starting TOF sensor...")
    tof = TofSensor()

    steering_ctrl = Steering(pin=18, center_angle=0, min_angle=-45, max_angle=45)
    motor = DriveMotorTB6612(pwm_pin=13, ain1_pin=5, ain2_pin=6, stby_pin=26)

    wall_gap = WallGapSteering(FRAME_WIDTH, FRAME_HEIGHT)
    lap_counter = LapLineCounter(FRAME_WIDTH, FRAME_HEIGHT)

    print("Waiting for first camera frame...")
    for _ in range(100):
        if cam.get_frame() is not None:
            break
        time.sleep(0.02)
    else:
        print("ERROR: never got a camera frame. Check the ribbon cable / picamera2 install.")
        cam.close()
        return

    state = "DRIVE"
    park_start_time = None
    last_status_time = 0.0
    last_lap_reported = -1

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                motor.stop()
                time.sleep(0.02)
                continue

            steer_info = wall_gap.process(frame)
            lap_info = lap_counter.process(frame)

            if lap_info["lap"] != last_lap_reported:
                print(f"Lap: {lap_info['lap']} / {TARGET_LAPS}  (mark {lap_info['mark']}/4, dir {lap_info['direction']})")
                last_lap_reported = lap_info["lap"]

            # --- Steering: same on every state - keep centered in the lane ---
            steer_angle = STEER_SIGN * STEER_GAIN * steer_info["offset"]
            steering_ctrl.set_angle(steer_angle)

            # --- State transitions ---
            if state == "DRIVE" and lap_info["lap"] >= TARGET_LAPS:
                print("Target lap count reached. Entering parking approach.")
                state = "PARKING"
                park_start_time = time.time()

            # --- Drive / speed per state ---
            if state == "DRIVE":
                speed = TURN_SPEED if abs(steer_info["offset"]) > TURN_OFFSET_THRESHOLD else BASE_SPEED
                motor.forward(speed)

            elif state == "PARKING":
                dist_cm = tof.get_distance_cm()
                timed_out = (time.time() - park_start_time) > PARK_TIMEOUT_S

                if dist_cm <= PARK_STOP_DISTANCE_CM or timed_out:
                    reason = "TOF threshold" if dist_cm <= PARK_STOP_DISTANCE_CM else "timeout"
                    print(f"Parking complete ({reason}, dist={dist_cm:.1f}cm). Stopping.")
                    motor.stop()
                    state = "STOPPED"
                else:
                    motor.forward(PARK_SPEED)

            elif state == "STOPPED":
                motor.stop()
                break

            now = time.time()
            if now - last_status_time > STATUS_PRINT_EVERY_S:
                print(f"[{state}] steer={steer_info['label']:7s} offset={steer_info['offset']:+.0f}px "
                      f"lap={lap_info['lap']} mark={lap_info['mark']}")
                last_status_time = now

            time.sleep(0.02)  # ~50Hz control loop

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Stopping motors and closing devices.")
        motor.stop()
        steering_ctrl.stop()
        cam.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
