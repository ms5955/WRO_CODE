"""
RPI Detection
LAB+HSV
"""

import cv2
import numpy as np
import time

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CAMERA_INDEX = 0            # only used for the USB-webcam fallback path
WIDTH, HEIGHT = 640, 480
ROI_START = int(HEIGHT * 0.40)
MIN_AREA = 250
MIN_PERIMETER = 80

# ---- HSV ranges -------------------------------------------------------
LOWER_GREEN = np.array([35, 70, 50]);    UPPER_GREEN = np.array([85, 255, 255])
LOWER_RED1 = np.array([0, 120, 80]);     UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 80]);   UPPER_RED2 = np.array([180, 255, 255])
LOWER_MAGENTA = np.array([135, 70, 70]); UPPER_MAGENTA = np.array([170, 255, 255])
LOWER_ORANGE = np.array([8, 130, 100]);  UPPER_ORANGE = np.array([22, 255, 255])
LOWER_BLUE = np.array([95, 100, 60]);    UPPER_BLUE = np.array([130, 255, 255])

# ---- LAB channel gating (A = green-red axis, B = blue-yellow axis) ----
RED_A_MIN = 155          # red -> high A
GREEN_A_MAX = 110        # green -> low A
MAGENTA_A_MIN = 150      # magenta -> high A
ORANGE_B_MIN = 145       # orange -> high B (yellow side)
BLUE_B_MAX = 115         # blue -> low B (blue side)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clahe = cv2.createCLAHE(2.0, (8, 8))

# ----------------------------------------------------------------------
# CAMERA BACKEND SELECTION
# Tries Picamera2 first (correct path for the Raspberry Pi Camera Module
# on both RPi5 and RPi3, running Bullseye/Bookworm with libcamera).
# Falls back to a plain OpenCV/V4L2 capture for USB webcams or systems
# where Picamera2 isn't installed.
# ----------------------------------------------------------------------

class PiCam2Camera:
    """Wraps Picamera2 so it behaves like cv2.VideoCapture (.read() -> ok, BGR frame)."""

    def __init__(self, width, height):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)  # let AE/AWB settle

    def read(self):
        frame = self.picam2.capture_array()  # RGB888
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame

    def release(self):
        self.picam2.stop()


def open_camera(width, height, index=0):
    """Return a camera object exposing .read() and .release()."""
    try:
        cam = PiCam2Camera(width, height)
        print("[camera] Using Picamera2 (Raspberry Pi camera module).")
        return cam
    except Exception as e:
        print(f"[camera] Picamera2 unavailable ({e}); falling back to OpenCV/V4L2.")
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            raise RuntimeError("No camera available (Picamera2 failed and no USB camera found).")
        print("[camera] Using OpenCV VideoCapture (USB webcam).")
        return cap


# ----------------------------------------------------------------------
# IMAGE PROCESSING
# ----------------------------------------------------------------------

def prep(mask):
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def objects(mask, name):
    out = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if area < MIN_AREA or peri < MIN_PERIMETER:
            continue
        eps = 0.003 * peri
        c = cv2.approxPolyDP(c, eps, True)
        x, y, w, h = cv2.boundingRect(c)
        c = c.copy()
        c[:, :, 1] += ROI_START
        y += ROI_START
        out.append({
            "contour": c,
            "color": name,
            "x": x + w // 2,
            "y": y + h // 2,
            "area": area,
            "bottom": y + h
        })
    return out


COLOR_BGR = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "MAGENTA": (255, 0, 255),
    "ORANGE": (0, 140, 255),
    "BLUE": (255, 0, 0),
}

# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------

cap = open_camera(WIDTH, HEIGHT, CAMERA_INDEX)
prev = time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        roi = frame[ROI_START:, :]

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = clahe.apply(L)
        hsv = cv2.cvtColor(
            cv2.GaussianBlur(cv2.cvtColor(cv2.merge((L, A, B)), cv2.COLOR_LAB2BGR), (5, 5), 0),
            cv2.COLOR_BGR2HSV
        )

        red = prep(cv2.bitwise_and(
            cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
                            cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)),
            cv2.inRange(A, RED_A_MIN, 255)))

        green = prep(cv2.bitwise_and(
            cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN),
            cv2.inRange(A, 0, GREEN_A_MAX)))

        magenta = prep(cv2.bitwise_and(
            cv2.inRange(hsv, LOWER_MAGENTA, UPPER_MAGENTA),
            cv2.inRange(A, MAGENTA_A_MIN, 255)))

        orange = prep(cv2.bitwise_and(
            cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE),
            cv2.inRange(B, ORANGE_B_MIN, 255)))

        blue = prep(cv2.bitwise_and(
            cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE),
            cv2.inRange(B, 0, BLUE_B_MAX)))

        objs = (objects(red, "RED") + objects(green, "GREEN") +
                objects(magenta, "MAGENTA") + objects(orange, "ORANGE") +
                objects(blue, "BLUE"))
        objs.sort(key=lambda o: (o["bottom"], o["area"]), reverse=True)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        cv2.line(frame, (0, ROI_START), (WIDTH, ROI_START), (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        for o in objs:
            col = COLOR_BGR[o["color"]]
            cv2.drawContours(frame, [o["contour"]], -1, col, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (o["x"], o["y"]), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(frame, o["color"], (o["x"] - 25, o["y"] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            cv2.putText(frame, f"X:{o['x']} Y:{o['y']}", (o["x"] - 35, o["y"] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"A:{int(o['area'])}", (o["x"] - 35, o["y"] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        cv2.imshow("WRO Vision v4", frame)
        cv2.imshow("Red", red)
        cv2.imshow("Green", green)
        cv2.imshow("Magenta", magenta)
        cv2.imshow("Orange", orange)
        cv2.imshow("Blue", blue)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
