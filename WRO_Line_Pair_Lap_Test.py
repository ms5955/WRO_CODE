
"""
WRO Future Engineers 2026
Line Pair Lap Counter Test

Rules
-----
- First detected line decides direction.
    ORANGE -> CLOCKWISE
    BLUE   -> ANTICLOCKWISE

Clockwise:
    Orange -> Blue = Mark

Anticlockwise:
    Blue -> Orange = Mark

4 Marks = 1 Lap
"""

import cv2
import numpy as np

# =======================
# Camera
# =======================
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
ROI_START = int(HEIGHT * 0.75)

# =======================
# HSV
# =======================
LOWER_BLUE   = np.array([95,80,80])
UPPER_BLUE   = np.array([130,255,255])

LOWER_ORANGE = np.array([5,120,120])
UPPER_ORANGE = np.array([25,255,255])

MIN_AREA = 2500
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# =======================
# Variables
# =======================
direction = "UNKNOWN"
first_line = False

expected_next = None
mark = 0
lap = 0

line_active = False
line_color = None

# =======================
# Functions
# =======================

def detect_color(mask):
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours,key=cv2.contourArea)
    return cv2.contourArea(c) > MIN_AREA

def process_cross(color):
    global direction, first_line, expected_next, mark, lap

    if not first_line:
        first_line = True

        if color == "ORANGE":
            direction = "CLOCKWISE"
            expected_next = "BLUE"
        else:
            direction = "ANTICLOCKWISE"
            expected_next = "ORANGE"

        print(f"Direction : {direction}")
        print(f"Waiting   : {expected_next}")
        return

    if color != expected_next:
        return

    if direction == "CLOCKWISE":
        if color == "BLUE":
            mark += 1
            print(f"Mark {mark}")
            expected_next = "ORANGE"
        else:
            expected_next = "BLUE"

    else:
        if color == "ORANGE":
            mark += 1
            print(f"Mark {mark}")
            expected_next = "BLUE"
        else:
            expected_next = "ORANGE"

    if mark == 4:
        lap += 1
        mark = 0
        print(f"===== LAP {lap} =====")

# =======================
# Main
# =======================

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT)

while True:

    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.rotate(frame, -1)

    roi = frame[ROI_START:HEIGHT,:]
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

    blueMask = cv2.inRange(hsv,LOWER_BLUE,UPPER_BLUE)
    orangeMask = cv2.inRange(hsv,LOWER_ORANGE,UPPER_ORANGE)

    blueMask = cv2.morphologyEx(blueMask,cv2.MORPH_OPEN,KERNEL)
    blueMask = cv2.morphologyEx(blueMask,cv2.MORPH_CLOSE,KERNEL)

    orangeMask = cv2.morphologyEx(orangeMask,cv2.MORPH_OPEN,KERNEL)
    orangeMask = cv2.morphologyEx(orangeMask,cv2.MORPH_CLOSE,KERNEL)

    current = None

    if detect_color(orangeMask):
        current = "ORANGE"
    elif detect_color(blueMask):
        current = "BLUE"

    if current is not None and not line_active:
        line_active = True
        line_color = current

    elif current is None and line_active:
        line_active = False
        process_cross(line_color)
        line_color = None

    cv2.line(frame,(0,ROI_START),(WIDTH,ROI_START),(0,255,0),2)

    cv2.putText(frame,f"Direction : {direction}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(frame,f"Next : {expected_next}",(10,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
    cv2.putText(frame,f"Mark : {mark}/4",(10,90),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2)
    cv2.putText(frame,f"Lap : {lap}",(10,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.imshow("Camera",frame)
    cv2.imshow("ROI",roi)
    cv2.imshow("Blue Mask",blueMask)
    cv2.imshow("Orange Mask",orangeMask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
