"""
vision.py
---------
Pure OpenCV vision, ported from your original Limelight snap-script to
run directly against frames from camera.py (no Limelight hardware
needed). Same two pieces, same tuned thresholds:

  - WallGapSteering: scans one lookahead row for the dark wall edges
    on each side, returns the gap center and a steer offset.
  - LapLineCounter: watches a bottom strip of the frame for the
    orange/blue lap-line colors, tracks direction/marks/laps.
"""

import cv2
import numpy as np


class WallGapSteering:
    def __init__(self, width, height, dark_thresh=70, deadzone_px=50, smooth_alpha=0.4):
        self.width = width
        self.height = height
        self.dir_roi_y = int(height * 0.70)   # lookahead row
        self.dark_thresh = dark_thresh        # walls assumed dark on lighter floor
        self.deadzone_px = deadzone_px        # drift (px) before calling it a turn
        self.smooth_alpha = smooth_alpha      # EMA smoothing factor
        self._smoothed_center = None

    def _find_wall_edges(self, row, center_x):
        left_x = 0
        for x in range(center_x, -1, -1):
            if row[x] < self.dark_thresh:
                left_x = x
                break
        right_x = self.width - 1
        for x in range(center_x, self.width):
            if row[x] < self.dark_thresh:
                right_x = x
                break
        return left_x, right_x

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        row = gray[self.dir_roi_y]

        seed_x = int(self._smoothed_center) if self._smoothed_center is not None else self.width // 2
        seed_x = max(0, min(self.width - 1, seed_x))

        left_x, right_x = self._find_wall_edges(row, seed_x)
        raw_center = (left_x + right_x) // 2

        self._smoothed_center = (
            raw_center if self._smoothed_center is None
            else self._smoothed_center * (1 - self.smooth_alpha) + raw_center * self.smooth_alpha
        )

        offset = self._smoothed_center - (self.width / 2)

        if offset < -self.deadzone_px:
            label, code = "LEFT", 1
        elif offset > self.deadzone_px:
            label, code = "RIGHT", 2
        else:
            label, code = "FORWARD", 0

        return {
            "label": label,
            "code": code,
            "path_center_x": int(self._smoothed_center),
            "left_x": left_x,
            "right_x": right_x,
            "offset": offset,
        }


class LapLineCounter:
    # Same tuned HSV ranges as the original script.
    LOWER_BLUE = np.array([106, 150, 40])
    UPPER_BLUE = np.array([130, 255, 155])
    LOWER_ORANGE = np.array([8, 70, 55])
    UPPER_ORANGE = np.array([25, 255, 255])
    MIN_PIXELS_LINE = 700

    def __init__(self, width, height, roi_start_frac=0.75):
        self.width = width
        self.height = height
        self.roi_start = int(height * roi_start_frac)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        self.direction = "UNKNOWN"
        self.first_line = False
        self.expected_next = None
        self.mark = 0
        self.lap = 0
        self.line_active = False
        self.line_color = None

    def _detect(self, mask):
        pixel_count = cv2.countNonZero(mask)
        return pixel_count >= self.MIN_PIXELS_LINE, pixel_count

    def _process_cross(self, color):
        if not self.first_line:
            self.first_line = True
            if color == "ORANGE":
                self.direction = "CLOCKWISE"
                self.expected_next = "BLUE"
            else:
                self.direction = "ANTICLOCKWISE"
                self.expected_next = "ORANGE"
            return

        if color != self.expected_next:
            return

        if self.direction == "CLOCKWISE":
            if color == "BLUE":
                self.mark += 1
                self.expected_next = "ORANGE"
            else:
                self.expected_next = "BLUE"
        else:
            if color == "ORANGE":
                self.mark += 1
                self.expected_next = "BLUE"
            else:
                self.expected_next = "ORANGE"

        if self.mark == 4:
            self.lap += 1
            self.mark = 0

    def process(self, frame):
        roi = frame[self.roi_start:self.height, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)
        orange_mask = cv2.inRange(hsv, self.LOWER_ORANGE, self.UPPER_ORANGE)

        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, self.kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, self.kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, self.kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, self.kernel)

        orange_hit, orange_px = self._detect(orange_mask)
        blue_hit, blue_px = self._detect(blue_mask)

        current = None
        if orange_hit:
            current = "ORANGE"
        elif blue_hit:
            current = "BLUE"

        if current is not None and not self.line_active:
            self.line_active = True
            self.line_color = current
        elif current is None and self.line_active:
            self.line_active = False
            self._process_cross(self.line_color)
            self.line_color = None

        return {
            "direction": self.direction,
            "mark": self.mark,
            "lap": self.lap,
            "orange_px": orange_px,
            "blue_px": blue_px,
        }
