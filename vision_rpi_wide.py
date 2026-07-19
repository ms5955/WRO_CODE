import cv2
import numpy as np

# ============================================================
# LIMELIGHT PYTHON SNAP SCRIPT - Open Challenge (simple)
# ============================================================
# Limelight calls runPipeline(image, llrobot) on every frame.
# Must return exactly three values:
#
#   largestContour : contour for the built-in crosshair overlay
#   image          : annotated frame streamed to the dashboard
#   llpython       : up to 8 numbers sent to the robot over
#                     NetworkTables (limelight.getpythoninputs())
# ============================================================

# ---------------- Frame geometry (set from incoming image) ----
WIDTH = 640
HEIGHT = 480
LINE_ROI_START = int(HEIGHT * 0.75)   # for lap-line detection
DIR_ROI_Y = int(HEIGHT * 0.70)        # single lookahead row for the arrow

# ---------------- Direction Arrow Settings --------------------
# Walls are assumed dark (black tape/boundary) on a lighter
# floor. If your floor/wall contrast is inverted, flip the
# comparison in find_wall_edges() from "< " to "> ".
DARK_THRESH = 70

# How far the path-center has to drift from frame-center (in
# pixels) before we call it a turn instead of straight. Lower =
# more sensitive / twitchier. Raise if it flickers LEFT/RIGHT on
# a straight section.
DEADZONE_PX = 50

# Smoothing so the arrow doesn't flicker frame to frame.
SMOOTH_ALPHA = 0.4
_smoothed_center = None  # persists between frames

# ---------------- Line Lap Counter Settings --------------------
# Real line samples: hue ~111-117, V ~108-141.
# Floor samples (not on the line): hue ~96-104, V ~163-249.
# The old range only bounded hue/saturation with no V ceiling, so
# it matched the bright floor just as happily as the darker line.
# Requiring BOTH a higher hue AND a lower V excludes the floor
# with margin on both sides.
LOWER_BLUE = np.array([106, 150, 40])
UPPER_BLUE = np.array([130, 255, 155])

# Loosened S/V floor - real orange tape under dim gym/venue lighting
# was sampling as low as S~90, V~72, which the old [120, 80] floor
# was rejecting outright. Hue stays tight (8-25) since that's still
# a huge gap from blue's hue (~95-130), so this can't bleed into
# blue detection.
LOWER_ORANGE = np.array([8, 70, 55])
UPPER_ORANGE = np.array([25, 255, 255])

# Detection is now based on TOTAL matching pixel count in the ROI,
# not the single largest contour - this makes it robust to a line
# that's diagonal, broken into pieces by noise, or clipped by the
# edge of the frame. Tune this to roughly the pixel count you'd
# expect from the thinnest sliver of line you still want to count.
MIN_PIXELS_LINE = 700
LINE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Set True while calibrating at the venue: tints matched pixels on
# the live feed and prints raw pixel counts, so you can watch the
# mask react in real time and re-tune the ranges above if needed.
DEBUG_LINE_MASKS = True

# ---------------- Persistent state (module-level globals) -----
direction = "UNKNOWN"     # lap direction: CLOCKWISE / ANTICLOCKWISE
first_line = False
expected_next = None
mark = 0
lap = 0
line_active = False
line_color = None
_geometry_initialized = False


# ============================================================
# DIRECTION ARROW (wall-based)
# ============================================================
def find_wall_edges(row, width, center_x):
    """Scan outward from center_x until a dark wall pixel is hit
    on each side. Falls back to the frame edge if no wall found."""
    left_x = 0
    for x in range(center_x, -1, -1):
        if row[x] < DARK_THRESH:
            left_x = x
            break

    right_x = width - 1
    for x in range(center_x, width):
        if row[x] < DARK_THRESH:
            right_x = x
            break

    return left_x, right_x


def detect_direction(frame):
    """Looks at one lookahead row, finds the open gap between the
    left and right walls, and returns (arrow, label, offset_px)."""
    global _smoothed_center

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    row = gray[DIR_ROI_Y]
    seed_x = int(_smoothed_center) if _smoothed_center is not None else WIDTH // 2
    seed_x = max(0, min(WIDTH - 1, seed_x))

    left_x, right_x = find_wall_edges(row, WIDTH, seed_x)
    raw_center = (left_x + right_x) // 2

    _smoothed_center = (
        raw_center if _smoothed_center is None
        else _smoothed_center * (1 - SMOOTH_ALPHA) + raw_center * SMOOTH_ALPHA
    )

    offset = _smoothed_center - (WIDTH / 2)

    if offset < -DEADZONE_PX:
        arrow, label, code = "\u2190", "LEFT", 1
    elif offset > DEADZONE_PX:
        arrow, label, code = "\u2192", "RIGHT", 2
    else:
        arrow, label, code = "\u2191", "FORWARD", 0

    return arrow, label, code, int(_smoothed_center), left_x, right_x, offset


def draw_dashed_line(frame, pt1, pt2, color, thickness=2, dash_len=12):
    """Draws a dashed line from pt1 to pt2 (used for the side wall
    indicators, like the mirror lines in a driving HUD)."""
    dist = max(1.0, np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    dashes = max(1, int(dist / dash_len))
    for i in range(dashes):
        r1 = i / dashes
        r2 = (i + 0.5) / dashes
        p1 = (int(pt1[0] + (pt2[0] - pt1[0]) * r1), int(pt1[1] + (pt2[1] - pt1[1]) * r1))
        p2 = (int(pt1[0] + (pt2[0] - pt1[0]) * r2), int(pt1[1] + (pt2[1] - pt1[1]) * r2))
        cv2.line(frame, p1, p2, color, thickness)


def draw_pin(frame, tip, color, size=12):
    """Draws a map-pin marker whose point touches `tip` (the target
    path-center point on the road ahead)."""
    cx, cy = tip[0], tip[1] - size * 2
    cv2.circle(frame, (cx, cy), size, color, -1)
    cv2.circle(frame, (cx, cy), size, (255, 255, 255), 2)
    pts = np.array([
        [cx - size // 2, cy + size - 2],
        [cx + size // 2, cy + size - 2],
        [tip[0], tip[1]]
    ], np.int32)
    cv2.fillPoly(frame, [pts], color)


def draw_direction_hud(frame, label, code, path_center_x, left_x, right_x):
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)

    color = GREEN if code == 0 else RED

    # "Driver" anchor point - bottom center of the frame, like the
    # dashboard position in the reference photo.
    base = (WIDTH // 2, HEIGHT - 15)
    target = (path_center_x, DIR_ROI_Y)
    left_pt = (left_x, DIR_ROI_Y)
    right_pt = (right_x, DIR_ROI_Y)

    # Dashed lines out to each detected wall (mirrors the side
    # awareness lines in the reference image)
    draw_dashed_line(frame, base, left_pt, RED, 2)
    draw_dashed_line(frame, base, right_pt, RED, 2)
    cv2.circle(frame, left_pt, 10, RED, 2)
    cv2.circle(frame, right_pt, 10, RED, 2)

    # Solid steering line from the car's position up to the target,
    # tilts naturally with the curve instead of snapping to 3 fixed
    # angles - this IS the "direction the car is moving".
    cv2.line(frame, base, target, color, 5)
    draw_pin(frame, target, color)

    cv2.putText(frame, label, (base[0] - 55, base[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Wall gap: {left_x}-{right_x}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)


# ============================================================
# LINE LAP COUNTER FUNCTIONS
# ============================================================
def detect_color(mask):
    """Total matching pixels in the mask, not the single largest
    contour - robust regardless of the line's angle, whether it's
    fragmented, or clipped by the frame edge."""
    pixel_count = cv2.countNonZero(mask)
    return pixel_count >= MIN_PIXELS_LINE, pixel_count


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

    if mark == 4:
        lap += 1
        mark = 0


def detect_line_cross(frame):
    global line_active, line_color

    roi = frame[LINE_ROI_START:HEIGHT, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    orange_mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, LINE_KERNEL)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, LINE_KERNEL)

    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, LINE_KERNEL)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, LINE_KERNEL)

    orange_hit, orange_px = detect_color(orange_mask)
    blue_hit, blue_px = detect_color(blue_mask)

    if DEBUG_LINE_MASKS:
        # Tint matched pixels directly on the live feed so you can
        # see exactly what the mask is (and isn't) picking up.
        tint = roi.copy()
        tint[orange_mask > 0] = (0, 140, 255)   # orange tint (BGR)
        tint[blue_mask > 0] = (255, 120, 0)     # blue tint (BGR)
        cv2.addWeighted(tint, 0.5, roi, 0.5, 0, roi)
        cv2.putText(frame, f"Orange px:{orange_px}  Blue px:{blue_px}",
                    (10, LINE_ROI_START - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    current = None
    if orange_hit:
        current = "ORANGE"
    elif blue_hit:
        current = "BLUE"

    if current is not None and not line_active:
        line_active = True
        line_color = current
    elif current is None and line_active:
        line_active = False
        process_cross(line_color)
        line_color = None


def draw_lap_info(frame):
    cv2.line(frame, (0, LINE_ROI_START), (WIDTH, LINE_ROI_START), (0, 255, 0), 2)
    cv2.putText(frame, f"Direction : {direction}", (10, HEIGHT - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Next : {expected_next}", (10, HEIGHT - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Mark : {mark}/4", (10, HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(frame, f"Lap : {lap}", (10, HEIGHT - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# ============================================================
# LIMELIGHT ENTRY POINT
# ============================================================
def process_frame(image):
    global WIDTH, HEIGHT, LINE_ROI_START, DIR_ROI_Y
    global direction, mark, lap, _geometry_initialized

    if not _geometry_initialized:
        HEIGHT, WIDTH = image.shape[:2]
        LINE_ROI_START = int(HEIGHT * 0.75)
        DIR_ROI_Y = int(HEIGHT * 0.70)
        _geometry_initialized = True

    # --- Direction arrow ---
    arrow, label, dir_code, path_center_x, left_x, right_x, offset = detect_direction(image)
    draw_direction_hud(image, label, dir_code, path_center_x, left_x, right_x)

    # --- Lap counter ---
    detect_line_cross(image)
    draw_lap_info(image)

    # --- Crosshair overlay: just show the detected path center point ---
    largestContour = np.array([[[path_center_x, DIR_ROI_Y]]])

    # --- Pack results for the robot ---
    lap_direction_code = 0
    if direction == "CLOCKWISE":
        lap_direction_code = 1
    elif direction == "ANTICLOCKWISE":
        lap_direction_code = 2

    llpython = [
        dir_code,            # [0] 0=FORWARD, 1=LEFT, 2=RIGHT
        path_center_x,        # [1] detected path center X (pixels)
        int(offset),            # [2] offset from frame center (pixels)
        lap_direction_code,       # [3] 0=unknown, 1=CW, 2=ACW
        mark,                      # [4] current mark count (0-4)
        lap,                        # [5] completed lap count
        0, 0                          # [6],[7] reserved
    ]

    robot_data = {
        "direction": dir_code,
        "path_center_x": path_center_x,
        "offset": int(offset),
        "lap_direction": lap_direction_code,
        "mark": mark,
        "lap": lap
    }
    return image, robot_data

if __name__ == "__main__":
    from picamera2 import Picamera2
    import time

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640,480), "format":"RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    try:
        while True:
            frame = picam2.capture_array()
            frame, robot_data = process_frame(frame)
            cv2.imshow("RPi Vision", frame)
            print(robot_data, end="\r")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

