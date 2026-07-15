import cv2
import numpy as np

# ============================================================
# CAMERA SETTINGS
# ============================================================
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480

# Obstacle detection ROI (ignores top 40% of frame)
OBSTACLE_ROI_START = int(HEIGHT * 0.40)

# Line detection ROI (only looks at bottom 25% of frame)
LINE_ROI_START = int(HEIGHT * 0.75)

# ---------------- Obstacle Detection Settings ----------------
RED_SAFE_X = 100
GREEN_SAFE_X = 500
MIN_AREA_OBSTACLE = 250

LOWER_GREEN = np.array([35, 70, 50])
UPPER_GREEN = np.array([85, 255, 255])

LOWER_RED1 = np.array([0, 120, 80])
UPPER_RED1 = np.array([10, 255, 255])

LOWER_RED2 = np.array([170, 120, 80])
UPPER_RED2 = np.array([180, 255, 255])

RED_A_MIN = 155
GREEN_A_MAX = 110

obstacle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ---------------- Line Lap Counter Settings ----------------
LOWER_BLUE = np.array([95, 80, 80])
UPPER_BLUE = np.array([130, 255, 255])

LOWER_ORANGE = np.array([5, 120, 120])
UPPER_ORANGE = np.array([25, 255, 255])

MIN_AREA_LINE = 2500
LINE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Lap counter state (module-level, updated inside process_cross)
direction = "UNKNOWN"
first_line = False
expected_next = None
mark = 0
lap = 0
line_active = False
line_color = None


# ============================================================
# OBSTACLE DETECTION FUNCTIONS
# ============================================================
def clean_mask(mask):
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, obstacle_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, obstacle_kernel)
    return mask


def find_objects(mask, color, roi_start):
    obs = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_OBSTACLE:
            continue

        x, y, w, h = cv2.boundingRect(c)
        y += roi_start

        obs.append({
            "color": color,
            "x": x + w // 2,
            "y": y + h // 2,
            "bottom": y + h,
            "area": area,
            "rect": (x, y, w, h)
        })

    return obs


def detect_obstacles(frame):
    roi = frame[OBSTACLE_ROI_START:, :]

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = clahe.apply(L)
    lab = cv2.merge((L, A, B))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(cv2.GaussianBlur(enhanced, (5, 5), 0), cv2.COLOR_BGR2HSV)

    red_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
        cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    )
    green_hsv = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    red_lab = cv2.inRange(A, RED_A_MIN, 255)
    green_lab = cv2.inRange(A, 0, GREEN_A_MAX)

    red = cv2.bitwise_and(red_hsv, red_lab)
    green = cv2.bitwise_and(green_hsv, green_lab)

    red = clean_mask(red)
    green = clean_mask(green)

    objs = (
        find_objects(red, "RED", OBSTACLE_ROI_START) +
        find_objects(green, "GREEN", OBSTACLE_ROI_START)
    )

    objs.sort(key=lambda o: (o["bottom"], o["area"]), reverse=True)

    return objs, red, green


def draw_obstacles(frame, objs):
    for i, o in enumerate(objs):
        x, y, w, h = o["rect"]
        color = (0, 0, 255) if o["color"] == "RED" else (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{i+1}: {o["color"]}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status = "NO OBSTACLE"
    scol = (255, 255, 255)

    if objs:
        n = objs[0]

        if n["color"] == "RED":
            if n["x"] < RED_SAFE_X:
                status = "RED SAFE"
                scol = (0, 255, 0)
            else:
                status = "RED DANGER -> MOVE RIGHT"
                scol = (0, 0, 255)
        else:
            if n["x"] > GREEN_SAFE_X:
                status = "GREEN SAFE"
                scol = (0, 255, 0)
            else:
                status = "GREEN DANGER -> MOVE LEFT"
                scol = (0, 0, 255)

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, scol, 2)
        cv2.putText(frame, f'Nearest {n["color"]} X={n["x"]} Bottom={n["bottom"]}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.line(frame, (RED_SAFE_X, OBSTACLE_ROI_START), (RED_SAFE_X, HEIGHT), (0, 255, 255), 2)
    cv2.line(frame, (GREEN_SAFE_X, OBSTACLE_ROI_START), (GREEN_SAFE_X, HEIGHT), (0, 255, 255), 2)
    cv2.line(frame, (0, OBSTACLE_ROI_START), (WIDTH, OBSTACLE_ROI_START), (255, 255, 0), 2)


# ============================================================
# LINE LAP COUNTER FUNCTIONS
# ============================================================
def detect_color(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    return cv2.contourArea(c) > MIN_AREA_LINE


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

    current = None
    if detect_color(orange_mask):
        current = "ORANGE"
    elif detect_color(blue_mask):
        current = "BLUE"

    if current is not None and not line_active:
        line_active = True
        line_color = current
    elif current is None and line_active:
        line_active = False
        process_cross(line_color)
        line_color = None

    return blue_mask, orange_mask


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
# MAIN LOOP (single camera feed, single main window)
# ============================================================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        # --- Obstacle Detection ---
        objs, red_mask, green_mask = detect_obstacles(frame)
        draw_obstacles(frame, objs)

        # --- Line Lap Counter ---
        blue_mask, orange_mask = detect_line_cross(frame)
        draw_lap_info(frame)

        # --- Single Main Display (both systems drawn on one frame) ---
        cv2.imshow("WRO Vision System", frame)

        # Optional debug mask windows - comment these out if you only
        # want the single main camera window
        cv2.imshow("Red Mask", red_mask)
        cv2.imshow("Green Mask", green_mask)
        cv2.imshow("Blue Mask", blue_mask)
        cv2.imshow("Orange Mask", orange_mask)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
