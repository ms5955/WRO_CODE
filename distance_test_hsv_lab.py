
import cv2
import numpy as np

CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480

ROI_START = int(HEIGHT * 0.40)

RED_SAFE_X = 100
GREEN_SAFE_X = 500
MIN_AREA = 250

# ---------------- HSV Thresholds ----------------
LOWER_GREEN = np.array([35,70,50])
UPPER_GREEN = np.array([85,255,255])

LOWER_RED1 = np.array([0,120,80])
UPPER_RED1 = np.array([10,255,255])

LOWER_RED2 = np.array([170,120,80])
UPPER_RED2 = np.array([180,255,255])

# ---------------- LAB Thresholds (Tune for your camera) ----------------
LOWER_GREEN_LAB = np.array([110,70,145])
UPPER_GREEN_LAB = np.array([255,105,200])

LOWER_RED_LAB = np.array([70,165,145])
UPPER_RED_LAB = np.array([255,210,190])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

def clean(mask):
    mask = cv2.GaussianBlur(mask,(5,5),0)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    return mask

def find_objects(mask,color):
    obs=[]
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area=cv2.contourArea(c)
        if area<MIN_AREA:
            continue

        x,y,w,h=cv2.boundingRect(c)
        y+=ROI_START

        obs.append({
            "color":color,
            "x":x+w//2,
            "y":y+h//2,
            "bottom":y+h,
            "area":area,
            "rect":(x,y,w,h)
        })

    return obs

while True:

    ok,frame=cap.read()
    if not ok:
        break

    frame=cv2.flip(frame,1)

    cv2.line(frame,(0,ROI_START),(WIDTH,ROI_START),(255,255,0),2)

    roi=frame[ROI_START:,:]
    blur=cv2.GaussianBlur(roi,(5,5),0)

    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(blur,cv2.COLOR_BGR2LAB)

    # HSV masks
    red_hsv=cv2.bitwise_or(
        cv2.inRange(hsv,LOWER_RED1,UPPER_RED1),
        cv2.inRange(hsv,LOWER_RED2,UPPER_RED2)
    )
    green_hsv=cv2.inRange(hsv,LOWER_GREEN,UPPER_GREEN)

    # LAB masks
    red_lab=cv2.inRange(lab,LOWER_RED_LAB,UPPER_RED_LAB)
    green_lab=cv2.inRange(lab,LOWER_GREEN_LAB,UPPER_GREEN_LAB)

    # Hybrid masks (HSV AND LAB)
    red=clean(cv2.bitwise_and(red_hsv,red_lab))
    green=clean(cv2.bitwise_and(green_hsv,green_lab))

    objs=find_objects(red,"RED")+find_objects(green,"GREEN")
    objs.sort(key=lambda o:(o["bottom"],o["area"]),reverse=True)

    for i,o in enumerate(objs):
        x,y,w,h=o["rect"]
        clr=(0,0,255) if o["color"]=="RED" else (0,255,0)
        cv2.rectangle(frame,(x,y),(x+w,y+h),clr,2) 
        cv2.putText(frame,f'{i+1}:{o["color"]}',(x,y-5),0,0.6,clr,2)

    status="NO OBSTACLE"; scol=(255,255,255)

    if objs:
        n=objs[0]

        if n["color"]=="RED":
            if n["x"]<RED_SAFE_X:
                status="RED SAFE"; scol=(0,255,0)
            else:
                status="RED DANGER -> MOVE RIGHT"; scol=(0,0,255)
        else:
            if n["x"]>GREEN_SAFE_X:
                status="GREEN SAFE"; scol=(0,255,0)
            else:
                status="GREEN DANGER -> MOVE LEFT"; scol=(0,0,255)

        cv2.putText(frame,status,(10,30),0,0.8,scol,2)
        cv2.putText(frame,
                    f'Nearest {n["color"]} X={n["x"]} Bottom={n["bottom"]}',
                    (10,60),0,0.6,(255,255,255),2)

    cv2.line(frame,(RED_SAFE_X,ROI_START),(RED_SAFE_X,HEIGHT),(0,255,255),2)
    cv2.line(frame,(GREEN_SAFE_X,ROI_START),(GREEN_SAFE_X,HEIGHT),(0,255,255),2)

    cv2.imshow("Distance Test",frame)
    cv2.imshow("Red Mask",red)
    cv2.imshow("Green Mask",green)
    cv2.imshow("LAB Red",red_lab)
    cv2.imshow("LAB Green",green_lab)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
