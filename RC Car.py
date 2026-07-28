"""
RC Car - Steering + Throttle Control
--------------------------------------
Controls:
  LEFT arrow  -> steer left
  RIGHT arrow -> steer right
  c           -> re-center steering
  w           -> forward (increase speed)
  r           -> reverse (increase speed)
  s           -> brake (reduce speed)
  x           -> emergency stop
  q / ESC     -> quit
"""

import cv2
import numpy as np
import RPi.GPIO as GPIO
from time import sleep

DIR_PIN = 17
PWM_PIN = 27
SERVO_PIN = 25

CENTER = 95
LEFT = 70
RIGHT = 125
STEER_STEP = 5

THROTTLE_STEP = 8
THROTTLE_MAX = 45
THROTTLE_MIN = 0

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(PWM_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

motor_pwm = GPIO.PWM(PWM_PIN, 1000)
motor_pwm.start(0)

servo_pwm = GPIO.PWM(SERVO_PIN, 50)
servo_pwm.start(0)


def steer(angle):
    angle = max(LEFT, min(RIGHT, angle))
    duty = 2.5 + (angle / 180.0) * 10.0
    servo_pwm.ChangeDutyCycle(duty)
    sleep(0.05)
    servo_pwm.ChangeDutyCycle(0)
    return angle


def forward(speed):
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(speed)


def reverse(speed):
    GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(speed)


def stop():
    motor_pwm.ChangeDutyCycle(0)


def main():
    steer_angle = CENTER
    steer(steer_angle)

    throttle = 0
    direction = "forward"

    stop()

    print("[INFO] Controls:")
    print("LEFT/RIGHT = Steering")
    print("w = Forward")
    print("r = Reverse")
    print("s = Brake")
    print("x = Emergency Stop")
    print("c = Center Steering")
    print("q or ESC = Quit")

    window_name = "Car Control"
    cv2.namedWindow(window_name)

    LEFT_KEYS = (81, 2424832, 65361, 63234)
    RIGHT_KEYS = (83, 2555904, 65363, 63235)

    try:
        while True:
            panel = np.zeros((240, 540, 3), dtype="uint8")

            cv2.putText(panel, f"Steering : {steer_angle}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(panel, f"Throttle : {throttle}", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(panel, f"Direction: {direction}", (15,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.putText(panel, "Arrows=Steer  W=Forward  R=Reverse", (15,170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            cv2.putText(panel, "S=Brake  X=Stop  C=Center  Q=Quit", (15,195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

            cv2.imshow(window_name, panel)

            key = cv2.waitKeyEx(30)

            if key == -1:
                continue

            if key in (ord('q'), 27):
                break

            elif key == ord('c'):
                steer_angle = steer(CENTER)

            elif key == ord('w'):
                direction = "forward"
                throttle = min(THROTTLE_MAX, throttle + THROTTLE_STEP)
                forward(throttle)

            elif key == ord('r'):
                direction = "reverse"
                throttle = min(THROTTLE_MAX, throttle + THROTTLE_STEP)
                reverse(throttle)

            elif key == ord('s'):
                throttle = max(THROTTLE_MIN, throttle - THROTTLE_STEP)

                if throttle == 0:
                    stop()
                else:
                    if direction == "forward":
                        forward(throttle)
                    else:
                        reverse(throttle)

            elif key == ord('x'):
                throttle = 0
                stop()

            elif key in LEFT_KEYS:
                steer_angle = steer(steer_angle - STEER_STEP)

            elif key in RIGHT_KEYS:
                steer_angle = steer(steer_angle + STEER_STEP)

    finally:
        cv2.destroyAllWindows()
        stop()
        steer(CENTER)
        sleep(0.2)
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
