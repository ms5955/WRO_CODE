"""
motor_control.py
-----------------
TB6612 motor driver (JGB37 drive motor) + MG90S steering servo control
for the Pi 5. Uses gpiozero with the lgpio backend - Pi 5's RP1 I/O
chip is NOT supported by the old RPi.GPIO library.

Install:
    pip install gpiozero lgpio
"""

from gpiozero import AngularServo, PWMOutputDevice, DigitalOutputDevice


class Steering:
    """MG90S on its own 5V supply; signal wire to a PWM-capable Pi GPIO.
    Common ground between the Pi, the 5V servo buck, and the main
    battery rail is required."""

    def __init__(self, pin=18, center_angle=0, min_angle=-45, max_angle=45):
        self.center_angle = center_angle
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.servo = AngularServo(
            pin,
            min_angle=min_angle,
            max_angle=max_angle,
            min_pulse_width=0.0005,   # 500us
            max_pulse_width=0.0025,   # 2500us - tune if the MG90S buzzes at full lock
        )
        self.center()

    def set_angle(self, angle):
        angle = max(self.min_angle, min(self.max_angle, angle))
        self.servo.angle = angle

    def center(self):
        self.servo.angle = self.center_angle

    def stop(self):
        self.servo.detach()


class DriveMotorTB6612:
    """
    TB6612 channel A wiring (single drive motor = channel A only):
        PWMA -> pwm_pin
        AIN1 -> ain1_pin
        AIN2 -> ain2_pin
        STBY -> stby_pin   (must be driven HIGH to enable the driver)
        AO1/AO2 -> JGB37 motor terminals
        VM  -> motor battery rail (through the TPS5430 if you're
               regulating it down; check your JGB37's rated voltage
               against whatever VM actually ends up being)
        VCC -> 3.3V or 5V logic (per TB6612 datasheet, usually 2.7-5.5V)
    """

    def __init__(self, pwm_pin=13, ain1_pin=5, ain2_pin=6, stby_pin=26):
        self.pwm = PWMOutputDevice(pwm_pin, frequency=1000)
        self.ain1 = DigitalOutputDevice(ain1_pin)
        self.ain2 = DigitalOutputDevice(ain2_pin)
        self.stby = DigitalOutputDevice(stby_pin)
        self.stby.on()  # enable the driver
        self.stop()

    def forward(self, speed):
        """speed: 0.0 - 1.0"""
        speed = max(0.0, min(1.0, speed))
        self.ain1.on()
        self.ain2.off()
        self.pwm.value = speed

    def reverse(self, speed):
        speed = max(0.0, min(1.0, speed))
        self.ain1.off()
        self.ain2.on()
        self.pwm.value = speed

    def stop(self):
        self.pwm.value = 0.0
        self.ain1.off()
        self.ain2.off()

    def disable(self):
        """Cuts STBY - fully disables the driver (motor free-spins)."""
        self.stop()
        self.stby.off()
