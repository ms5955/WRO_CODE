"""
tof_sensor.py
-------------
Wrapper around the TOF050C module - this is a breakout board for ST's
VL6180X time-of-flight sensor (I2C, ~0-20cm reliable range in practice,
marketed up to 50cm under ideal conditions with a bright flat target).
Used here for the end-of-run parking maneuver.

Install:
    pip install adafruit-circuitpython-vl6180x adafruit-blinka

Wiring: SDA/SCL -> Pi I2C pins (through your 5V<->3.3V level shifter
if your specific breakout's logic is 5V - the VL6180X die itself is
3.3V logic). Default I2C address is 0x29.
"""

import board
import busio
import adafruit_vl6180x


class TofSensor:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_vl6180x.VL6180X(i2c)

    def get_distance_mm(self):
        """Range in mm. Trust readings under ~200mm most; beyond that
        the VL6180X gets noisy - don't rely on it past ~20cm."""
        return self.sensor.range

    def get_distance_cm(self):
        return self.get_distance_mm() / 10.0
