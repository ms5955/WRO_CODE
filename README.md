# Open Challenge – Pi 5 Native Build

Runs entirely on the Pi 5: camera-based wall-gap steering + orange/blue
lap counting, TB6612+JGB37 drive, MG90S steering, TOF050C-based parking
after 3 laps. No Limelight hardware required.

## Files
- `camera.py` – Pi Camera 3 continuous-capture wrapper (picamera2).
- `vision.py` – wall-gap steering + lap-line counting, ported 1:1 from
  your original Limelight snap-script logic (same tuned thresholds).
- `tof_sensor.py` – TOF050C (VL6180X) distance readout over I2C.
- `motor_control.py` – TB6612 drive motor + MG90S steering, gpiozero/lgpio.
- `main.py` – state machine: DRIVE → PARKING → STOPPED.

## Install (on the Pi 5, Raspberry Pi OS Bookworm or later)
```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv i2c-tools
pip install gpiozero lgpio adafruit-circuitpython-vl6180x adafruit-blinka --break-system-packages
sudo raspi-config   # Interface Options -> enable I2C
```
Verify the TOF sensor is visible before running anything:
```bash
i2cdetect -y 1        # should show a device at address 0x29
```

## Wiring (edit pin numbers in the code if yours differ)
| Function                  | Pi 5 GPIO |
|----------------------------|-----------|
| Steering servo (MG90S)     | 18        |
| TB6612 PWMA                | 13        |
| TB6612 AIN1                | 5         |
| TB6612 AIN2                | 6         |
| TB6612 STBY                | 26        |
| TOF050C / MPU6050 I2C      | SDA/SCL (via your logic-level shifter) |

Power notes for your specific BOM:
- **MG90S** runs off the separate 5V buck, not the Pi's 5V rail — share
  ground with the Pi and the main battery.
- **TB6612 VM** (motor rail) comes from your battery/TPS5430 side, **VCC**
  (logic) from 3.3V/5V — check the JGB37's rated voltage against whatever
  VM ends up being after your buck stages.
- **SY8205** feeds the Pi 5 itself — make sure it's rated for the Pi 5's
  peak current draw (camera + USB + compute can spike over 3A on a Pi 5).
- **TOF050C**: this is a VL6180X breakout. Its reliable range in practice
  is closer to 0–20cm even though it's marketed to 50cm — plan your
  `PARK_STOP_DISTANCE_CM` and parking approach geometry around that, not
  the marketing number.

## Required tuning before you trust it on the track
1. **`STEER_SIGN`** in `main.py` — run on blocks first, confirm the servo
   turns the correct way relative to which wall is closer.
2. **`MAX_STEER_ANGLE` / servo min/max** — set to your steering linkage's
   real mechanical limits.
3. **`BASE_SPEED` / `TURN_SPEED` / `PARK_SPEED`** — start low, raise once
   steering direction is confirmed.
4. **`DARK_THRESH`, `DEADZONE_PX`** in `vision.py`'s `WallGapSteering` —
   still venue-lighting dependent, same as before.
5. **`LOWER_BLUE/UPPER_BLUE`, `LOWER_ORANGE/UPPER_ORANGE`** in
   `LapLineCounter` — recheck against your actual tape/lighting; camera
   sensor and lens differ from whatever the thresholds were tuned on
   before.
6. **`PARK_STOP_DISTANCE_CM` / `PARK_TIMEOUT_S`** — tune against your
   actual parking bay depth and approach angle.
7. **`CAMERA_HFLIP` / `CAMERA_VFLIP`** — set based on how the camera
   ends up mounted (upside-down mounts are common for keeping the wire
   routing clean).

## Not wired in yet
- **MPU6050 gyro** — you're navigating on camera only for now, so the
  gyro isn't used. If wall-gap steering turns out too twitchy on fast
  straights, adding gyro-based heading hold (drive straight by holding
  yaw rate near zero) is a natural next step — say the word.
- **Obstacle Challenge logic** (red/green pillar avoidance) — this is
  Open Challenge only.
