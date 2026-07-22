"""
camera.py
---------
Continuous-capture wrapper around the Raspberry Pi Camera Module 3
(wide) via picamera2. Captures in a background thread and always
exposes the latest frame, so the control loop is never blocked on I/O.

Install:
    sudo apt install -y python3-picamera2
"""

import threading
import time

from picamera2 import Picamera2


class PiCamera:
    def __init__(self, width=640, height=480, hflip=False, vflip=False):
        self.width = width
        self.height = height
        self.picam2 = Picamera2()

        config_kwargs = {"main": {"size": (width, height), "format": "BGR888"}}
        if hflip or vflip:
            from libcamera import Transform
            config_kwargs["transform"] = Transform(hflip=hflip, vflip=vflip)

        config = self.picam2.create_video_configuration(**config_kwargs)
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)  # let auto-exposure/white-balance settle

        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            frame = self.picam2.capture_array("main")
            with self._lock:
                self._latest_frame = frame

    def get_frame(self):
        """Returns the most recent frame (BGR, numpy array) or None."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.picam2.stop()
