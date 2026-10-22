"""Pi CSI camera via Picamera2 (Raspberry Pi ONLY — import guarded).

Install on Pi: `sudo apt install -y python3-picamera2`. This module must NOT be
imported on Windows/desktop; callers fall back to USBCamera.
"""
from __future__ import annotations

import numpy as np

from nidar.camera.base import CameraBase


class PiCamera(CameraBase):
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30) -> None:
        self.width, self.height, self.fps = width, height, fps
        self._picam = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError("picamera2 not installed (Pi only): sudo apt install -y python3-picamera2") from e
        self._picam = Picamera2()
        config = self._picam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"})
        self._picam.configure(config)
        self._picam.start()

    def read(self) -> np.ndarray | None:
        if self._picam is None:
            return None
        try:
            frame = self._picam.capture_array()
            import cv2

            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception:
            return None  # caller handles graceful recovery

    def close(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            finally:
                self._picam = None


def open_camera(cfg: dict) -> CameraBase:
    """Factory from deployment.yaml camera section."""
    ctype = str(cfg.get("type", "usb")).lower()
    if ctype == "picamera2":
        return PiCamera(cfg.get("width", 1920), cfg.get("height", 1080), cfg.get("fps", 30))
    from nidar.camera.usb_camera import USBCamera

    return USBCamera(cfg.get("usb_index", 0), cfg.get("width", 1920),
                     cfg.get("height", 1080), cfg.get("fps", 30))
