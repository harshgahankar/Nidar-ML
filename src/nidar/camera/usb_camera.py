"""USB/webcam camera via OpenCV (works on desktop + Pi with USB camera)."""
from __future__ import annotations

import cv2
import numpy as np

from nidar.camera.base import CameraBase


class USBCamera(CameraBase):
    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080, fps: int = 30) -> None:
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera index {self.index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
