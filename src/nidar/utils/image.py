"""Image helpers shared by dataset tooling and inference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image_bgr(path: str | Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash for near-duplicate detection (grayscale)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_LINEAR)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
