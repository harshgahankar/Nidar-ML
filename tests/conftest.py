"""Shared pytest fixtures: synthetic YOLO dataset + boxes."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path):
    img_dir = tmp_path / "images"
    lab_dir = tmp_path / "labels"
    img_dir.mkdir()
    lab_dir.mkdir()
    rng = np.random.RandomState(0)
    for i in range(6):
        arr = (rng.rand(64, 64, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(img_dir / f"img_{i}.jpg")
        (lab_dir / f"img_{i}.txt").write_text(f"0 0.5 0.5 0.2 0.2\n")
    return img_dir, lab_dir


@pytest.fixture
def sample_detections():
    from nidar.inference.detector import Detection

    return [Detection(10, 10, 50, 50, 0.9), Detection(12, 12, 52, 52, 0.8),
            Detection(200, 200, 260, 260, 0.7)]
