"""Tiled inference: overlapping tiles -> global coords -> NMS merge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nidar.inference.detector import Detection


@dataclass
class TileConfig:
    width: int = 960
    height: int = 960
    overlap: float = 0.2
    infer_size: int = 640


def generate_tiles(image: np.ndarray, cfg: TileConfig):
    """Yield (x0, y0, tile) covering the image with overlap (edges included)."""
    h, w = image.shape[:2]
    sx = max(1, int(cfg.width * (1 - cfg.overlap)))
    sy = max(1, int(cfg.height * (1 - cfg.overlap)))
    xs = sorted(set(list(range(0, max(1, w - cfg.width + 1), sx)) + [max(0, w - cfg.width)]))
    ys = sorted(set(list(range(0, max(1, h - cfg.height + 1), sy)) + [max(0, h - cfg.height)]))
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, image[y0:y0 + min(cfg.height, h), x0:x0 + min(cfg.width, w)].copy()


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / ua if ua > 0 else 0.0


def merge_detections_nms(dets: list[Detection], iou_thr: float = 0.45) -> list[Detection]:
    """Class-agnostic greedy NMS over global-coordinate tile predictions."""
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: list[Detection] = []
    for d in dets:
        if all(_iou(d, k) < iou_thr for k in kept):
            kept.append(d)
    return kept
