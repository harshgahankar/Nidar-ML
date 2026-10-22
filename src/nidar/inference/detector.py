"""Unified detector: full-frame + tiled inference with global-coord NMS merge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nidar.inference.tiling import TileConfig, generate_tiles, merge_detections_nms


@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    conf: float; cls: int = 0; track_id: int | None = None

    def to_xywhn(self, w: int, h: int) -> tuple[float, float, float, float]:
        return ((self.x1 + self.x2) / 2 / w, (self.y1 + self.y2) / 2 / h,
                (self.x2 - self.x1) / w, (self.y2 - self.y1) / h)

    def to_dict(self) -> dict[str, Any]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "conf": self.conf, "cls": self.cls, "track_id": self.track_id}


class SurvivorDetector:
    """Thin wrapper over ultralytics YOLO (pt/onnx) or NCNN-exported model dir."""

    def __init__(self, weights: str | Path, conf: float = 0.25, iou: float = 0.45,
                 imgsz: int = 640, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device

    def predict_full(self, image: np.ndarray) -> list[Detection]:
        res = self.model.predict(image, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out: list[Detection] = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(Detection(x1, y1, x2, y2, float(b.conf[0]), int(b.cls[0])))
        return out

    def predict_tiled(self, image: np.ndarray, tile_cfg: TileConfig | None = None) -> list[Detection]:
        tile_cfg = tile_cfg or TileConfig()
        h, w = image.shape[:2]
        all_dets: list[Detection] = []
        for x0, y0, tile in generate_tiles(image, tile_cfg):
            res = self.model.predict(tile, conf=self.conf, iou=self.iou,
                                     imgsz=tile_cfg.infer_size, device=self.device,
                                     verbose=False)[0]
            th, tw = tile.shape[:2]
            sx, sy = tw / tile_cfg.width, th / tile_cfg.height  # letterbox approx
            for b in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                all_dets.append(Detection(x0 + x1 / max(sx, 1e-6), y0 + y1 / max(sy, 1e-6),
                                          x0 + x2 / max(sx, 1e-6), y0 + y2 / max(sy, 1e-6),
                                          float(b.conf[0]), int(b.cls[0])))
            _ = (w, h)
        return merge_detections_nms(all_dets, self.iou)

    def predict(self, image: np.ndarray, mode: str = "full_frame",
                tile_cfg: TileConfig | None = None) -> list[Detection]:
        if mode == "tiled":
            return self.predict_tiled(image, tile_cfg)
        return self.predict_full(image)

    @staticmethod
    def draw(image: np.ndarray, dets: list[Detection]) -> np.ndarray:
        vis = image.copy()
        for d in dets:
            cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
            label = f"survivor {d.conf:.2f}" + (f" id:{d.track_id}" if d.track_id is not None else "")
            cv2.putText(vis, label, (int(d.x1), max(0, int(d.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return vis
