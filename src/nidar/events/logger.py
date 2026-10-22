"""Detection event logger: JSONL + CSV + evidence screenshots.

One JSON object per confirmed detection; CSV mirrors it for spreadsheets.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class EventLogger:
    def __init__(self, out_dir: Path | str = "outputs/evidence",
                 save_evidence: bool = True) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save_evidence = save_evidence
        self.jsonl = self.out_dir / "detections.jsonl"
        self.csv = self.out_dir / "detections.csv"
        if not self.csv.exists():
            with open(self.csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "track_id", "conf", "x1", "y1", "x2", "y2",
                                        "lat", "lon", "coord_kind", "uncertainty_m",
                                        "telemetry_source", "image"])

    def log(self, det, frame: np.ndarray | None = None,
            telemetry=None, projection=None) -> dict[str, Any]:
        ts = time.time()
        img_name = ""
        if self.save_evidence and frame is not None:
            img_name = f"ev_{ts:.0f}_id{det.track_id}.jpg"
            cv2.imwrite(str(self.out_dir / img_name), frame)
        event = {
            "timestamp": ts, "track_id": det.track_id, "conf": det.conf,
            "bbox": [det.x1, det.y1, det.x2, det.y2],
            "lat": getattr(telemetry, "lat", None), "lon": getattr(telemetry, "lon", None),
            "alt_m": getattr(telemetry, "alt_m", None),
            "telemetry_source": getattr(telemetry, "source", None),
            "coord_kind": getattr(projection, "kind", None),
            "target_lat": getattr(projection, "lat", None),
            "target_lon": getattr(projection, "lon", None),
            "uncertainty_m": getattr(projection, "uncertainty_m", None),
            "image": img_name,
        }
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        with open(self.csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [ts, det.track_id, f"{det.conf:.3f}", *[f"{v:.1f}" for v in event["bbox"]],
                 event["target_lat"], event["target_lon"], event["coord_kind"],
                 event["uncertainty_m"], event["telemetry_source"], img_name])
        return event
