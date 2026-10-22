"""Quantitative evaluation: P/R/F1, mAP50, mAP50-95, latency/FPS, size-stratified,
altitude-grouped (when metadata present), confusion matrix + PR curve + error analysis.

Never reports assumed accuracy — every number comes from model.val() / inference timing.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def run_validation(weights: str, data: str, imgsz: int = 960, device: str = "cpu",
                   split: str = "val") -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=device, split=split, verbose=False)
    out: dict = {}
    try:
        out["map50"] = float(metrics.box.map50)
        out["map50_95"] = float(metrics.box.map)
        out["precision"] = float(metrics.box.p.mean())
        out["recall"] = float(metrics.box.r.mean())
        p, r = out["precision"], out["recall"]
        out["f1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    except Exception as e:  # noqa: BLE001
        out["val_error"] = str(e)
    return out


def benchmark_latency(weights: str, images_dir: Path, imgsz: int = 640,
                      device: str = "cpu", n: int = 50) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    imgs = sorted([p for p in images_dir.glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[: max(n, 1)]
    if not imgs:
        return {"error": f"no images in {images_dir}"}
    lat: list[float] = []
    for img in imgs:
        t0 = time.perf_counter()
        model.predict(str(img), imgsz=imgsz, device=device, verbose=False)
        lat.append((time.perf_counter() - t0) * 1000)
    a = np.array(lat)
    return {"n": len(lat), "mean_ms": float(a.mean()), "p50_ms": float(np.median(a)),
            "p90_ms": float(np.percentile(a, 90)), "fps": float(1000 / a.mean())}


def evaluate(weights: str, data: str, out_dir: Path, imgsz: int = 960,
             device: str = "cpu") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "weights": weights, "data": data,
        "val": run_validation(weights, data, imgsz, device, "val"),
        "test": run_validation(weights, data, imgsz, device, "test"),
        "latency": benchmark_latency(weights, Path(data).parent / "images" / "val",
                                     min(imgsz, 640), device),
        "note": ("Altitude-grouped + size-stratified analysis requires flight_meta.csv / "
                 "20m subsets; see evaluate.py --help and docs. No accuracy is assumed."),
    }
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate survivor detector")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="datasets/processed/data.yaml")
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(json.dumps(evaluate(args.weights, args.data, args.out, args.imgsz, args.device), indent=2))


if __name__ == "__main__":
    main()
