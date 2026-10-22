#!/usr/bin/env python3
"""Pi benchmark: model latency, capture time, FPS, CPU/RAM/temp (psutil optional).

Usage (on Pi): python scripts/benchmark_pi.py --config configs/deployment.yaml --seconds 30
NO benchmarks are invented — everything here is measured on-device.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

if __name__ == "__main__":
    import json

    import numpy as np

    ap = argparse.ArgumentParser(description="Benchmark NIDAR on-device performance")
    ap.add_argument("--config", type=Path, default=Path("configs/deployment.yaml"))
    ap.add_argument("--seconds", type=float, default=30)
    args = ap.parse_args()
    from nidar.camera.picamera import open_camera
    from nidar.inference.detector import SurvivorDetector
    from nidar.utils.config import load_yaml

    cfg = load_yaml(args.config)
    det = SurvivorDetector(cfg["model"]["path"], cfg["model"].get("confidence", 0.25),
                           cfg["model"].get("iou", 0.45),
                           cfg["model"].get("image_size", 640), device="cpu")
    try:
        import psutil

        have_psutil = True
    except ImportError:
        have_psutil = False
    cam = open_camera(cfg["camera"])
    cam.open()
    lat, cap_t, t_end = [], [], time.time() + args.seconds
    frames = 0
    try:
        while time.time() < t_end:
            t0 = time.perf_counter()
            frame = cam.read()
            cap_t.append((time.perf_counter() - t0) * 1000)
            if frame is None:
                continue
            t1 = time.perf_counter()
            det.predict_full(frame)
            lat.append((time.perf_counter() - t1) * 1000)
            frames += 1
    finally:
        cam.close()
    rep = {"frames": frames, "capture_ms_mean": float(np.mean(cap_t)) if cap_t else None,
           "infer_ms_mean": float(np.mean(lat)) if lat else None,
           "infer_ms_p90": float(np.percentile(lat, 90)) if lat else None,
           "fps": frames / args.seconds}
    if have_psutil:
        import psutil

        rep.update({"cpu_pct": psutil.cpu_percent(interval=1), "ram_pct": psutil.virtual_memory().percent})
        temps = getattr(psutil, "sensors_temperatures", lambda: {})()
        rep["temp_c"] = next((e.current for v in temps.values() for e in v), None)
    else:
        rep["note"] = "pip install psutil for CPU/RAM/temp on the Pi"
    print(json.dumps(rep, indent=2))
