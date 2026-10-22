#!/usr/bin/env python3
"""Desktop video inference: full-frame or tiled, with IoU tracking + confirmation.

Usage:
  python scripts/detect_video.py --source path/to/drone.mp4 --weights runs/.../best.pt
  python scripts/detect_video.py --source 0 --weights yolo11n.pt --mode tiled --save runs/out.mp4
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, "src")
from nidar.inference.detector import SurvivorDetector
from nidar.inference.tiling import TileConfig
from nidar.inference.tracking import IoUTracker
from nidar.events.logger import EventLogger


def main() -> None:
    ap = argparse.ArgumentParser(description="Run survivor detection on video/webcam")
    ap.add_argument("--source", required=True, help="video file or webcam index")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mode", default="full_frame", choices=["full_frame", "tiled"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirmation-frames", type=int, default=3)
    ap.add_argument("--save", default=None, help="output annotated video path")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        src: str | int = int(args.source)
    except ValueError:
        src = args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")
    det = SurvivorDetector(args.weights, args.conf, 0.45, args.imgsz, args.device)
    tracker = IoUTracker(confirmation_frames=args.confirmation_frames)
    logger = EventLogger(save_evidence=False)
    tiles = TileConfig()
    writer = None
    fps_t0, frames, lat = time.time(), 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        dets = det.predict(frame, args.mode, tiles)
        lat.append((time.perf_counter() - t0) * 1000)
        dets, new = tracker.update(dets)
        frames += 1
        vis = SurvivorDetector.draw(frame, dets)
        fps = frames / max(time.time() - fps_t0, 1e-6)
        cv2.putText(vis, f"FPS {fps:.1f} | conf {len(dets)} prov / {len(tracker.confirmed_tracks())} conf",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if args.save:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
            writer.write(vis)
        if args.show:
            cv2.imshow("nidar", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    import numpy as np

    a = np.array(lat)
    print(f"frames={frames} mean_lat={a.mean():.1f}ms p90={float(__import__('numpy').percentile(a,90)):.1f}ms")


if __name__ == "__main__":
    main()
