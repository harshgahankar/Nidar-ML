#!/usr/bin/env python3
"""Raspberry Pi live inference: CSI/USB camera -> NCNN model -> tracking -> evidence log.

Runs headless + offline. Recovers from transient camera failures; releases
resources on exit. Ground-station API optional via --groundstation.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, "src")
from nidar.camera.picamera import open_camera
from nidar.events.logger import EventLogger
from nidar.geolocation.projection import project_to_ground
from nidar.inference.detector import SurvivorDetector
from nidar.inference.tiling import TileConfig
from nidar.inference.tracking import IoUTracker
from nidar.telemetry.mavlink import open_telemetry
from nidar.utils.config import load_yaml
from nidar.utils.log import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="NIDAR Pi live detection")
    ap.add_argument("--config", type=Path, default=Path("configs/deployment.yaml"))
    ap.add_argument("--groundstation", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    log = setup_logging(cfg.get("system", {}).get("log_level", "INFO"),
                        cfg.get("system", {}).get("log_file"))
    cam_cfg, m_cfg, d_cfg = cfg["camera"], cfg["model"], cfg["detection"]
    det = SurvivorDetector(m_cfg["path"], m_cfg.get("confidence", 0.25),
                           m_cfg.get("iou", 0.45), m_cfg.get("image_size", 640), device="cpu")
    tracker = IoUTracker(cfg.get("tracking", {}).get("iou_threshold", 0.3),
                         cfg.get("tracking", {}).get("max_missing", 10),
                         d_cfg.get("confirmation_frames", 3))
    events = EventLogger(d_cfg.get("evidence_dir", "outputs/evidence"),
                         d_cfg.get("save_evidence", True))
    tiles = TileConfig(d_cfg.get("tile_width", 960), d_cfg.get("tile_height", 960),
                       d_cfg.get("tile_overlap", 0.2), m_cfg.get("image_size", 640))
    try:
        telem = open_telemetry(cfg.get("telemetry", {}))
    except Exception as e:  # noqa: BLE001
        log.warning("Telemetry unavailable (%s); continuing vision-only.", e)
        telem = None

    gs_app = None
    if args.groundstation or cfg.get("groundstation", {}).get("enabled"):
        try:
            from src.nidar.groundstation.app import create_app

            gs_app = create_app()
            log.info("Ground-station API available (serve separately with uvicorn).")
        except Exception as e:  # noqa: BLE001
            log.warning("Ground-station unavailable: %s", e)

    cam = open_camera(cam_cfg)
    try:
        cam.open()
    except Exception as e:  # noqa: BLE001
        log.error("Camera open failed: %s", e)
        raise SystemExit(1)
    log.info("Inference started (mode=%s, headless=%s). Ctrl-C to stop.",
             d_cfg.get("mode"), cfg.get("system", {}).get("headless"))
    fails = 0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                fails += 1
                log.warning("Empty frame (%d); retrying...", fails)
                time.sleep(0.2)
                if fails > 50:
                    log.error("Camera lost; reopening...")
                    try:
                        cam.close(); cam.open(); fails = 0
                    except Exception as e:  # noqa: BLE001
                        log.error("Reopen failed: %s", e)
                        time.sleep(2)
                continue
            fails = 0
            t0 = time.perf_counter()
            dets = det.predict(frame, d_cfg.get("mode", "full_frame"), tiles)
            ms = (time.perf_counter() - t0) * 1000
            dets, newly = tracker.update(dets)
            sample = None
            if telem is not None:
                try:
                    sample = telem.read()
                except Exception:  # noqa: BLE001
                    sample = None
            for d in newly:
                h, w = frame.shape[:2]
                proj = None
                if sample is not None:
                    proj = project_to_ground((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2, w, h,
                                             sample.lat, sample.lon, sample.alt_m, sample.yaw)
                ev = events.log(d, frame if d_cfg.get("save_evidence") else None, sample, proj)
                log.info("CONFIRMED id=%s conf=%.2f %.0fms %s", d.track_id, d.conf, ms, ev.get("coord_kind"))
            if gs_app is not None and hasattr(gs_app, "state"):
                gs_app.state["last"] = {"dets": [d.to_dict() for d in dets], "ms": ms}
            if not cfg.get("system", {}).get("headless", True):
                cv2.imshow("nidar", SurvivorDetector.draw(frame, dets))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if telem is not None:
            try:
                telem.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
