# Architecture

```
RGB camera (CSI/USB @1920x1080 capture)
  -> SurvivorDetector (YOLO pt/onnx/NCNN @640/960 infer, full-frame | tiled+NMS)
    -> IoUTracker (provisional -> confirmed after N frames)
      -> EventLogger (jpg + jsonl + csv) + projection (drone_location | projected_target)
        -> FastAPI ground station (/detections, /status)   [optional, non-blocking]
Telemetry (mock | MAVLink) feeds projection only — vision never blocks on it.
```

- `src/nidar/data/` — importers (visdrone/sard/mannequin/custom), converters (visdrone/coco),
  validation, preparation (extract-frames, dedup, session split, stats)
- `src/nidar/training/` — train (yolo26n→yolo11n fallback), evaluate (P/R/F1, mAP, latency, size/altitude cuts)
- `src/nidar/inference/` — detector, tiling (overlap + global NMS), tracking (IoU + confirmation)
- `src/nidar/camera/` — base/usb/picamera2-guarded; `telemetry/` — mock/mavlink-guarded
- `src/nidar/geolocation/` — ground projection with uncertainty + `drone_location` fallback
- `src/nidar/events/` — evidence logger; `groundstation/` — FastAPI observer only

Design rules: offline-first, headless, graceful camera/telemetry loss, capture≠inference
resolution, no flight-control actions, no alive/dead claims from RGB.
