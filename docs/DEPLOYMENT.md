# Raspberry Pi Deployment (4/5, Pi OS Bookworm 64-bit)

## Hardware checklist
- [ ] Pi 4 (4 GB+) or Pi 5 (8 GB preferred), official 5 V/5 A PSU, heatsink/fan
- [ ] Pi Camera Module 3 (CSI) *or* USB camera; SD card ≥ 32 GB, full offline flight tested
- [ ] Measure with `scripts/benchmark_pi.py` — do not quote desktop FPS as Pi performance

## Setup (on the Pi)
```bash
sudo apt update && sudo apt install -y python3-picamera2 python3-opencv libncnn-dev
pip install -r requirements/raspberry_pi.txt
# copy exported model dir + deployment.yaml from desktop:
#   models/survivor_ncnn/  +  configs/deployment.yaml (model.path -> that dir)
python scripts/detect_camera.py --config configs/deployment.yaml
```

## Config notes (`configs/deployment.yaml`)
- `camera.width/height` = **capture** resolution; `model.image_size` = **inference** resolution. Different things.
- `detection.mode`: `full_frame` (fast) vs `tiled` (better for tiny 20 m targets, slower — benchmark both).
- `confirmation_frames: 3` — a target must persist 3 frames before an evidence event fires.
- Telemetry `mock` on bench, `mavlink` on aircraft; vision continues if telemetry drops.

## Verification (needs hardware)
- [ ] CSI frames flowing, `detect_camera.py` draws boxes in non-headless debug
- [ ] Evidence in `outputs/evidence/` (jpg + jsonl + csv)
- [ ] `benchmark_pi.py` numbers recorded (latency, FPS, CPU/RAM/temp)
- [ ] Camera-pull test: inference recovers without restart
