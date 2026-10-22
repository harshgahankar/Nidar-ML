opencode -s ses_ec750076bffeCtcWa1AG7xhqkt
# NIDAR AI Survivor Detection System

AI-powered aerial survivor detection for the NIDAR competition — RGB drone footage at ~20 m AGL,
YOLO single-class `survivor` (humans + rescue mannequins), trained on GPU (Colab), deployed offline
on Raspberry Pi (NCNN + OpenCV + Picamera2).

> Status: Phases 1–8 implemented. No accuracy/benchmark numbers are claimed — run
> `scripts/evaluate.py` and `scripts/benchmark_pi.py` to measure on your data/hardware.

## Quickstart (Windows/Linux dev machine)

```powershell
pip install -r requirements/training.txt
python -m nidar.utils.config          # env check (add src to PYTHONPATH if needed)
$env:PYTHONPATH="src"; python scripts/prepare_dataset.py --help
pytest tests/ -v
```

## Dataset pipeline

See `docs/DATASETS.md`. Raw downloads are manual (licensing/auth):
- VisDrone-DET → `datasets/raw/visdrone/` (pedestrian+people → survivor)
- SARD (Kaggle) → `datasets/raw/sard/`
- Mannequin (Roboflow) → `datasets/raw/mannequin/`
- Custom 20 m footage → `datasets/raw/custom/`

```powershell
$env:PYTHONPATH="src"
python scripts/prepare_dataset.py --skip-visdrone --skip-sard   # whatever is missing
```

## Training (Google Colab, GPU)

1. Upload repo + processed dataset (or mount Drive), open `notebooks/train_colab.ipynb`.
2. Run all cells (checks GPU, installs ultralytics, trains `yolo26n.pt` → falls back to `yolo11n.pt`).
3. Or locally: `python scripts/train.py --device cpu --epochs 2` (smoke test only).

## Evaluation / inference / export

```powershell
python scripts/evaluate.py --weights runs/nidar_survivor/weights/best.pt
python scripts/detect_video.py --source drone.mp4 --weights runs/.../best.pt --mode tiled --show
python scripts/export_ncnn.py --weights runs/.../best.pt   # then copy to Pi
```

## Raspberry Pi deployment

See `docs/DEPLOYMENT.md`. On the Pi:

```bash
pip install -r requirements/raspberry_pi.txt
python scripts/detect_camera.py --config configs/deployment.yaml
python scripts/benchmark_pi.py --seconds 30   # measured numbers only
```

## Layout

`configs/` · `src/nidar/` (data, training, inference, camera, telemetry, geolocation, events,
groundstation, utils) · `scripts/` · `tests/` · `docs/` · `notebooks/train_colab.ipynb`

An RGB camera cannot tell whether a detected target is alive — the class is `survivor`
(candidate requiring rescue), never "alive/dead".
