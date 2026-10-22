# NIDAR — Aerial Survivor Detection

Detect people (survivors / rescue mannequins) in aerial RGB drone footage (~20 m AGL) with a
single-class YOLO model, and run it on a laptop, a drone companion computer, or a Raspberry Pi.

- **Class:** `survivor` (one class, box + confidence). An RGB camera cannot tell if someone is
  alive — a detection is a *candidate requiring rescue*, never a medical assessment.
- **Model:** YOLO26n, 640 px input, trained on VisDrone + SARD + mannequin data.
- **Shipped weights:** `models/survivor_best.pt` (PyTorch) and `models/survivor_ncnn/` (NCNN, for Pi).

## Measured performance

Measured on the held-out processed test split (2,621 images) and a desktop RTX 3050 Laptop GPU.
Re-run `scripts/evaluate.py` on your own data/hardware — do not trust these numbers for your site.

| Split | Images | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| val | 1,286 | 0.685 | 0.482 | 0.544 | 0.237 |
| **test (unseen)** | 2,621 | 0.629 | 0.403 | **0.441** | 0.185 |

Speed: ~26 ms/image inference at 640 on the RTX 3050 (~38 FPS); ~19 FPS end-to-end on video
(decode + tracking + drawing included). Raspberry Pi numbers are **not** assumed — measure with
`scripts/benchmark_pi.py`.

## 1. Install (laptop / desktop)

Requires **Python 3.10+** and Git. CPU works; a CUDA GPU is much faster.

```bash
git clone https://github.com/harshgahankar/Nidar-ML.git
cd Nidar-ML
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -e .
```

Optional GPU acceleration (pick the wheel matching your driver at pytorch.org):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

`pip install -e .` installs `ultralytics`, OpenCV, NumPy, PyYAML, Pillow and the `nidar` package,
so no `PYTHONPATH` setup is needed. Run commands from the repository root.

## 2. Detect on your own media

### Images or a folder of images
```bash
yolo predict model=models/survivor_best.pt source="path/to/photo.jpg" imgsz=640 conf=0.25 save=True
yolo predict model=models/survivor_best.pt source="path/to/folder" imgsz=640 conf=0.25 save=True
```
Annotated copies are written to `runs/detect/predict/` (same filenames).

### Video file
```bash
yolo predict model=models/survivor_best.pt source="path/to/drone.mp4" imgsz=640 conf=0.25 save=True
```
This saves an annotated `.mp4` next to the input copy in `runs/detect/predict/`.

### Webcam / live camera (desktop)
```bash
python scripts/detect_video.py --source 0 --weights models/survivor_best.pt --show
```
`--source 0` is the default webcam index. Add `--save runs/out.mp4` to record.

### Tiled mode (better for tiny targets, slower)
```bash
python scripts/detect_video.py --source "path/to/drone.mp4" --weights models/survivor_best.pt --mode tiled --save runs/tiled.mp4
```
Tiles the frame, runs the model on each tile, and merges with NMS.

### Python API
```python
import sys; sys.path.insert(0, "src")
import cv2
from nidar.inference.detector import SurvivorDetector

det = SurvivorDetector("models/survivor_best.pt", conf=0.25, iou=0.45, imgsz=640, device="cpu")
frame = cv2.imread("photo.jpg")
dets = det.predict(frame, mode="full_frame")   # or mode="tiled"
for d in dets:
    print(d.x1, d.y1, d.x2, d.y2, d.conf, d.track_id)
cv2.imwrite("out.jpg", SurvivorDetector.draw(frame, dets))
```

## 3. Run on a Raspberry Pi (live camera)

Works with a **CSI camera (Picamera2)** or a **USB camera** — that's the `camera.type` setting.

On the Pi (Pi OS Bookworm 64-bit, Pi 4 4 GB+ / Pi 5 recommended):

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv
git clone https://github.com/harshgahankar/Nidar-ML.git
cd Nidar-ML
python3 -m venv --system-site-packages .venv    # keeps apt picamera2 visible
source .venv/bin/activate
pip install -r requirements/raspberry_pi.txt
pip install ultralytics ncnn                    # needed to load the NCNN model via YOLO()
```

Edit `configs/deployment.yaml`:

```yaml
model:
  path: models/survivor_ncnn     # already points here in the repo
  image_size: 640
camera:
  type: picamera2                # or: usb  (+ usb_index: 0)
  width: 1920
  height: 1080
detection:
  mode: full_frame               # or tiled (better on tiny targets, slower)
  confirmation_frames: 3
system:
  headless: false                # true when running over SSH without a display
```

Run:

```bash
python scripts/detect_camera.py --config configs/deployment.yaml
python scripts/benchmark_pi.py --config configs/deployment.yaml --seconds 30
```

Confirmed detections are logged to `outputs/evidence/` (JPEG + JSONL) after persisting
`confirmation_frames` consecutive frames. Telemetry defaults to `mock`; MAVLink is optional and
vision keeps running if telemetry drops.

## 4. Tune for your footage

| Setting | Where | Effect |
|---|---|---|
| `conf` / `confidence: 0.25` | CLI / `configs/deployment.yaml` | Lower → more detections, more false positives |
| `imgsz` / `image_size: 640` | CLI / config | Higher → smaller objects found, slower |
| `mode: tiled` | CLI `--mode tiled` / config | Best for tiny 20 m targets, costs speed |
| `confirmation_frames: 3` | `configs/deployment.yaml` | Frames a target must persist before an event fires |

**Expect a domain gap on your own drone**: the model generalizes modestly (test mAP50 0.441).
For production, fine-tune on a few hundred labeled frames from your aircraft/altitude — see below.

## 5. Retrain / fine-tune (optional)

```bash
# 1. Put datasets in datasets/raw/ (see docs/DATASETS.md for licenses and links)
# 2. Build the unified dataset
python scripts/prepare_dataset.py
# 3. Train (configs/training_local.yaml is tuned for a 6 GB GPU)
python scripts/train.py --config configs/training_local.yaml
#    resume an interrupted run:
python scripts/train.py --config configs/training_local.yaml --resume runs/nidar_survivor_local/weights/last.pt
# 4. Evaluate and export for the Pi
python scripts/evaluate.py --weights runs/nidar_survivor_local/weights/best.pt --device 0 --imgsz 640
python scripts/export_ncnn.py --weights runs/nidar_survivor_local/weights/best.pt --imgsz 640
```

Datasets: VisDrone-DET (pedestrians/people → `survivor`), SARD (Kaggle), mannequin (Roboflow).
These are **not** committed (licensing + size); `docs/DATASETS.md` has the download steps.

## 6. Troubleshooting

**`cv2.error: The function is not implemented ... cvShowImage`** — the headless OpenCV build is
installed. Remove it:
```bash
pip uninstall -y opencv-python-headless
pip install --force-reinstall opencv-python
```

**`ModuleNotFoundError: No module named 'nidar'`** — run `pip install -e .` from the repo root,
or set `PYTHONPATH`:
```bash
# Windows:  $env:PYTHONPATH="src"
# Linux/macOS:  export PYTHONPATH=src
```

**`yolo: command not found`** — activate the virtualenv, then `pip install ultralytics`.

**PowerShell: `curl` downloads return HTML** — `curl` is aliased to `Invoke-WebRequest`; use
`curl.exe`.

## 7. Repository layout

```
configs/      dataset, training, training_local, deployment YAML
src/nidar/    data pipeline, training, inference (detector/tiling/tracking), camera,
              telemetry, geolocation, events, ground station, utils
scripts/      prepare_dataset, train, evaluate, detect_video, detect_camera,
              export_ncnn, benchmark_pi
models/       survivor_best.pt (PyTorch) + survivor_ncnn/ (Pi deployment)
docs/         DATASETS, TRAINING, DEPLOYMENT, ARCHITECTURE
tests/        dataset/converter/tracking/event tests
notebooks/    train_colab.ipynb
```

## License

Code: MIT. Bundled weights were trained on VisDrone, SARD and Roboflow mannequin data; those
datasets keep their own licenses and are not redistributed here — check them before commercial use.
