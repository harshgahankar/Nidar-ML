# Training (Colab)

1. Runtime → T4/L4 GPU. Upload `datasets/processed/` (or mount Drive and symlink).
2. Open `notebooks/train_colab.ipynb` → Run all.
3. Baseline: `yolo26n.pt` (Jan 2026; NMS-free, DFL-free, ProgLoss+STAL helps tiny 20 m targets),
   `imgsz=960, epochs=100, batch=8, patience=20`. Falls back to `yolo11n.pt` if download fails.
4. Augmentations (in `configs/training.yaml`): fliplr 0.5, flipud 0.2 (nadir-plausible),
   rotation ±15°, HSV jitter, scale 0.3, mosaic 0.5 (reduced to protect tiny boxes),
   mixup OFF, copy-paste 0.1. No augmentation destroys boxes or invents unrealistic samples.
5. Outputs: `runs/nidar_survivor/weights/{best,last}.pt` + `experiment_meta.json`.
   Resume: `python scripts/train.py --resume runs/nidar_survivor/weights/last.pt`.
6. Report only measured metrics from `scripts/evaluate.py` — never assumed accuracy.
