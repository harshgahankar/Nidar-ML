"""YOLO training entry point (Colab + local).

Prefers yolo26n.pt (Jan 2026 SOTA, NMS-free, DFL-free, ProgLoss+STAL for small
targets — ideal for ~20m aerial views); falls back to yolo11n.pt if the
download fails. All settings come from configs/training.yaml (overridable by CLI).
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from nidar.utils.config import load_yaml
from nidar.utils.seed import set_seed


def resolve_weights(preferred: str, fallback: str):
    from ultralytics import YOLO

    for name in (preferred, fallback):
        try:
            model = YOLO(name)  # triggers download on first use
            return model, name
        except Exception as e:  # noqa: BLE001
            print(f"Could not load {name}: {e}")
    raise RuntimeError(f"Neither {preferred} nor {fallback} could be loaded.")


def train(cfg_path: Path = Path("configs/training.yaml"), resume: str | None = None,
          device: str | None = None, epochs: int | None = None) -> dict:
    from ultralytics.utils import SETTINGS

    SETTINGS["wandb"] = False
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    model_cfg, data = cfg.get("model", {}), cfg.get("data", "datasets/processed/data.yaml")
    t, aug = cfg.get("train", {}), cfg.get("augment", {})
    if resume:
        from ultralytics import YOLO

        model = YOLO(resume)
        used = resume
    else:
        model, used = resolve_weights(model_cfg.get("weights", "yolo26n.pt"),
                                      model_cfg.get("fallback_weights", "yolo11n.pt"))
    import torch

    if device is None:
        device = t.get("device", 0 if torch.cuda.is_available() else "cpu")
    print(f"GPU available: {torch.cuda.is_available()} | using device={device} | weights={used}")
    # Resolve project to an absolute path: ultralytics prefixes RELATIVE project dirs with
    # SETTINGS['runs_dir']/task, which produced confusing paths like runs/detect/runs/<name>.
    project_dir = Path(t.get("project", "runs")).resolve()
    overrides = dict(
        data=data, epochs=epochs or t.get("epochs", 100), imgsz=t.get("imgsz", 960),
        batch=t.get("batch", 8), patience=t.get("patience", 20),
        project=str(project_dir), name=t.get("name", "nidar_survivor"),
        device=device, workers=t.get("workers", 4), val=t.get("val", True),
        save_period=t.get("save_period", 10), seed=t.get("seed", 42),
        hsv_h=aug.get("hsv_h", 0.015), hsv_s=aug.get("hsv_s", 0.4), hsv_v=aug.get("hsv_v", 0.3),
        degrees=aug.get("degrees", 15.0), translate=aug.get("translate", 0.1),
        scale=aug.get("scale", 0.3), shear=aug.get("shear", 2.0),
        fliplr=aug.get("fliplr", 0.5), flipud=aug.get("flipud", 0.2),
        mosaic=aug.get("mosaic", 0.5), mixup=aug.get("mixup", 0.0),
        copy_paste=aug.get("copy_paste", 0.1),
    )
    results = model.train(resume=bool(resume), **overrides)
    # Use the trainer's ACTUAL save dir (handles existing-run suffixes like -2).
    trainer = getattr(model, "trainer", None)
    run_dir = Path(trainer.save_dir) if trainer is not None else project_dir / overrides["name"]
    meta = {"weights": used, "overrides": {k: str(v) for k, v in overrides.items()},
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2))
    return {"run_dir": str(run_dir), "weights": used, "results": str(results)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train NIDAR survivor detector")
    ap.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    ap.add_argument("--resume", default=None, help="Path to last.pt to resume")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    print(train(args.config, args.resume, args.device, args.epochs))


if __name__ == "__main__":
    main()
