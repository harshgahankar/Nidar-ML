#!/usr/bin/env python3
"""Thin wrapper: python scripts/train.py [--config ...] [--resume ...] [--device ...] [--epochs N]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")
from nidar.training.train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    print(train(cfg_path=args.config, resume=args.resume, device=args.device, epochs=args.epochs))
