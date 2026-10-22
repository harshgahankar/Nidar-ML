#!/usr/bin/env python3
"""Thin wrapper: python scripts/evaluate.py --weights runs/.../best.pt [--data ...]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from nidar.training.evaluate import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="datasets/processed/data.yaml")
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(json.dumps(evaluate(args.weights, args.data, args.out, args.imgsz, args.device), indent=2))
