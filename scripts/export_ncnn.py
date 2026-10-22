#!/usr/bin/env python3
"""Export trained model to NCNN for Raspberry Pi.

Usage: python scripts/export_ncnn.py --weights runs/nidar_survivor/weights/best.pt
Requires: ultralytics + onnx + ncnn tooling (do this on desktop/Colab, copy dir to Pi).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export YOLO weights to NCNN")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = model.export(format="ncnn", imgsz=args.imgsz)
    print(f"NCNN export -> {out}\nCopy the directory to the Pi and set model.path in configs/deployment.yaml")
