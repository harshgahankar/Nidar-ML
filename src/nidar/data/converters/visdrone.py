"""VisDrone (x1,y1,w,h in pixels, 1-indexed classes) -> YOLO normalized.

VisDrone annotation format (per `annotations/*.txt`, comma separated):
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>
Categories: 0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck,
  7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others
Only pedestrian(1) + people(2) map to survivor(0); everything else is excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

VISDRONE_KEEP = {1: 0, 2: 0}  # visdrone class -> yolo class (single-class survivor)


def convert_file(ann_file: Path, img_w: int, img_h: int, keep: dict[int, int] | None = None) -> list[str]:
    keep = keep if keep is not None else VISDRONE_KEEP
    lines: list[str] = []
    with open(ann_file, encoding="utf-8-sig") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = [p.strip() for p in raw.replace(";", ",").split(",")]
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
                cls = int(float(parts[5]))
            except ValueError:
                continue
            if cls not in keep or w <= 0 or h <= 0:
                continue
            # clamp into image
            x = max(0.0, min(x, img_w - 1))
            y = max(0.0, min(y, img_h - 1))
            w = max(0.0, min(w, img_w - x))
            h = max(0.0, min(h, img_h - y))
            if w < 1 or h < 1:
                continue
            cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
            nw, nh = w / img_w, h / img_h
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                continue
            lines.append(f"{keep[cls]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_dataset(visdrone_root: Path, out_images: Path, out_labels: Path, split: str = "train") -> dict:
    """Convert a VisDrone split. Expects <root>/sequences or images + annotations layout.

    Supports both common layouts:
     PRE: <root>/VisDrone2019-DET-train/{images,annotations}
      fallback: search for */images and */annotations recursively.
    Copies nothing — caller decides copy/symlink; here we only write labels next
    to *existing* image names and return stats. Images must be copied separately
    by the importer.
    """
    stats = {"images": 0, "boxes": 0, "skipped": 0}
    img_dirs = list(visdrone_root.rglob("images"))
    ann_dirs = list(visdrone_root.rglob("annotations"))
    if not img_dirs or not ann_dirs:
        raise FileNotFoundError(f"No images/annotations found under {visdrone_root}")
    ann_map = {p.stem: p for d in ann_dirs for p in d.glob("*.txt")}
    out_labels.mkdir(parents=True, exist_ok=True)
    for img_dir in img_dirs:
        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            ann = ann_map.get(img.stem)
            if ann is None:
                stats["skipped"] += 1
                continue
            with Image.open(img) as im:
                iw, ih = im.size
            yolo_lines = convert_file(ann, iw, ih)
            (out_labels / f"{img.stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            stats["images"] += 1
            stats["boxes"] += len(yolo_lines)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO format")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-labels", required=True, type=Path)
    ap.add_argument("--out-images", required=False, default=None, type=Path)
    args = ap.parse_args()
    stats = convert_dataset(args.root, args.out_images or args.out_labels, args.out_labels)
    print(stats)


if __name__ == "__main__":
    main()
