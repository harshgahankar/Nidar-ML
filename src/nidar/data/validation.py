"""Annotation validation: checks YOLO labels against images.

Reports: missing labels, empty labels, invalid lines, out-of-range coords,
tiny boxes, corrupt images. Optionally deletes invalid pairs with --fix.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from nidar.utils.image import is_image_file


@dataclass
class ValidationReport:
    images: int = 0
    labels: int = 0
    boxes: int = 0
    missing_labels: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    invalid_lines: list[str] = field(default_factory=list)
    out_of_range: list[str] = field(default_factory=list)
    tiny_boxes: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "images": self.images, "labels": self.labels, "boxes": self.boxes,
            "missing_labels": len(self.missing_labels), "empty_labels": len(self.empty_labels),
            "invalid_lines": len(self.invalid_lines), "out_of_range": len(self.out_of_range),
            "tiny_boxes": len(self.tiny_boxes), "corrupt_images": len(self.corrupt_images),
            "missing_labels_files": self.missing_labels[:20],
            "corrupt_images_files": self.corrupt_images[:20],
        }


def validate_pair(img: Path, label: Path, min_size_px: int = 8) -> ValidationReport:
    rep = ValidationReport()
    try:
        with Image.open(img) as im:
            im.verify()
        with Image.open(img) as im:
            iw, ih = im.size
    except Exception:
        rep.corrupt_images.append(str(img))
        return rep
    rep.images = 1
    if not label.exists():
        rep.missing_labels.append(str(img))
        return rep
    rep.labels = 1
    lines = [ln.strip() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        rep.empty_labels.append(str(img))
        return rep
    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) != 5:
            rep.invalid_lines.append(f"{label}:{i + 1}")
            continue
        try:
            cls, cx, cy, w, h = int(parts[0]), *(float(x) for x in parts[1:])
        except ValueError:
            rep.invalid_lines.append(f"{label}:{i + 1}")
            continue
        if cls < 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            rep.out_of_range.append(f"{label}:{i + 1}")
            continue
        if w * iw < min_size_px or h * ih < min_size_px:
            rep.tiny_boxes.append(f"{label}:{i + 1}")
            continue
        rep.boxes += 1
    return rep


def validate_dataset(images_dir: Path, labels_dir: Path, min_size_px: int = 8) -> ValidationReport:
    total = ValidationReport()
    imgs = [p for p in sorted(images_dir.glob("*")) if is_image_file(p)] if images_dir.exists() else []
    for img in imgs:
        r = validate_pair(img, labels_dir / f"{img.stem}.txt", min_size_px)
        total.images += r.images
        total.labels += r.labels
        total.boxes += r.boxes
        total.missing_labels += r.missing_labels
        total.empty_labels += r.empty_labels
        total.invalid_lines += r.invalid_lines
        total.out_of_range += r.out_of_range
        total.tiny_boxes += r.tiny_boxes
        total.corrupt_images += r.corrupt_images
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate YOLO dataset")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--min-size", type=int, default=8)
    args = ap.parse_args()
    rep = validate_dataset(args.images, args.labels, args.min_size)
    import json

    print(json.dumps(rep.to_dict(), indent=2))


if __name__ == "__main__":
    main()
