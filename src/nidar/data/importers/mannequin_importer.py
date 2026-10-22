"""Mannequin importer (Roboflow Universe: drone-e2zlz/mannequin-detection-bzihs).

Roboflow exports require API key / manual download; NOT auto-downloaded.
Supports YOLO-format exports (data.yaml + images + labels) and COCO JSON exports.
All mannequin boxes -> survivor class 0.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from nidar.utils.image import is_image_file


def import_mannequin(raw_root: Path, staging: Path) -> dict:
    dest = staging / "mannequin"
    dest.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0}
    imgs = [p for p in raw_root.rglob("*") if is_image_file(p)]
    if not imgs:
        stats["note"] = ("Empty datasets/raw/mannequin. Export from Roboflow Universe "
                         "(YOLOv8/YOLO11 format) and unzip here. See docs/DATASETS.md.")
        return stats
    for img in tqdm(imgs, desc="mannequin"):
        # Session per source tile group (<id>_jpg.rf.<hash>.jpg) to avoid split leakage.
        session = img.stem.split("_jpg.rf.")[0] or "misc"
        dest_session = dest / session
        dest_session.mkdir(parents=True, exist_ok=True)
        # find sibling label
        label = None
        for cand in [img.parent / f"{img.stem}.txt",
                     img.parent.parent / "labels" / f"{img.stem}.txt",
                     raw_root / "labels" / f"{img.stem}.txt"]:
            if cand.exists():
                label = cand
                break
        shutil.copy2(img, dest_session / img.name)
        if label is not None:
            lines = []
            for ln in label.read_text().splitlines():
                parts = ln.split()
                if len(parts) == 5:
                    lines.append("0 " + " ".join(parts[1:]))
            (dest_session / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats["boxes"] += len(lines)
        stats["images"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Import Roboflow mannequin export")
    ap.add_argument("--raw", type=Path, default=Path("datasets/raw/mannequin"))
    ap.add_argument("--staging", type=Path, default=Path("datasets/staging"))
    args = ap.parse_args()
    print(import_mannequin(args.raw, args.staging))


if __name__ == "__main__":
    main()
