"""VisDrone importer.

Official source: https://github.com/VisDrone/VisDrone-Dataset
Download is manual (challenge page / GitHub releases require acceptance).
Expected layout after manual download + unzip into datasets/raw/visdrone/:
  VisDrone2019-DET-train/{images,annotations}
  VisDrone2019-DET-val/{images,annotations}
This importer converts annotations (pedestrian+people -> survivor) and stages
files into staging/visdrone/<session>/ with YOLO labels.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from nidar.data.converters.visdrone import convert_file
from nidar.utils.image import is_image_file


def import_visdrone(raw_root: Path, staging: Path) -> dict:
    from PIL import Image

    staging.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0, "missing_ann": 0}
    img_files = [p for p in raw_root.rglob("*") if is_image_file(p)]
    # only keep files under an 'images' dir to avoid staged copies
    img_files = [p for p in img_files if "images" in p.parts]
    for img in tqdm(img_files, desc="visdrone"):
        # sibling annotations dir
        ann = None
        for parent in img.parents:
            cand = parent / "annotations" / f"{img.stem}.txt"
            if cand.exists():
                ann = cand
                break
            cand2 = parent.parent / "annotations" / f"{img.stem}.txt"
            if cand2.exists():
                ann = cand2
                break
        if ann is None:
            stats["missing_ann"] += 1
            continue
        try:
            with Image.open(img) as im:
                iw, ih = im.size
        except Exception:
            continue
        lines = convert_file(ann, iw, ih)
        # VisDrone DET names: <sequence>_<frame>_d_<id>.jpg -> session per sequence
        # so near-identical frames never leak across train/val/test.
        session = img.stem.split("_")[0] or "unknown_seq"
        dest = staging / "visdrone" / session
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest / img.name)
        (dest / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["boxes"] += len(lines)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Import VisDrone (requires manual download)")
    ap.add_argument("--raw", type=Path, default=Path("datasets/raw/visdrone"))
    ap.add_argument("--staging", type=Path, default=Path("datasets/staging"))
    args = ap.parse_args()
    if not args.raw.exists():
        print(f"Missing {args.raw}. Download VisDrone2019-DET from "
              "https://github.com/VisDrone/VisDrone-Dataset and unzip here.")
        raise SystemExit(1)
    print(import_visdrone(args.raw, args.staging))


if __name__ == "__main__":
    main()
