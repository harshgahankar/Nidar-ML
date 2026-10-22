"""SARD importer (https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue).

Requires Kaggle auth (`kaggle datasets download ...`) — NOT auto-downloaded.
After `kaggle datasets download nikolasgegenava/sard-search-and-rescue -p datasets/raw/sard --unzip`:
  - if a COCO JSON is present, convert with keep-names {person,human};
  - if YOLO txt labels are present, remap class ids to 0;
  - otherwise stage images for manual annotation and report the blocker.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from nidar.utils.image import is_image_file


def import_sard(raw_root: Path, staging: Path) -> dict:
    from nidar.data.converters.coco import convert_coco

    dest = staging / "sard"
    dest.mkdir(parents=True, exist_ok=True)
    stats: dict = {"images": 0, "boxes": 0, "mode": None}
    coco_files = list(raw_root.rglob("*.json"))
    coco_files = [c for c in coco_files if "coco" in c.name.lower() or "annot" in c.name.lower() or "instance" in c.name.lower()]
    if coco_files:
        tmp_labels = dest / "_tmp_labels"
        res = convert_coco(coco_files[0], tmp_labels, keep_names={"person", "human", "pedestrian"})
        # copy images alongside
        imgs = [p for p in raw_root.rglob("*") if is_image_file(p)]
        for img in tqdm(imgs, desc="sard"):
            shutil.copy2(img, dest / img.name)
            lab = tmp_labels / f"{img.stem}.txt"
            if lab.exists():
                shutil.copy2(lab, dest / lab.name)
        stats.update({"mode": f"coco:{coco_files[0].name}", **res})
        return stats
    # YOLO txt present?
    txts = {p.stem for p in raw_root.rglob("*.txt")}
    imgs = [p for p in raw_root.rglob("*") if is_image_file(p)]
    if txts:
        for img in tqdm(imgs, desc="sard"):
            src = next((t for t in raw_root.rglob(f"{img.stem}.txt")), None)
            # SARD names: <orig>_jpg.rf.<hash>.jpg (tiles of one source image).
            # Session per source image so tiles never leak across splits.
            session = img.stem.split("_jpg.rf.")[0] or "misc"
            dest_session = dest / session
            dest_session.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest_session / img.name)
            if src is not None:
                lines = []
                for ln in src.read_text().splitlines():
                    parts = ln.split()
                    if len(parts) == 5:
                        lines.append("0 " + " ".join(parts[1:]))
                (dest_session / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                stats["boxes"] += len(lines)
            stats["images"] += 1
        stats["mode"] = "yolo-remap"
        return stats
    for img in tqdm(imgs, desc="sard-stage-unlabelled"):
        shutil.copy2(img, dest / img.name)
        stats["images"] += 1
    stats["mode"] = "staged-unlabelled-needs-manual-annotation"
    stats["note"] = ("No machine-readable annotations found. Annotate staged images "
                     "with Label Studio/CVAT (YOLO format) before training.")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Import SARD (requires Kaggle download first)")
    ap.add_argument("--raw", type=Path, default=Path("datasets/raw/sard"))
    ap.add_argument("--staging", type=Path, default=Path("datasets/staging"))
    args = ap.parse_args()
    if not args.raw.exists() or not any(args.raw.iterdir()):
        print("Missing datasets/raw/sard. Run:\n"
              "  kaggle datasets download nikolasgegenava/sard-search-and-rescue "
              "-p datasets/raw/sard --unzip")
        raise SystemExit(1)
    print(import_sard(args.raw, args.staging))


if __name__ == "__main__":
    main()
