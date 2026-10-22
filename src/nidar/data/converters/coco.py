"""COCO -> YOLO converter (for SARD / Roboflow mannequin exports in COCO JSON).

Maps a configurable set of COCO category ids (or names) to a single YOLO class 0 (survivor).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_coco(
    coco_json: Path,
    out_labels: Path,
    keep_category_ids: set[int] | None = None,
    keep_names: set[str] | None = None,
    target_class: int = 0,
) -> dict:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data.get("images", [])}
    cats = {c["id"]: c.get("name", "") for c in data.get("categories", [])}
    if keep_category_ids is None:
        if keep_names is not None:
            keep_category_ids = {cid for cid, n in cats.items() if n in keep_names}
        else:
            keep_category_ids = set(cats.keys())  # keep all if unspecified
    assert keep_category_ids is not None
    out_labels.mkdir(parents=True, exist_ok=True)
    per_image: dict[int, list[str]] = {iid: [] for iid in images}
    skipped = 0
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        if ann.get("category_id") not in keep_category_ids:
            continue
        im = images.get(ann.get("image_id"))
        if im is None:
            skipped += 1
            continue
        iw, ih = float(im["width"]), float(im["height"])
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0 or iw <= 0 or ih <= 0:
            skipped += 1
            continue
        cx, cy, nw, nh = (x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and nw > 0 and nh > 0):
            skipped += 1
            continue
        per_image[im["id"]].append(f"{target_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    written = 0
    for iid, im in images.items():
        stem = Path(im["file_name"]).stem
        (out_labels / f"{stem}.txt").write_text(
            "\n".join(per_image.get(iid, [])) + ("\n" if per_image.get(iid) else "")
        )
        written += 1
    return {"images": written, "boxes": sum(len(v) for v in per_image.values()), "skipped": skipped}


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert COCO JSON to YOLO txt labels")
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--out-labels", required=True, type=Path)
    ap.add_argument("--keep-ids", nargs="*", type=int, default=None)
    ap.add_argument("--keep-names", nargs="*", default=None)
    args = ap.parse_args()
    print(convert_coco(args.coco, args.out_labels,
                       set(args.keep_ids) if args.keep_ids else None,
                       set(args.keep_names) if args.keep_names else None))


if __name__ == "__main__":
    main()
