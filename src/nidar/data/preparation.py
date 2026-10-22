"""Dataset preparation: merge sources, dedup, session-aware split, stats, frame extraction.

Single-class `survivor` (0). All sources are normalised to YOLO txt before merge.
Split is grouped by session (parent folder name) when group_by_session=True to
avoid near-identical frames leaking across train/val/test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from nidar.utils.image import dhash, hamming, is_image_file, read_image_bgr


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(images_dir: Path, ham_threshold: int = 5) -> list[list[str]]:
    """Group near-duplicate images via dHash. Returns groups of filenames."""
    hashes: list[tuple[str, int]] = []
    for img_path in sorted(images_dir.glob("*")):
        if not is_image_file(img_path):
            continue
        img = read_image_bgr(img_path)
        if img is None:
            continue
        hashes.append((img_path.name, dhash(img)))
    groups: list[list[str]] = []
    used: set[str] = set()
    for i, (name_i, hi) in enumerate(hashes):
        if name_i in used:
            continue
        group = [name_i]
        for name_j, hj in hashes[i + 1:]:
            if name_j not in used and hamming(hi, hj) <= ham_threshold:
                group.append(name_j)
                used.add(name_j)
        used.add(name_i)
        if len(group) > 1:
            groups.append(group)
    return groups


def extract_frames(
    video: Path, out_dir: Path, every_n: int = 30, prefix: str | None = None,
    ham_threshold: int = 5, keep_blurry: bool = False,
) -> dict:
    """Extract frames every N frames; skip near-duplicates; record session metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    session = prefix or video.stem
    saved, skipped_dup, frame_idx = 0, 0, 0
    last_hash: int | None = None
    meta_rows: list[dict] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total if total > 0 else None, desc=f"extract {video.name}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every_n == 0:
            h = dhash(frame)
            is_dup = last_hash is not None and hamming(h, last_hash) <= ham_threshold
            if not is_dup:
                name = f"{session}_f{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                last_hash = h
                saved += 1
                meta_rows.append({"file": name, "session": session, "frame": frame_idx,
                                  "source_video": str(video)})
            else:
                skipped_dup += 1
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    meta_file = out_dir / f"{session}_meta.csv"
    if meta_rows:
        with open(meta_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            w.writeheader()
            w.writerows(meta_rows)
    return {"saved": saved, "skipped_duplicates": skipped_dup, "meta": str(meta_file)}


def collect_staged(staging_dir: Path) -> list[tuple[Path, Path, str]]:
    """Collect (image, label, session) pairs from staging/<source>/<session>/* layout.

    Session = path of the image's folder relative to staging (e.g. 'visdrone/0000001',
    'sard/gss1008', 'mannequin/10'), so splits group near-identical frames/tiles.
    """
    pairs: list[tuple[Path, Path, str]] = []
    for img in sorted(staging_dir.rglob("*")):
        if not is_image_file(img):
            continue
        label = img.parent / f"{img.stem}.txt"
        # labels may sit in sibling labels dir (Roboflow exports copied flat)
        if not label.exists():
            alt = img.parent.parent / "labels" / f"{img.stem}.txt"
            if alt.exists():
                label = alt
        session = "/".join(img.relative_to(staging_dir).parent.parts) or "misc"
        pairs.append((img, label, session))
    return pairs


def session_split(
    pairs: list[tuple[Path, Path, str]], train: float = 0.7, val: float = 0.15,
    seed: int = 42, group_by_session: bool = True,
) -> dict[str, list[tuple[Path, Path, str]]]:
    rng = random.Random(seed)
    if group_by_session:
        sessions: dict[str, list] = {}
        for p in pairs:
            sessions.setdefault(p[2], []).append(p)
        keys = sorted(sessions.keys())
        rng.shuffle(keys)
        n = len(keys)
        if n < 3:
            return file_split(pairs, train, val, seed)
        test_frac = max(0.0, 1.0 - train - val)
        # Guaranteed non-empty val AND test (never a degenerate 0-image split).
        n_test = min(max(1, int(round(n * test_frac))), n - 2)
        n_val = min(max(1, int(round(n * val))), n - 1 - n_test)
        n_train = n - n_val - n_test
        train_keys = set(keys[:n_train])
        val_keys = set(keys[n_train:n_train + n_val])
        out: dict[str, list] = {"train": [], "val": [], "test": []}
        for k in keys:
            dest = "train" if k in train_keys else ("val" if k in val_keys else "test")
            out[dest].extend(sessions[k])
        return out
    return file_split(pairs, train, val, seed)


def file_split(pairs, train=0.7, val=0.15, seed=42):
    rng = random.Random(seed)
    items = list(pairs)
    rng.shuffle(items)
    n = len(items)
    n_train, n_val = int(n * train), int(n * val)
    return {"train": items[:n_train], "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:]}


def materialize(
    splits: dict[str, list[tuple[Path, Path, str]]], processed_dir: Path,
    names: dict[int, str] | None = None,
) -> dict:
    """Copy images+labels into processed/{images,labels}/{train,val,test} + write data.yaml.

    Files are prefixed with their source (session's top folder, e.g. 'sard_...jpg')
    to prevent cross-source filename collisions and to keep source attribution.
    """
    stats: dict = {}
    for split, items in splits.items():
        idir = processed_dir / "images" / split
        ldir = processed_dir / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        n_boxes = 0
        for img, label, session in items:
            source = session.split("/")[0] if "/" in session else session
            new_name = f"{source}_{img.name}"
            new_stem = Path(new_name).stem
            shutil.copy2(img, idir / new_name)
            if label.exists():
                shutil.copy2(label, ldir / f"{new_stem}.txt")
                for ln in (ldir / f"{new_stem}.txt").read_text().splitlines():
                    if ln.strip():
                        n_boxes += 1
            else:
                (ldir / f"{new_stem}.txt").write_text("")
        stats[split] = {"images": len(items), "boxes": n_boxes}
    data_yaml = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names or {0: "survivor"},
    }
    with open(processed_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    stats["data_yaml"] = str(processed_dir / "data.yaml")
    return stats


def dataset_statistics(processed_dir: Path, reports_dir: Path) -> dict:
    """Box-size distribution, source distribution, counts -> JSON report."""
    from PIL import Image

    report: dict = {"splits": {}, "box_area_frac": [], "sources": Counter()}
    for split in ("train", "val", "test"):
        idir, ldir = processed_dir / "images" / split, processed_dir / "labels" / split
        imgs = [p for p in idir.glob("*")] if idir.exists() else []
        boxes, areas = 0, []
        for img in imgs:
            if not is_image_file(img):
                continue
            try:
                with Image.open(img) as im:
                    iw, ih = im.size
            except Exception:
                continue
            report["sources"][img.stem.split("_")[0]] += 1
            lab = ldir / f"{img.stem}.txt"
            if not lab.exists():
                continue
            for ln in lab.read_text().splitlines():
                parts = ln.split()
                if len(parts) != 5:
                    continue
                try:
                    _, _, _, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                boxes += 1
                areas.append(w * h)
        report["splits"][split] = {"images": len(imgs), "boxes": boxes}
        report["box_area_frac"].extend(areas)
    areas = report.pop("box_area_frac")
    if areas:
        import numpy as np

        a = np.array(areas)
        report["box_area_frac_stats"] = {
            "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "small_lt_0.001": int((a < 0.001).sum()), "tiny_lt_0.0002": int((a < 0.0002).sum()),
        }
    report["sources"] = dict(report["sources"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "dataset_stats.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare unified YOLO dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract-frames", help="Extract frames from a drone video")
    e.add_argument("--video", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--every-n", type=int, default=30)
    e.add_argument("--session", default=None)
    s = sub.add_parser("split", help="Split staged dataset into processed/")
    s.add_argument("--staging", required=True, type=Path)
    s.add_argument("--processed", required=True, type=Path)
    s.add_argument("--train", type=float, default=0.7)
    s.add_argument("--val", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--no-group", action="store_true")
    d = sub.add_parser("stats", help="Write dataset statistics report")
    d.add_argument("--processed", required=True, type=Path)
    d.add_argument("--reports", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "extract-frames":
        print(extract_frames(args.video, args.out, args.every_n, args.session))
    elif args.cmd == "split":
        pairs = collect_staged(args.staging)
        splits = session_split(pairs, args.train, args.val, args.seed, not args.no_group)
        print(materialize(splits, args.processed))
    elif args.cmd == "stats":
        print(json.dumps(dataset_statistics(args.processed, args.reports), indent=2))


if __name__ == "__main__":
    main()
