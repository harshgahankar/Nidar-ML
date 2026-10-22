"""Custom 20m drone footage importer: videos/images -> staging/custom/<session>/.

- Copies still images as-is (preserves full resolution).
- Extracts video frames via preparation.extract_frames (configurable stride + dedup).
- Writes session metadata CSV (camera, altitude, session) for later altitude-wise eval.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from tqdm import tqdm

from nidar.data.preparation import extract_frames
from nidar.utils.image import is_image_file

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def import_custom(raw_root: Path, staging: Path, every_n: int = 30,
                  altitude_m: float = 20.0, camera: str = "unknown") -> dict:
    dest_root = staging / "custom"
    dest_root.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "frames": 0, "videos": 0}
    for img in tqdm([p for p in raw_root.rglob("*") if is_image_file(p)], desc="custom-images"):
        session = img.parent.name if img.parent != raw_root else "session_01"
        dest = dest_root / session
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest / img.name)
        stats["images"] += 1
    for vid in tqdm([p for p in raw_root.rglob("*") if p.suffix.lower() in VIDEO_EXTS], desc="custom-video"):
        session = vid.stem
        dest = dest_root / session
        res = extract_frames(vid, dest, every_n=every_n, prefix=session)
        stats["frames"] += res["saved"]
        stats["videos"] += 1
        # append flight metadata
        meta = dest / "flight_meta.csv"
        with open(meta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["session", "altitude_m", "camera", "source_video"])
            w.writerow([session, altitude_m, camera, str(vid)])
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Import custom drone footage")
    ap.add_argument("--raw", type=Path, default=Path("datasets/raw/custom"))
    ap.add_argument("--staging", type=Path, default=Path("datasets/staging"))
    ap.add_argument("--every-n", type=int, default=30)
    ap.add_argument("--altitude", type=float, default=20.0)
    ap.add_argument("--camera", default="unknown")
    args = ap.parse_args()
    print(import_custom(args.raw, args.staging, args.every_n, args.altitude, args.camera))


if __name__ == "__main__":
    main()
