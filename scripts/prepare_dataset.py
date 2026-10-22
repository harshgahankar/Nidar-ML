#!/usr/bin/env python3
"""Full dataset pipeline: import all sources -> validate -> session split -> stats.

Usage:
  python scripts/prepare_dataset.py --help
  python scripts/prepare_dataset.py --skip-visdrone   # when raw data absent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from nidar.data.preparation import collect_staged, dataset_statistics, materialize, session_split
from nidar.data.validation import validate_dataset
from nidar.utils.config import load_yaml
from nidar.utils.log import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare unified NIDAR dataset")
    ap.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    ap.add_argument("--staging", type=Path, default=Path("datasets/staging"))
    ap.add_argument("--skip-visdrone", action="store_true")
    ap.add_argument("--skip-sard", action="store_true")
    ap.add_argument("--skip-mannequin", action="store_true")
    ap.add_argument("--skip-custom", action="store_true")
    args = ap.parse_args()
    log = setup_logging()
    cfg = load_yaml(args.config)
    raw = {k: Path(v) for k, v in cfg.get("raw_dirs", {}).items()}
    processed = Path(cfg.get("processed_dir", "datasets/processed"))
    reports = Path(cfg.get("reports_dir", "datasets/reports"))
    reports.mkdir(parents=True, exist_ok=True)

    if not args.skip_visdrone:
        from nidar.data.importers.visdrone_importer import import_visdrone

        if raw["visdrone"].exists():
            log.info("VisDrone: %s", import_visdrone(raw["visdrone"], args.staging))
        else:
            log.warning("No datasets/raw/visdrone — download per docs/DATASETS.md; skipping.")
    if not args.skip_sard:
        from nidar.data.importers.sard_importer import import_sard

        if raw.get("sard", Path()).exists() and any(raw["sard"].iterdir()):
            log.info("SARD: %s", import_sard(raw["sard"], args.staging))
        else:
            log.warning("No datasets/raw/sard — Kaggle download required; skipping.")
    if not args.skip_mannequin:
        from nidar.data.importers.mannequin_importer import import_mannequin

        log.info("Mannequin: %s", import_mannequin(raw["mannequin"], args.staging))
    if not args.skip_custom:
        from nidar.data.importers.custom_importer import import_custom

        if raw["custom"].exists():
            log.info("Custom: %s", import_custom(raw["custom"], args.staging))
        else:
            log.warning("No datasets/raw/custom; skipping.")

    pairs = collect_staged(args.staging)
    log.info("Staged pairs: %d", len(pairs))
    if not pairs:
        log.error("Nothing staged — add data per docs/DATASETS.md then rerun.")
        raise SystemExit(2)
    sp = cfg.get("split", {})
    splits = session_split(pairs, sp.get("train", 0.7), sp.get("val", 0.15),
                           sp.get("seed", 42), sp.get("group_by_session", True))
    log.info("Split sizes: %s", {k: len(v) for k, v in splits.items()})
    log.info("Materialize: %s", materialize(splits, processed, {0: "survivor"}))
    for s in ("train", "val", "test"):
        rep = validate_dataset(processed / "images" / s, processed / "labels" / s,
                               cfg.get("filters", {}).get("min_box_width_px", 8))
        (reports / f"validation_{s}.json").write_text(json.dumps(rep.to_dict(), indent=2))
        log.info("%s validation: %s", s, rep.to_dict())
    log.info("Stats: %s", dataset_statistics(processed, reports))


if __name__ == "__main__":
    main()
