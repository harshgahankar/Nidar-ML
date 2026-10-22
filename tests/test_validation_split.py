"""Tests: validation + session-aware splitting."""
import sys

sys.path.insert(0, "src")

from nidar.data.preparation import file_split, session_split
from nidar.data.validation import validate_dataset


def test_validation_counts(tiny_dataset):
    img_dir, lab_dir = tiny_dataset
    rep = validate_dataset(img_dir, lab_dir)
    assert rep.images == 6 and rep.boxes == 6 and not rep.missing_labels


def test_validation_missing_label(tiny_dataset):
    img_dir, lab_dir = tiny_dataset
    (lab_dir / "img_0.txt").unlink()
    rep = validate_dataset(img_dir, lab_dir)
    assert len(rep.missing_labels) == 1


def test_session_split_no_leak():
    pairs = [(f"img{i}", f"lab{i}", f"flight{i // 3}") for i in range(9)]
    splits = session_split(pairs, 0.66, 0.2, seed=1, group_by_session=True)
    sess = {k: {p[2] for p in v} for k, v in splits.items()}
    assert not (sess["train"] & sess["val"]) and not (sess["train"] & sess["test"])


def test_session_split_never_degenerate():
    # regression: 4 giant sessions previously produced an EMPTY val split
    pairs = [(f"img{i}", f"lab{i}", f"s{i // 100}") for i in range(400)]
    splits = session_split(pairs, 0.7, 0.15, seed=42, group_by_session=True)
    assert len(splits["train"]) > 0 and len(splits["val"]) > 0 and len(splits["test"]) > 0
    total = sum(len(v) for v in splits.values())
    assert total == 400


def test_file_split_ratios():
    pairs = [(str(i), str(i), "s") for i in range(100)]
    splits = file_split(pairs, 0.7, 0.15, seed=42)
    assert len(splits["train"]) == 70 and len(splits["val"]) == 15 and len(splits["test"]) == 15
