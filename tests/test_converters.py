"""Tests: VisDrone + COCO conversion math (no network, no weights)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

from nidar.data.converters.coco import convert_coco
from nidar.data.converters.visdrone import convert_file


def test_visdrone_basic(tmp_path):
    ann = tmp_path / "a.txt"
    ann.write_text("10,20,100,200,1,1,0,0\n0,0,50,50,1,4,0,0\n")  # pedestrian kept, car dropped
    lines = convert_file(ann, 1000, 1000)
    assert len(lines) == 1
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"
    assert abs(float(cx) - 0.06) < 1e-6 and abs(float(cy) - 0.12) < 1e-6
    assert abs(float(w) - 0.1) < 1e-6 and abs(float(h) - 0.2) < 1e-6


def test_visdrone_clamps_and_skips(tmp_path):
    ann = tmp_path / "b.txt"
    ann.write_text("bad-line\n5,5,0,10,1,1,0,0\n")
    assert convert_file(ann, 100, 100) == []


def test_coco_keep_names(tmp_path):
    import json

    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
                            {"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20], "iscrowd": 0}]}
    cj = tmp_path / "a.json"
    cj.write_text(json.dumps(coco))
    out = tmp_path / "labels"
    res = convert_coco(cj, out, keep_names={"person"})
    assert res["boxes"] == 1
    assert (out / "a.txt").read_text().strip().startswith("0 ")
