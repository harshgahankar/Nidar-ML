"""Tests: IoU tracker confirmation, NMS merge, event logger."""
import sys

sys.path.insert(0, "src")

from nidar.events.logger import EventLogger
from nidar.inference.detector import Detection
from nidar.inference.tiling import merge_detections_nms
from nidar.inference.tracking import IoUTracker


def test_tracker_confirmation():
    tr = IoUTracker(iou_threshold=0.3, max_missing=2, confirmation_frames=3)
    box = (10.0, 10.0, 50.0, 50.0)
    newly_total = 0
    for _ in range(4):
        dets, newly = tr.update([Detection(*box, 0.9)])
        newly_total += len(newly)
    assert newly_total == 1  # confirmed exactly once
    assert dets[0].track_id == 1


def test_tracker_holds_missing():
    tr = IoUTracker(max_missing=2, confirmation_frames=1)
    tr.update([Detection(0, 0, 10, 10, 0.9)])
    tr.update([])
    assert len(tr.tracks) == 1
    tr.update([])
    tr.update([])
    assert len(tr.tracks) == 0


def test_nms_removes_dup(sample_detections):
    kept = merge_detections_nms(sample_detections, 0.45)
    assert len(kept) == 2 and kept[0].conf == 0.9


def test_event_logger_writes(tmp_path):
    import json

    ev = EventLogger(tmp_path, save_evidence=False)
    d = Detection(1, 2, 3, 4, 0.8)
    d.track_id = 7
    out = ev.log(d)
    assert out["track_id"] == 7
    assert (tmp_path / "detections.jsonl").exists()
    assert json.loads((tmp_path / "detections.jsonl").read_text().splitlines()[0])["conf"] == 0.8
    assert (tmp_path / "detections.csv").exists()
