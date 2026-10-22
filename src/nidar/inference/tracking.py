"""Lightweight IoU tracker + multi-frame confirmation.

Provisional detections become *confirmed* only after `confirmation_frames`
consecutive associations; tracks survive `max_missing` frames without a match.
This suppresses flicker FPs and duplicate alerts on stationary targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nidar.inference.detector import Detection


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    track_id: int
    det: Detection
    hits: int = 1
    missing: int = 0
    confirmed: bool = False


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 10,
                 confirmation_frames: int = 3) -> None:
        self.iou_thr = iou_threshold
        self.max_missing = max_missing
        self.confirmation_frames = confirmation_frames
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, dets: list[Detection]) -> tuple[list[Detection], list[Detection]]:
        """Return (all_with_ids, newly_confirmed). Mutates dets with track ids."""
        matched_ids: set[int] = set()
        for d in sorted(dets, key=lambda x: x.conf, reverse=True):
            best, best_iou = None, self.iou_thr
            for t in self.tracks:
                if t.track_id in matched_ids:
                    continue
                iou = _iou(d, t.det)
                if iou >= best_iou:
                    best, best_iou = t, iou
            if best is not None:
                best.det, best.hits, best.missing = d, best.hits + 1, 0
                d.track_id = best.track_id
                matched_ids.add(best.track_id)
            else:
                t = Track(self._next_id, d)
                self._next_id += 1
                self.tracks.append(t)
                d.track_id = t.track_id
                matched_ids.add(t.track_id)
        newly: list[Detection] = []
        alive: list[Track] = []
        for t in self.tracks:
            if t.track_id not in matched_ids:
                t.missing += 1
            if t.missing <= self.max_missing:
                alive.append(t)
                if not t.confirmed and t.hits >= self.confirmation_frames:
                    t.confirmed = True
                    newly.append(t.det)
        self.tracks = alive
        return dets, newly

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]
