from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from detectors.base import Detection
from trackers.base import Track, Tracker
from trackers.factory import register_tracker

logger = logging.getLogger(__name__)


class _DetectionBatch:
    def __init__(self, detections: Sequence[Detection]) -> None:
        if detections:
            xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
            self.conf = np.array([d.confidence for d in detections], dtype=np.float32)
            self.cls = np.array([d.class_id for d in detections], dtype=np.float32)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            self.conf = np.zeros((0,), dtype=np.float32)
            self.cls = np.zeros((0,), dtype=np.float32)

        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack([xyxy[:, 0] + w / 2, xyxy[:, 1] + h / 2, w, h], axis=1)
        self.xyxy = xyxy

    def __len__(self) -> int:
        return len(self.conf)


@register_tracker("bytetrack")
class ByteTrackTracker(Tracker):
    def __init__(
        self,
        track_high_thresh: float = 0.4,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.4,
        track_buffer: int = 60,
        match_thresh: float = 0.8,
        fps: int = 30,
    ) -> None:
        self._args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
        )
        self._fps = fps
        self._tracker = BYTETracker(self._args, frame_rate=fps)

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> List[Track]:
        out = self._tracker.update(_DetectionBatch(detections), frame)
        tracks: List[Track] = []
        rows = np.asarray(out).reshape(-1, 8) if len(out) else []
        for row in rows:
            x1, y1, x2, y2, track_id, score = row[:6]
            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(score),
                )
            )
        return tracks

    def reset(self) -> None:
        self._tracker = BYTETracker(self._args, frame_rate=self._fps)
