from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from detectors.base import Detection
from trackers.base import Track, Tracker
from trackers.factory import register_tracker
from utils.device import is_cuda, resolve_device

logger = logging.getLogger(__name__)


@register_tracker("deepsort")
class DeepSortTracker(Tracker):
    def __init__(
        self,
        max_age: int = 90,
        n_init: int = 3,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100,
        nms_max_overlap: float = 0.7,
        embedder: str = "mobilenet",
        max_coast_frames: int = 0,
        device: str = "auto",
    ) -> None:
        self.max_coast_frames = max_coast_frames
        self._params = dict(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            nms_max_overlap=nms_max_overlap,
        )
        self._embedder = embedder
        self._use_gpu = is_cuda(resolve_device(device))
        self._tracker = self._build()
        logger.info("deepsort ready (embedder=%s, gpu=%s)", embedder, self._use_gpu)

    def _build(self) -> DeepSort:
        return DeepSort(
            embedder=self._embedder,
            embedder_gpu=self._use_gpu,
            half=self._use_gpu,
            bgr=True,
            **self._params,
        )

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> List[Track]:
        if frame is None or frame.size == 0:
            return []

        raw = [(d.to_ltwh(), d.confidence, d.class_name) for d in detections]
        if raw:
            ds_tracks = self._tracker.update_tracks(raw, frame=frame)
        else:
            ds_tracks = self._tracker.update_tracks([], embeds=[], frame=frame)

        return self._convert(ds_tracks)

    def reset(self) -> None:
        self._tracker = self._build()

    def _convert(self, ds_tracks) -> List[Track]:
        tracks: List[Track] = []
        for t in ds_tracks:
            if not t.is_confirmed():
                continue
            if t.time_since_update > self.max_coast_frames:
                continue

            x1, y1, x2, y2 = t.to_ltrb()
            conf = t.det_conf if getattr(t, "det_conf", None) is not None else 0.0
            tracks.append(
                Track(
                    track_id=int(t.track_id),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    time_since_update=int(t.time_since_update),
                )
            )
        return tracks
