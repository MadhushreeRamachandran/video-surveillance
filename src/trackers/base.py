"""Tracker interface (Strategy pattern) and the shared Track data class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from detectors.base import Detection


@dataclass(frozen=True)
class Track:
    """One tracked person in one frame (pixel coordinates)."""

    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float                        # confidence of the matched detection
    time_since_update: int = 0               # 0 = matched this frame, >0 = coasting (occluded)
    class_name: str = "person"

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Bottom-centre of the box, used for zone tests."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Tracker(ABC):
    """Interface every tracking strategy implements. The pipeline only
    ever calls update() and reset()."""

    @abstractmethod
    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> List[Track]:
        """Consume this frame's detections and return the active tracks."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all state (call between videos)."""