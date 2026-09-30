from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from detectors.base import Detection


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    time_since_update: int = 0
    class_name: str = "person"

    @property
    def foot_point(self) -> Tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> List[Track]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
