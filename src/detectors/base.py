from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "person"

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def foot_point(self) -> Tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def to_ltwh(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def detect_batch(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        return [self.detect(f) for f in frames]
