"""Detector interface and the shared Detection data class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame (pixel coordinates)."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
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
        """Bottom-centre of the box: where the person touches the ground.
        Used for zone tests, since it is more accurate than the box centre."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def to_ltwh(self) -> Tuple[float, float, float, float]:
        """(left, top, width, height): the format DeepSORT expects."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)


class Detector(ABC):
    """Interface every detector implements, so the pipeline never depends
    on a specific model."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in one BGR frame (H, W, 3, uint8)."""

    def detect_batch(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Default: loop over detect(). Subclasses override for real batching."""
        return [self.detect(f) for f in frames]