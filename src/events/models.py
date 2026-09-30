from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple


class EventType(str, Enum):
    INTRUSION = "zone_intrusion"
    LOITERING = "loitering"


def format_timecode(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass(frozen=True)
class Event:
    event_id: int
    event_type: EventType
    zone_id: str
    zone_name: str
    track_id: int
    frame_idx: int
    timestamp_sec: float
    bbox: Tuple[float, float, float, float]
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "track_id": self.track_id,
            "frame": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "timecode": format_timecode(self.timestamp_sec),
            "bbox": [round(v, 1) for v in self.bbox],
            "confidence": round(self.confidence, 3),
            "details": self.details,
        }
