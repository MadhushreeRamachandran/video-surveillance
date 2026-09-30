from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from events.models import Event, format_timecode

logger = logging.getLogger(__name__)


class EventObserver(ABC):
    @abstractmethod
    def on_event(self, event: Event) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class EventPublisher:
    def __init__(self) -> None:
        self._observers: List[EventObserver] = []

    def subscribe(self, observer: EventObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def unsubscribe(self, observer: EventObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def publish(self, event: Event) -> None:
        for observer in list(self._observers):
            try:
                observer.on_event(event)
            except Exception:
                logger.exception("observer %s failed on event %s", type(observer).__name__, event.event_id)

    def close(self) -> None:
        for observer in self._observers:
            try:
                observer.close()
            except Exception:
                logger.exception("observer %s failed to close", type(observer).__name__)


class ConsoleLogObserver(EventObserver):
    def on_event(self, event: Event) -> None:
        logger.warning(
            "alert %-14s zone=%s track=%d frame=%d t=%s conf=%.2f",
            event.event_type.value,
            event.zone_id,
            event.track_id,
            event.frame_idx,
            format_timecode(event.timestamp_sec),
            event.confidence,
        )


class InMemoryObserver(EventObserver):
    def __init__(self) -> None:
        self.events: List[Event] = []

    def on_event(self, event: Event) -> None:
        self.events.append(event)


class CsvEventLogger(EventObserver):
    FIELDS = [
        "event_id", "event_type", "zone_id", "zone_name", "track_id", "frame",
        "timestamp_sec", "timecode", "x1", "y1", "x2", "y2", "confidence", "details",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._fh.flush()

    def on_event(self, event: Event) -> None:
        d = event.to_dict()
        x1, y1, x2, y2 = d["bbox"]
        self._writer.writerow({
            "event_id": d["event_id"], "event_type": d["event_type"],
            "zone_id": d["zone_id"], "zone_name": d["zone_name"],
            "track_id": d["track_id"], "frame": d["frame"],
            "timestamp_sec": d["timestamp_sec"], "timecode": d["timecode"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": d["confidence"], "details": json.dumps(d["details"]),
        })
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class JsonEventLogger(EventObserver):
    def __init__(self, path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self._events: List[Dict[str, Any]] = []
        self._closed = False

    def on_event(self, event: Event) -> None:
        self._events.append(event.to_dict())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        payload = {
            "metadata": self.metadata,
            "summary": {
                "total_events": len(self._events),
                "by_type": dict(Counter(e["event_type"] for e in self._events)),
            },
            "events": self._events,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)
