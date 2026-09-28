"""Zone intrusion + loitering engine.

State is kept per (track_id, zone_id). All timing uses VIDEO timestamps.
The engine only needs objects with .track_id, .bbox and .confidence, so it has
no dependency on a specific tracker.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from events.models import Event, EventType
from events.observers import EventPublisher
from events.zones import Zone, ZoneRules

logger = logging.getLogger(__name__)


class _MeanAcc:
    """Running mean of detection confidences (ignores 0 = predicted/coasting boxes)."""

    __slots__ = ("total", "n")

    def __init__(self) -> None:
        self.total = 0.0
        self.n = 0

    def add(self, value: float) -> None:
        if value > 0:
            self.total += value
            self.n += 1

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


@dataclass
class _ZoneState:
    entered_at: float                    # start of this visit
    last_seen: float                     # last time observed inside the zone
    anchor: Tuple[float, float]          # reference position for the stationarity test
    stationary_since: float              # when the person last settled at `anchor`
    radius: float = 0.0
    intrusion_done: bool = False
    loiter_done: bool = False
    visit_conf: _MeanAcc = field(default_factory=_MeanAcc)
    stationary_conf: _MeanAcc = field(default_factory=_MeanAcc)


class ZoneEventEngine:
    def __init__(
        self,
        zones: Sequence[Zone],
        publisher: Optional[EventPublisher] = None,
        anchor: str = "foot",
    ) -> None:
        if not zones:
            raise ValueError("At least one zone is required")
        if anchor not in ("foot", "center"):
            raise ValueError("anchor must be 'foot' or 'center'")

        self.zones = list(zones)
        self.publisher = publisher or EventPublisher()
        self._anchor = anchor
        self._zones_by_id = {z.id: z for z in self.zones}
        self._max_cooldown = max(z.rules.cooldown_seconds for z in self.zones)

        self._states: Dict[Tuple[int, str], _ZoneState] = {}
        self._last_fired: Dict[Tuple[int, str, EventType], float] = {}
        self._next_event_id = 1

    # ------------------------------------------------------------------ API
    def update(self, tracks: Sequence, frame_idx: int, timestamp: float) -> List[Event]:
        """Process one frame. `timestamp` is video time in seconds
        (frame_idx / fps). Returns the events emitted on this frame and also
        publishes them to observers."""
        self._expire(timestamp)  # first, so a long-gone track can't resume an old visit

        emitted: List[Event] = []
        for track in tracks:
            point = self._point(track)
            for zone in self.zones:
                if not zone.contains(point):
                    continue
                rules = zone.rules
                state = self._get_state(track.track_id, zone.id, point, timestamp)
                self._observe(state, track, point, timestamp, rules)

                if rules.detect_intrusion:
                    ev = self._check_intrusion(zone, state, track, frame_idx, timestamp)
                    if ev:
                        emitted.append(ev)
                if rules.detect_loitering:
                    ev = self._check_loitering(zone, state, track, frame_idx, timestamp)
                    if ev:
                        emitted.append(ev)

        self._prune_cooldowns(timestamp)
        return emitted

    def active_alerts(self) -> List[Tuple[int, str, EventType]]:
        """(track_id, zone_id, event_type) currently in alert state; used to
        colour boxes and zones in the annotated video."""
        alerts = []
        for (track_id, zone_id), st in self._states.items():
            if st.intrusion_done:
                alerts.append((track_id, zone_id, EventType.INTRUSION))
            if st.loiter_done:
                alerts.append((track_id, zone_id, EventType.LOITERING))
        return alerts

    def reset(self) -> None:
        self._states.clear()
        self._last_fired.clear()
        self._next_event_id = 1

    # ------------------------------------------------------------- internals
    def _point(self, track) -> Tuple[float, float]:
        x1, y1, x2, y2 = track.bbox
        cx = (x1 + x2) / 2.0
        return (cx, y2) if self._anchor == "foot" else (cx, (y1 + y2) / 2.0)

    def _get_state(self, track_id: int, zone_id: str, point, t: float) -> _ZoneState:
        key = (track_id, zone_id)
        state = self._states.get(key)
        if state is None:
            state = _ZoneState(entered_at=t, last_seen=t, anchor=point, stationary_since=t)
            self._states[key] = state
        return state

    def _observe(self, st: _ZoneState, track, point, t: float, rules: ZoneRules) -> None:
        st.last_seen = t
        st.visit_conf.add(track.confidence)

        # Stationarity: tolerance scales with person height, so it adapts to perspective.
        height = max(track.bbox[3] - track.bbox[1], 1.0)
        st.radius = rules.loiter_radius_frac * height
        if math.dist(point, st.anchor) > st.radius:
            # Person moved: restart the stationary episode from here.
            st.anchor = point
            st.stationary_since = t
            st.loiter_done = False
            st.stationary_conf = _MeanAcc()
        st.stationary_conf.add(track.confidence)

    def _check_intrusion(self, zone, st, track, frame_idx, t) -> Optional[Event]:
        rules = zone.rules
        if st.intrusion_done or (t - st.entered_at) < rules.min_inside_seconds:
            return None
        st.intrusion_done = True  # once per visit, even if suppressed below
        if self._in_cooldown(track.track_id, zone.id, EventType.INTRUSION, t, rules):
            return None
        return self._emit(
            EventType.INTRUSION, zone, track, frame_idx, t, st.visit_conf.mean,
            {"entered_at_sec": round(st.entered_at, 3)},
        )

    def _check_loitering(self, zone, st, track, frame_idx, t) -> Optional[Event]:
        rules = zone.rules
        stationary_for = t - st.stationary_since
        if st.loiter_done or stationary_for < rules.loiter_seconds:
            return None
        st.loiter_done = True  # once per stationary episode
        if self._in_cooldown(track.track_id, zone.id, EventType.LOITERING, t, rules):
            return None
        return self._emit(
            EventType.LOITERING, zone, track, frame_idx, t, st.stationary_conf.mean,
            {
                "stationary_seconds": round(stationary_for, 2),
                "radius_px": round(st.radius, 1),
                "threshold_seconds": rules.loiter_seconds,
            },
        )

    def _in_cooldown(self, track_id, zone_id, etype, t, rules: ZoneRules) -> bool:
        last = self._last_fired.get((track_id, zone_id, etype))
        return last is not None and (t - last) < rules.cooldown_seconds

    def _emit(self, etype, zone, track, frame_idx, t, confidence, details) -> Event:
        event = Event(
            event_id=self._next_event_id,
            event_type=etype,
            zone_id=zone.id,
            zone_name=zone.name,
            track_id=int(track.track_id),
            frame_idx=frame_idx,
            timestamp_sec=t,
            bbox=tuple(float(v) for v in track.bbox),
            confidence=confidence,
            details=details,
        )
        self._next_event_id += 1
        self._last_fired[(track.track_id, zone.id, etype)] = t
        self.publisher.publish(event)
        return event

    def _expire(self, t: float) -> None:
        """Drop states not seen inside their zone for longer than the grace period.
        This bounds memory on long videos."""
        gone = [
            key for key, st in self._states.items()
            if (t - st.last_seen) > self._zones_by_id[key[1]].rules.exit_grace_seconds
        ]
        for key in gone:
            del self._states[key]

    def _prune_cooldowns(self, t: float) -> None:
        stale = [k for k, v in self._last_fired.items() if (t - v) > self._max_cooldown]
        for k in stale:
            del self._last_fired[k]