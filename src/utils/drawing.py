"""Frame annotation: track boxes, IDs, zone overlays, alert highlighting, HUD."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from events.models import EventType, format_timecode
from events.zones import Zone
from trackers.base import Track

_ALERT_COLORS = {
    EventType.INTRUSION: (0, 0, 255),     # red (BGR)
    EventType.LOITERING: (0, 140, 255),   # orange
}
_ZONE_COLOR = (255, 200, 0)
_ZONE_ALERT_COLOR = (0, 0, 255)
_ZONE_ALPHA = 0.15
_ID_COLOR_SEED = 37  # arbitrary; just needs to spread IDs across the hue wheel


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministic, visually distinct color per track ID (BGR)."""
    hue = (track_id * _ID_COLOR_SEED) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_zones(frame: np.ndarray, zones: Sequence[Zone], active_zone_ids: Optional[Set[str]] = None) -> np.ndarray:
    """Semi-transparent zone fills + outlines + labels. Zones with an active
    alert (in active_zone_ids) are highlighted in red."""
    active_zone_ids = active_zone_ids or set()
    overlay = frame.copy()
    for zone in zones:
        pts = zone.vertices.reshape(-1, 1, 2)
        color = _ZONE_ALERT_COLOR if zone.id in active_zone_ids else _ZONE_COLOR
        cv2.fillPoly(overlay, [pts], color)
    frame = cv2.addWeighted(overlay, _ZONE_ALPHA, frame, 1 - _ZONE_ALPHA, 0)

    for zone in zones:
        pts = zone.vertices.reshape(-1, 1, 2)
        color = _ZONE_ALERT_COLOR if zone.id in active_zone_ids else _ZONE_COLOR
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        x, y = zone.vertices[0]
        cv2.putText(frame, zone.name, (int(x), max(int(y) - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame


def draw_tracks(frame: np.ndarray, tracks: Sequence[Track], alerts: Dict[int, EventType]) -> np.ndarray:
    """alerts: {track_id: EventType} — the active alert (if any) that colors
    that person's box, overriding their default per-ID color."""
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        alert = alerts.get(t.track_id)
        color = _ALERT_COLORS[alert] if alert else _color_for_id(t.track_id)
        thickness = 3 if alert else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"ID {t.track_id}" + (f" | {alert.value.upper()}" if alert else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, label_y), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Foot point marker: shows exactly what the zone/loiter logic tests against.
        fx, fy = t.foot_point
        cv2.circle(frame, (int(fx), int(fy)), 3, color, -1)
    return frame


def draw_hud(frame: np.ndarray, frame_idx: int, timestamp_sec: float,
             fps_actual: float, active_tracks: int, total_events: int) -> np.ndarray:
    """Two-line status overlay, top-left. White text with a black outline so
    it stays legible over any background."""
    lines = [
        f"Frame {frame_idx} | t={format_timecode(timestamp_sec)}",
        f"FPS: {fps_actual:.1f} | Tracks: {active_tracks} | Events: {total_events}",
    ]
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return frame


def annotate_frame(
    frame: np.ndarray,
    tracks: Sequence[Track],
    zones: Sequence[Zone],
    active_alerts: Sequence[Tuple[int, str, EventType]],
    frame_idx: int,
    timestamp_sec: float,
    fps_actual: float,
    total_events: int,
    draw_zones_flag: bool = True,
) -> np.ndarray:
    """Single entry point the pipeline calls. Never mutates the input frame."""
    out = frame.copy()
    active_zone_ids = {zid for (_tid, zid, _etype) in active_alerts}

    if draw_zones_flag:
        out = draw_zones(out, zones, active_zone_ids)

    # One alert per track; prefer LOITERING if both are active (it's the
    # longer-running, generally more significant condition).
    per_track_alert: Dict[int, EventType] = {}
    for track_id, _zone_id, etype in active_alerts:
        if track_id not in per_track_alert or etype == EventType.LOITERING:
            per_track_alert[track_id] = etype

    out = draw_tracks(out, tracks, per_track_alert)
    out = draw_hud(out, frame_idx, timestamp_sec, fps_actual, len(tracks), total_events)
    return out