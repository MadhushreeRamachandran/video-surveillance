from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.validation import explain_validity

logger = logging.getLogger(__name__)

_ZONE_KEYS = {"id", "name", "polygon"}
_RULE_FIELDS = {
    "loiter_seconds",
    "loiter_radius_frac",
    "min_inside_seconds",
    "exit_grace_seconds",
    "cooldown_seconds",
}
_EVENT_FLAGS = {"intrusion": "detect_intrusion", "loitering": "detect_loitering"}


@dataclass(frozen=True)
class ZoneRules:
    detect_intrusion: bool = True
    detect_loitering: bool = True
    loiter_seconds: float = 10.0
    loiter_radius_frac: float = 0.5
    min_inside_seconds: float = 0.3
    exit_grace_seconds: float = 2.0
    cooldown_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.loiter_seconds <= 0:
            raise ValueError("loiter_seconds must be > 0")
        if self.loiter_radius_frac <= 0:
            raise ValueError("loiter_radius_frac must be > 0")
        for name in ("min_inside_seconds", "exit_grace_seconds", "cooldown_seconds"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


class Zone:
    def __init__(
        self,
        zone_id: str,
        name: str,
        points: Sequence[Sequence[float]],
        rules: ZoneRules,
    ) -> None:
        if len(points) < 3:
            raise ValueError(f"zone '{zone_id}' needs at least 3 points")
        polygon = Polygon(points)
        if not polygon.is_valid:
            raise ValueError(f"zone '{zone_id}' polygon is invalid: {explain_validity(polygon)}")
        if polygon.area <= 0:
            raise ValueError(f"zone '{zone_id}' polygon has zero area")

        shapely.prepare(polygon)
        self.id = zone_id
        self.name = name
        self.polygon = polygon
        self.rules = rules
        self.vertices = np.array(points, dtype=np.int32)

    def contains(self, point: Tuple[float, float]) -> bool:
        return bool(shapely.intersects_xy(self.polygon, point[0], point[1]))


def _build_rules(defaults: Dict[str, Any], zone_cfg: Dict[str, Any], where: str) -> ZoneRules:
    merged = {**defaults, **{k: v for k, v in zone_cfg.items() if k not in _ZONE_KEYS}}
    events = merged.pop("events", list(_EVENT_FLAGS))

    bad_events = set(events) - set(_EVENT_FLAGS)
    if bad_events:
        raise ValueError(f"{where}: unknown event(s) {sorted(bad_events)}; use {sorted(_EVENT_FLAGS)}")
    unknown = set(merged) - _RULE_FIELDS
    if unknown:
        raise ValueError(f"{where}: unknown setting(s) {sorted(unknown)}; valid: {sorted(_RULE_FIELDS)}")

    try:
        return ZoneRules(
            detect_intrusion="intrusion" in events,
            detect_loitering="loitering" in events,
            **merged,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc


def load_zones(path: str | Path, frame_size: Optional[Tuple[int, int]] = None) -> List[Zone]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"zones file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p}: invalid JSON ({exc})") from exc

    mode = data.get("coordinate_mode", "pixel")
    if mode not in ("pixel", "normalized"):
        raise ValueError("coordinate_mode must be 'pixel' or 'normalized'")
    if mode == "normalized" and frame_size is None:
        raise ValueError("frame_size is required for normalized coordinates")

    raw_zones = data.get("zones")
    if not raw_zones:
        raise ValueError(f"{p}: no zones defined")
    defaults = data.get("defaults", {})

    zones: List[Zone] = []
    seen_ids = set()
    for i, cfg in enumerate(raw_zones):
        zone_id = str(cfg.get("id", f"zone_{i + 1}"))
        where = f"zone '{zone_id}'"
        if zone_id in seen_ids:
            raise ValueError(f"duplicate zone id '{zone_id}'")
        seen_ids.add(zone_id)

        try:
            points = [(float(x), float(y)) for x, y in cfg["polygon"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where}: 'polygon' must be a list of [x, y] pairs") from exc

        if mode == "normalized":
            if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in points):
                raise ValueError(f"{where}: normalized coordinates must be within [0, 1]")
            w, h = frame_size
            points = [(x * w, y * h) for x, y in points]
        elif frame_size is not None:
            w, h = frame_size
            if any(not (0 <= x <= w and 0 <= y <= h) for x, y in points):
                logger.warning("%s has points outside the %dx%d frame", where, w, h)

        rules = _build_rules(defaults, cfg, where)
        zones.append(Zone(zone_id, str(cfg.get("name", zone_id)), points, rules))

    logger.info("loaded %d zone(s) from %s (%s coordinates)", len(zones), p, mode)
    return zones
