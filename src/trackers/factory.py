from __future__ import annotations

from typing import Callable, Dict, List, Type

from trackers.base import Tracker

_REGISTRY: Dict[str, Type[Tracker]] = {}


def register_tracker(name: str) -> Callable[[Type[Tracker]], Type[Tracker]]:
    def decorator(cls: Type[Tracker]) -> Type[Tracker]:
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def available_trackers() -> List[str]:
    return sorted(_REGISTRY)


def create_tracker(name: str, **kwargs) -> Tracker:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown tracker '{name}'. Available: {available_trackers()}")
    return _REGISTRY[key](**kwargs)
