"""Factory for building detectors by name (Factory + registry pattern)."""
from __future__ import annotations

from typing import Callable, Dict, List, Type

from detectors.base import Detector

_REGISTRY: Dict[str, Type[Detector]] = {}


def register_detector(name: str) -> Callable[[Type[Detector]], Type[Detector]]:
    """Class decorator that registers a detector under `name`."""

    def decorator(cls: Type[Detector]) -> Type[Detector]:
        _REGISTRY[name.lower()] = cls
        return cls

    return decorator


def available_detectors() -> List[str]:
    return sorted(_REGISTRY)


def create_detector(name: str, **kwargs) -> Detector:
    """Build a detector by name; kwargs go to its constructor."""
    key = name.lower()
    try:
        cls = _REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown detector '{name}'. Available: {available_detectors()}"
        ) from None
    return cls(**kwargs)