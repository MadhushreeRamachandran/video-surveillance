from __future__ import annotations

from typing import Callable, Dict, List, Type

from detectors.base import Detector

_REGISTRY: Dict[str, Type[Detector]] = {}


def register_detector(name: str) -> Callable[[Type[Detector]], Type[Detector]]:
    def decorator(cls: Type[Detector]) -> Type[Detector]:
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def available_detectors() -> List[str]:
    return sorted(_REGISTRY)


def create_detector(name: str, **kwargs) -> Detector:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown detector '{name}'. Available: {available_detectors()}")
    return _REGISTRY[key](**kwargs)
