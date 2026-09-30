from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    pref = (preference or "auto").lower()

    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if pref.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("cuda requested but not available, falling back to cpu")
        return "cpu"

    return pref


def is_cuda(device: str) -> bool:
    return device.startswith("cuda")


def describe_device(device: str) -> str:
    if is_cuda(device):
        return f"{device} ({torch.cuda.get_device_name(0)})"
    return device
