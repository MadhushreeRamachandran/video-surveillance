"""Application logging setup + a run manifest for reproducibility.

Two independent responsibilities on purpose (kept apart from events/observers.py):
  - setup_logging(): configures Python's `logging` module (console + optional file).
  - RunManifest: records exactly how a run was produced — config, environment,
    package versions, timing — so results can be reproduced or debugged later.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# ------------------------------------------------------------------ logging setup
class _JsonFormatter(logging.Formatter):
    """One JSON object per line — greppable and machine-parseable."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    json_format: bool = False,
) -> None:
    """Configure the root logger. Call this once, at CLI startup.

    - Console: human-readable by default; set json_format=True for structured
      stdout (useful when logs are piped into another tool).
    - File (if log_file is given): always JSON lines, and rotates at 5 MB with
      3 backups, so a long-running or looping process can't fill the disk.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()  # calling this twice (e.g. in tests) shouldn't duplicate handlers

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        _JsonFormatter() if json_format
        else logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%H:%M:%S")
    )
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(_JsonFormatter())
        root.addHandler(file_handler)

    # Ultralytics is chatty at INFO; keep it at WARNING unless we're debugging.
    if level.upper() != "DEBUG":
        logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ------------------------------------------------------------------ reproducibility
def _package_versions() -> Dict[str, str]:
    versions = {}
    for name in ("ultralytics", "cv2", "shapely", "deep_sort_realtime", "torch", "numpy"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _cuda_info() -> Dict[str, Any]:
    try:
        import torch
        available = torch.cuda.is_available()
        return {
            "cuda_available": available,
            "gpu_name": torch.cuda.get_device_name(0) if available else None,
        }
    except ImportError:
        return {"cuda_available": False, "gpu_name": None}


@dataclass
class RunManifest:
    """Captures everything needed to reproduce or debug a run: the exact CLI
    config, environment, timing and outcome. Written as JSON next to the
    other outputs (e.g. results/run_manifest.json)."""

    config: Dict[str, Any]
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    video_info: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def finish(self, stats: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        self.ended_at = time.time()
        if stats:
            self.stats.update(stats)
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        duration = (self.ended_at - self.started_at) if self.ended_at else None
        return {
            "config": self.config,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_commit": _git_commit(),
                "packages": _package_versions(),
                **_cuda_info(),
            },
            "video_info": self.video_info,
            "timing": {
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "duration_sec": round(duration, 2) if duration else None,
            },
            "stats": self.stats,
            "error": self.error,
        }

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(p)  # atomic write, same technique as JsonEventLogger in Step 4