from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_CONF_BY_TRACKER = {"deepsort": 0.35, "bytetrack": 0.15}

_DEFAULT_TRACKER_PARAMS: Dict[str, Dict[str, Any]] = {
    "deepsort": dict(max_age=90, n_init=3, max_cosine_distance=0.3, nn_budget=100),
    "bytetrack": dict(track_high_thresh=0.4, track_low_thresh=0.1, new_track_thresh=0.4, track_buffer=60),
}


@dataclass
class DetectorConfig:
    name: str = "yolov8"
    weights: str = "yolov8n.pt"
    conf_threshold: Optional[float] = None
    iou_threshold: float = 0.5
    imgsz: int = 640
    device: str = "auto"

    def as_kwargs(self) -> Dict[str, Any]:
        return dict(
            weights=self.weights, conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold, imgsz=self.imgsz, device=self.device,
        )


@dataclass
class TrackerConfig:
    name: str = "deepsort"
    params: Dict[str, Any] = field(default_factory=dict)
    device: str = "auto"

    def as_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self.params)
        if self.name == "deepsort":
            kwargs["device"] = self.device
        return kwargs


@dataclass
class OutputConfig:
    output_dir: Path = Path("outputs")
    save_video: bool = True
    save_json: bool = True
    save_csv: bool = True
    fourcc: str = "mp4v"
    draw_zones: bool = True

    @property
    def video_path(self) -> Path:
        return self.output_dir / "annotated.mp4"

    @property
    def json_path(self) -> Path:
        return self.output_dir / "events.json"

    @property
    def csv_path(self) -> Path:
        return self.output_dir / "events.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"


@dataclass
class PipelineConfig:
    video_path: Path
    zones_path: Path
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    frame_skip: int = 0
    anchor: str = "foot"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        self.zones_path = Path(self.zones_path)
        self.output.output_dir = Path(self.output.output_dir)

        if not self.video_path.is_file():
            raise FileNotFoundError(f"video not found: {self.video_path}")
        if not self.zones_path.is_file():
            raise FileNotFoundError(f"zones file not found: {self.zones_path}")
        if self.frame_skip < 0:
            raise ValueError("frame_skip must be >= 0")
        if self.anchor not in ("foot", "center"):
            raise ValueError("anchor must be 'foot' or 'center'")

        if self.detector.conf_threshold is None:
            self.detector.conf_threshold = _DEFAULT_CONF_BY_TRACKER.get(self.tracker.name, 0.35)
        if not self.tracker.params:
            self.tracker.params = dict(_DEFAULT_TRACKER_PARAMS.get(self.tracker.name, {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "zones_path": str(self.zones_path),
            "detector": {"name": self.detector.name, **self.detector.as_kwargs()},
            "tracker": {"name": self.tracker.name, "params": self.tracker.params},
            "frame_skip": self.frame_skip,
            "anchor": self.anchor,
            "output_dir": str(self.output.output_dir),
        }
