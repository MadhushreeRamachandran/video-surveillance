"""Pipeline Pattern: video -> detection -> tracking -> events -> annotation -> output.

FrameContext threads through an ordered list of stages. Each stage reads
what it needs from the context and writes its own result back onto it;
no stage imports another stage's internals.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from detectors.base import Detection, Detector
from events.models import Event
from events.observers import EventPublisher
from events.zone_logic import ZoneEventEngine
from events.zones import Zone
from trackers.base import Track, Tracker
from utils.drawing import annotate_frame
from utils.video_io import FrameData, VideoWriter

logger = logging.getLogger(__name__)


@dataclass
class FrameContext:
    frame_idx: int
    timestamp_sec: float
    image: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    annotated: Optional[np.ndarray] = None
    process_time_sec: float = 0.0  # wall-clock time to run detect+track+events on this frame


class Stage:
    """Base class for a pipeline stage. Subclasses implement process()."""

    def process(self, ctx: FrameContext) -> FrameContext:
        raise NotImplementedError


class DetectionStage(Stage):
    def __init__(self, detector: Detector) -> None:
        self.detector = detector

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.detections = self.detector.detect(ctx.image)
        return ctx


class TrackingStage(Stage):
    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.tracks = self.tracker.update(ctx.detections, ctx.image)
        return ctx


class EventStage(Stage):
    def __init__(self, engine: ZoneEventEngine) -> None:
        self.engine = engine

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.events = self.engine.update(ctx.tracks, ctx.frame_idx, ctx.timestamp_sec)
        return ctx


class AnnotationStage(Stage):
    """Draws overlays. Skips the (comparatively expensive) drawing work
    entirely if no video output was requested."""

    def __init__(self, zones: Sequence[Zone], engine: ZoneEventEngine, draw_zones: bool, enabled: bool) -> None:
        self.zones = zones
        self.engine = engine
        self.draw_zones = draw_zones
        self.enabled = enabled
        self._total_events = 0
        self._last_fps = 0.0

    def note_totals(self, total_events: int, fps: float) -> None:
        self._total_events = total_events
        self._last_fps = fps

    def process(self, ctx: FrameContext) -> FrameContext:
        if not self.enabled:
            return ctx
        ctx.annotated = annotate_frame(
            frame=ctx.image,
            tracks=ctx.tracks,
            zones=self.zones,
            active_alerts=self.engine.active_alerts(),
            frame_idx=ctx.frame_idx,
            timestamp_sec=ctx.timestamp_sec,
            fps_actual=self._last_fps,
            total_events=self._total_events,
            draw_zones_flag=self.draw_zones,
        )
        return ctx


@dataclass
class PipelineStats:
    frames_processed: int = 0
    total_detections: int = 0
    total_events: int = 0
    events_by_type: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    avg_fps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "total_detections": self.total_detections,
            "total_events": self.total_events,
            "events_by_type": self.events_by_type,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "avg_fps": round(self.avg_fps, 2),
        }


class SurveillancePipeline:
    """Orchestrates the stages over a stream of frames.

    Built from already-constructed components (detector, tracker, zones,
    publisher) rather than building them itself -- that's the Factory's job
    and run.py's job, keeping this class testable with fakes and free of
    CLI/config concerns.
    """

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        zones: Sequence[Zone],
        publisher: EventPublisher,
        anchor: str = "foot",
        annotate: bool = True,
        draw_zones: bool = True,
    ) -> None:
        self.engine = ZoneEventEngine(zones, publisher=publisher, anchor=anchor)
        self.stages: List[Stage] = [
            DetectionStage(detector),
            TrackingStage(tracker),
            EventStage(self.engine),
        ]
        self.annotation_stage = AnnotationStage(zones, self.engine, draw_zones, enabled=annotate)
        self.stages.append(self.annotation_stage)
        self.stats = PipelineStats()

    def run(self, frames: Sequence[FrameData], writer: Optional[VideoWriter] = None) -> PipelineStats:
        """Process an iterable/sequence of FrameData. Streams -- never
        materializes more than one frame's context at a time, and each
        annotated frame is written and discarded immediately, so memory
        stays flat regardless of video length."""
        start = time.perf_counter()
        events_by_type: dict = {}

        for frame_data in frames:
            frame_start = time.perf_counter()
            ctx = FrameContext(frame_data.frame_idx, frame_data.timestamp_sec, frame_data.image)

            running_fps = self.stats.frames_processed / max(time.perf_counter() - start, 1e-9)
            self.annotation_stage.note_totals(self.stats.total_events, running_fps)

            for stage in self.stages:
                ctx = stage.process(ctx)

            ctx.process_time_sec = time.perf_counter() - frame_start

            self.stats.frames_processed += 1
            self.stats.total_detections += len(ctx.detections)
            self.stats.total_events += len(ctx.events)
            for ev in ctx.events:
                key = ev.event_type.value
                events_by_type[key] = events_by_type.get(key, 0) + 1

            if writer is not None and ctx.annotated is not None:
                writer.write(ctx.annotated)

            if self.stats.frames_processed % 200 == 0:
                logger.info(
                    "Processed %d frames (%.1f fps, %d events so far)",
                    self.stats.frames_processed, running_fps, self.stats.total_events,
                )

        self.stats.elapsed_sec = time.perf_counter() - start
        self.stats.events_by_type = events_by_type
        self.stats.avg_fps = (
            self.stats.frames_processed / self.stats.elapsed_sec if self.stats.elapsed_sec > 0 else 0.0
        )
        return self.stats

    def reset(self) -> None:
        """Reuse this pipeline for another video (clears tracker + event state)."""
        self.engine.reset()
        for stage in self.stages:
            if isinstance(stage, TrackingStage):
                stage.tracker.reset()