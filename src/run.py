#!/usr/bin/env python3
"""CLI entry point for the video surveillance pipeline.

Usage:
    python run.py --video input.mp4 --zones zones.json --output results/

Pipeline (Pipeline Pattern, see pipeline/pipeline.py):
    video --> detection --> tracking --> event logic --> annotation --> output
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from detectors import create_detector
from events import ConsoleLogObserver, CsvEventLogger, EventPublisher, JsonEventLogger, load_zones
from pipeline import SurveillancePipeline
from trackers import create_tracker
from utils.config import DetectorConfig, OutputConfig, PipelineConfig, TrackerConfig
from utils.device import describe_device, resolve_device
from utils.logger import RunManifest, setup_logging
from utils.video_io import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ CLI parsing
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Video surveillance: person detection, tracking and zone-based event recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    p.add_argument("--video", required=True, help="Path to the input video file.")
    p.add_argument("--zones", required=True, help="Path to the zones JSON config.")
    p.add_argument("--output", required=True, help="Output directory for results.")

    # --- model / tracker selection ---
    p.add_argument("--detector", default="yolov8", choices=["yolov8", "yolo"], help="Detector backend.")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO weights file (auto-downloaded if missing).")
    p.add_argument("--tracker", default="deepsort", choices=["deepsort", "bytetrack"], help="Tracking strategy.")
    p.add_argument("--conf", type=float, default=None,
                   help="Detector confidence threshold. Default depends on --tracker (0.35 DeepSORT / 0.15 ByteTrack).")
    p.add_argument("--iou", type=float, default=0.5, help="Detector NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Detector inference resolution.")

    # --- runtime / performance ---
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda:0", "mps"],
                   help="Compute device. 'auto' picks CUDA/MPS if available, else CPU.")
    p.add_argument("--frame-skip", type=int, default=0,
                   help="Process every (N+1)th frame. 0 = every frame. Speeds up long/CPU runs.")
    p.add_argument("--anchor", default="foot", choices=["foot", "center"],
                   help="Reference point used for zone/loitering tests.")

    # --- output control ---
    p.add_argument("--no-video", action="store_true", help="Skip writing the annotated video (events only, faster).")
    p.add_argument("--no-zones-overlay", action="store_true", help="Don't draw zone polygons on the output video.")
    p.add_argument("--fourcc", default="mp4v", help="Video codec fourcc for the output file.")

    # --- logging / misc ---
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-json", action="store_true", help="Emit console logs as JSON lines.")
    p.add_argument("--quiet", action="store_true", help="Suppress the per-frame console ALERT lines.")

    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        video_path=Path(args.video),
        zones_path=Path(args.zones),
        detector=DetectorConfig(
            name=args.detector, weights=args.weights, conf_threshold=args.conf,
            iou_threshold=args.iou, imgsz=args.imgsz, device=args.device,
        ),
        tracker=TrackerConfig(name=args.tracker, device=args.device),
        output=OutputConfig(
            output_dir=Path(args.output),
            save_video=not args.no_video,
            draw_zones=not args.no_zones_overlay,
            fourcc=args.fourcc,
        ),
        frame_skip=args.frame_skip,
        anchor=args.anchor,
        log_level=args.log_level,
    )


# ------------------------------------------------------------------ wiring
def build_pipeline(config: PipelineConfig, publisher: EventPublisher, video_size: tuple[int, int]) -> SurveillancePipeline:
    """Wires Factory (detector) + Strategy (tracker) + zones + Observer
    (publisher, attached by the caller) into one pipeline instance."""
    detector = create_detector(config.detector.name, **config.detector.as_kwargs())
    tracker = create_tracker(config.tracker.name, **config.tracker.as_kwargs())
    zones = load_zones(config.zones_path, frame_size=video_size)

    logger.info("Detector: %s (weights=%s, conf=%.2f)",
                config.detector.name, config.detector.weights, config.detector.conf_threshold)
    logger.info("Tracker: %s", config.tracker.name)
    logger.info("Zones: %s", ", ".join(f"{z.name}[{z.id}]" for z in zones))

    return SurveillancePipeline(
        detector=detector, tracker=tracker, zones=zones, publisher=publisher,
        anchor=config.anchor, annotate=config.output.save_video, draw_zones=config.output.draw_zones,
    )


def build_observers(config: PipelineConfig, quiet: bool) -> EventPublisher:
    """Observer Pattern: every enabled sink subscribes independently. The
    pipeline/engine never knows which of these exist."""
    publisher = EventPublisher()
    if not quiet:
        publisher.subscribe(ConsoleLogObserver())
    if config.output.save_csv:
        publisher.subscribe(CsvEventLogger(config.output.csv_path))
    if config.output.save_json:
        metadata = {"video": str(config.video_path), "zones": str(config.zones_path),
                    "detector": config.detector.name, "tracker": config.tracker.name}
        publisher.subscribe(JsonEventLogger(config.output.json_path, metadata=metadata))
    return publisher


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)  # raises FileNotFoundError / ValueError on bad input

    config.output.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=config.log_level, log_file=config.output.output_dir / "run.log", json_format=args.log_json)

    logger.info("=" * 60)
    logger.info("Video Surveillance Pipeline starting")
    logger.info("Video: %s | Zones: %s | Output: %s", config.video_path, config.zones_path, config.output.output_dir)
    logger.info("Device: %s", describe_device(resolve_device(args.device)))
    logger.info("=" * 60)

    manifest = RunManifest(config=config.to_dict())
    publisher = build_observers(config, quiet=args.quiet)
    exit_code = 0

    try:
        with VideoReader(config.video_path, frame_skip=config.frame_skip) as reader:
            manifest.video_info = {
                "width": reader.info.width, "height": reader.info.height,
                "fps": reader.info.fps, "frame_count": reader.info.frame_count,
            }

            pipeline = build_pipeline(config, publisher, video_size=reader.info.size)

            writer = None
            if config.output.save_video:
                writer = VideoWriter(config.output.video_path, reader.info.size, reader.info.fps, config.output.fourcc)

            try:
                stats = pipeline.run(reader, writer=writer)
            finally:
                if writer is not None:
                    writer.close()

        manifest.finish(stats=stats.to_dict())
        _print_summary(config, stats)

    except (FileNotFoundError, ValueError, IOError) as exc:
        logger.error("Run failed: %s", exc)
        manifest.finish(error=str(exc))
        exit_code = 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; partial results (if any) were already flushed to disk.")
        manifest.finish(error="interrupted")
        exit_code = 130
    except Exception as exc:  # noqa: BLE001 — last resort so the manifest is still written
        logger.exception("Unexpected error: %s", exc)
        manifest.finish(error=str(exc))
        exit_code = 1
    finally:
        publisher.close()  # flush/close CSV+JSON files even on failure or Ctrl+C
        manifest.write(config.output.manifest_path)

    return exit_code


def _print_summary(config: PipelineConfig, stats) -> None:
    logger.info("=" * 60)
    logger.info("Run complete: %d frames in %.1fs (%.1f fps avg)",
                stats.frames_processed, stats.elapsed_sec, stats.avg_fps)
    logger.info("Events: %d total %s", stats.total_events, stats.events_by_type or "")
    if config.output.save_video:
        logger.info("Annotated video: %s", config.output.video_path)
    if config.output.save_json:
        logger.info("Event log (JSON): %s", config.output.json_path)
    if config.output.save_csv:
        logger.info("Event log (CSV): %s", config.output.csv_path)
    logger.info("Run manifest: %s", config.output.manifest_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())