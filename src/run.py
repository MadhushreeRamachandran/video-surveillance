from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from detectors import create_detector
from events import ConsoleLogObserver, CsvEventLogger, EventPublisher, JsonEventLogger, load_zones
from pipeline import LowLightStage, StabilizationStage, SurveillancePipeline
from trackers import create_tracker
from utils.config import DetectorConfig, OutputConfig, PipelineConfig, PreprocessConfig, TrackerConfig
from utils.device import describe_device, resolve_device
from utils.logger import RunManifest, setup_logging
from utils.video_io import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Video surveillance: person detection, tracking and zone-based event recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--video", required=True, help="Path to the input video file.")
    p.add_argument("--zones", required=True, help="Path to the zones JSON config.")
    p.add_argument("--output", required=True, help="Output directory for results.")

    p.add_argument("--detector", default="yolov8", choices=["yolov8", "yolo"], help="Detector backend.")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO weights file.")
    p.add_argument("--tracker", default="deepsort", choices=["deepsort", "bytetrack"], help="Tracking strategy.")
    p.add_argument("--conf", type=float, default=None, help="Detector confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5, help="Detector NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Detector inference resolution.")

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda:0", "mps"], help="Compute device.")
    p.add_argument("--frame-skip", type=int, default=0, help="Process every (N+1)th frame.")
    p.add_argument("--anchor", default="foot", choices=["foot", "center"], help="Reference point for zone tests.")
    p.add_argument("--stabilize", action="store_true", help="Compensate camera shake before detection.")
    p.add_argument("--no-lowlight", action="store_true", help="Disable adaptive low-light enhancement.")
    p.add_argument("--no-video", action="store_true", help="Skip writing the annotated video.")
    p.add_argument("--no-zones-overlay", action="store_true", help="Don't draw zone polygons on the output video.")
    p.add_argument("--fourcc", default="mp4v", help="Video codec fourcc for the output file.")

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-json", action="store_true", help="Emit console logs as JSON lines.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-frame console alert lines.")

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
        preprocess=PreprocessConfig(low_light=not args.no_lowlight, stabilize=args.stabilize),
    )


def build_pipeline(config: PipelineConfig, publisher: EventPublisher, video_size: tuple[int, int]) -> SurveillancePipeline:
    detector = create_detector(config.detector.name, **config.detector.as_kwargs())
    tracker = create_tracker(config.tracker.name, **config.tracker.as_kwargs())
    zones = load_zones(config.zones_path, frame_size=video_size)
    preprocessors = []
    if config.preprocess.stabilize:
        preprocessors.append(StabilizationStage())
    if config.preprocess.low_light:
        preprocessors.append(LowLightStage())
    logger.info("preprocess: %s", ", ".join(type(s).__name__ for s in preprocessors) or "none")

    logger.info("detector: %s (weights=%s, conf=%.2f)",
                config.detector.name, config.detector.weights, config.detector.conf_threshold)
    logger.info("tracker: %s", config.tracker.name)
    logger.info("zones: %s", ", ".join(f"{z.name}[{z.id}]" for z in zones))

    return SurveillancePipeline(
        detector=detector, tracker=tracker, zones=zones, publisher=publisher,
        anchor=config.anchor, annotate=config.output.save_video, draw_zones=config.output.draw_zones,
        preprocessors=preprocessors,
    )

def build_observers(config: PipelineConfig, quiet: bool) -> EventPublisher:
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


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    config.output.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=config.log_level, log_file=config.output.output_dir / "run.log", json_format=args.log_json)

    logger.info("=" * 60)
    logger.info("video surveillance pipeline starting")
    logger.info("video: %s | zones: %s | output: %s", config.video_path, config.zones_path, config.output.output_dir)
    logger.info("device: %s", describe_device(resolve_device(args.device)))
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
        logger.error("run failed: %s", exc)
        manifest.finish(error=str(exc))
        exit_code = 1
    except KeyboardInterrupt:
        logger.warning("interrupted by user, partial results were flushed to disk")
        manifest.finish(error="interrupted")
        exit_code = 130
    except Exception as exc:
        logger.exception("unexpected error: %s", exc)
        manifest.finish(error=str(exc))
        exit_code = 1
    finally:
        publisher.close()
        manifest.write(config.output.manifest_path)

    return exit_code


def _print_summary(config: PipelineConfig, stats) -> None:
    logger.info("=" * 60)
    logger.info("run complete: %d frames in %.1fs (%.1f fps avg)",
                stats.frames_processed, stats.elapsed_sec, stats.avg_fps)
    logger.info("events: %d total %s", stats.total_events, stats.events_by_type or "")
    if config.output.save_video:
        logger.info("annotated video: %s", config.output.video_path)
    if config.output.save_json:
        logger.info("event log (json): %s", config.output.json_path)
    if config.output.save_csv:
        logger.info("event log (csv): %s", config.output.csv_path)
    logger.info("run manifest: %s", config.output.manifest_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
