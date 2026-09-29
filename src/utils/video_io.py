"""Video reading and writing, isolated from detection/tracking/event logic.

Design choices:
  - Frames are streamed one at a time (a generator), never loaded fully into
    memory, so long videos don't blow up RAM.
  - Timestamps always come from the ORIGINAL frame index / source fps, even
    when frames are skipped. Step 4's loitering timers depend on this.
  - Corrupt/unreadable frames are skipped rather than crashing the pipeline
    (handles camera glitches and truncated files).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Fallback order: try the requested codec, then these, before giving up.
_FOURCC_FALLBACKS = ["mp4v", "avc1", "XVID", "MJPG"]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int  # as reported by the container; may be 0/inaccurate for some streams

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def duration_sec(self) -> Optional[float]:
        return (self.frame_count / self.fps) if self.frame_count and self.fps else None


@dataclass(frozen=True)
class FrameData:
    frame_idx: int          # index in the ORIGINAL, unskipped sequence
    timestamp_sec: float    # frame_idx / source_fps — video time, not wall-clock
    image: np.ndarray       # BGR, HxWx3, uint8


class VideoReader:
    """Reads a video and yields (frame_idx, timestamp, image) tuples.

    Use as a context manager so the capture is always released, even on error:
        with VideoReader("clip.mp4", frame_skip=1) as reader:
            for frame in reader:
                ...
    """

    def __init__(
        self,
        path: str | Path,
        frame_skip: int = 0,
        max_consecutive_read_failures: int = 30,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        if frame_skip < 0:
            raise ValueError("frame_skip must be >= 0")

        self.frame_skip = frame_skip
        self.max_consecutive_read_failures = max_consecutive_read_failures

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video (unsupported codec or corrupt file): {self.path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps != fps:  # NaN check; some containers report garbage
            logger.warning("%s reports invalid FPS (%s); assuming 30.0", self.path.name, fps)
            fps = 30.0

        self.info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        logger.info(
            "Opened %s: %dx%d @ %.2f fps (~%d frames)",
            self.path.name, self.info.width, self.info.height, self.info.fps, self.info.frame_count,
        )

        self.frames_read = 0     # frames actually decoded and yielded
        self.frames_skipped = 0  # frames skipped by frame_skip
        self.frames_dropped = 0  # frames that failed to decode

    # ---------------------------------------------------------------- iteration
    def __iter__(self) -> Iterator[FrameData]:
        idx = -1
        consecutive_failures = 0

        while True:
            ok, image = self._cap.read()
            idx += 1

            if not ok:
                consecutive_failures += 1
                self.frames_dropped += 1
                if consecutive_failures >= self.max_consecutive_read_failures:
                    logger.info(
                        "Stopping after %d consecutive failed reads (likely end of stream, at frame %d)",
                        consecutive_failures, idx,
                    )
                    return
                logger.debug("Failed to decode frame %d; skipping", idx)
                continue
            consecutive_failures = 0

            if self.frame_skip and (idx % (self.frame_skip + 1)) != 0:
                self.frames_skipped += 1
                continue

            if image is None or image.size == 0:
                self.frames_dropped += 1
                continue

            self.frames_read += 1
            # Timestamp always from the true frame index, so downstream temporal
            # logic (loitering, cooldowns) is correct regardless of skipping.
            yield FrameData(frame_idx=idx, timestamp_sec=idx / self.info.fps, image=image)

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()
        logger.info(
            "%s: read=%d skipped=%d dropped=%d",
            self.path.name, self.frames_read, self.frames_skipped, self.frames_dropped,
        )

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """Writes annotated frames to a video file, with codec fallback.

    Output FPS defaults to the source FPS (not the skipped/effective rate), so
    played-back video keeps real-world timing even if we skipped frames while
    processing.
    """

    def __init__(
        self,
        path: str | Path,
        frame_size: tuple[int, int],
        fps: float,
        fourcc: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = frame_size  # (width, height)
        self.fps = fps if fps and fps > 0 else 30.0

        candidates = [fourcc] + [c for c in _FOURCC_FALLBACKS if c != fourcc]
        self._writer = None
        self._used_fourcc = None
        for code in candidates:
            writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*code), self.fps, self.frame_size
            )
            if writer.isOpened():
                self._writer, self._used_fourcc = writer, code
                break
            writer.release()

        if self._writer is None:
            raise IOError(
                f"Could not open a video writer for {self.path} with any codec in {candidates}. "
                "Check that OpenCV was built with video support."
            )
        if self._used_fourcc != fourcc:
            logger.warning("Codec '%s' unavailable; using '%s' instead", fourcc, self._used_fourcc)

        self.frames_written = 0

    def write(self, image: np.ndarray) -> None:
        if image.shape[1::-1] != self.frame_size:
            image = cv2.resize(image, self.frame_size)  # guards against an off-by-one annotator bug
        self._writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("Wrote %d frames to %s (codec=%s)", self.frames_written, self.path, self._used_fourcc)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()