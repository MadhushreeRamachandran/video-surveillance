from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FOURCC_FALLBACKS = ["mp4v", "avc1", "XVID", "MJPG"]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def duration_sec(self) -> Optional[float]:
        return (self.frame_count / self.fps) if self.frame_count and self.fps else None


@dataclass(frozen=True)
class FrameData:
    frame_idx: int
    timestamp_sec: float
    image: np.ndarray


class VideoReader:
    def __init__(
        self,
        path: str | Path,
        frame_skip: int = 0,
        max_consecutive_read_failures: int = 30,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")
        if frame_skip < 0:
            raise ValueError("frame_skip must be >= 0")

        self.frame_skip = frame_skip
        self.max_consecutive_read_failures = max_consecutive_read_failures

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"could not open video: {self.path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps != fps:
            logger.warning("%s reports invalid fps (%s), assuming 30.0", self.path.name, fps)
            fps = 30.0

        self.info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        logger.info(
            "opened %s: %dx%d @ %.2f fps (~%d frames)",
            self.path.name, self.info.width, self.info.height, self.info.fps, self.info.frame_count,
        )

        self.frames_read = 0
        self.frames_skipped = 0
        self.frames_dropped = 0

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
                    logger.info("stopping after %d failed reads at frame %d", consecutive_failures, idx)
                    return
                continue
            consecutive_failures = 0

            if self.frame_skip and (idx % (self.frame_skip + 1)) != 0:
                self.frames_skipped += 1
                continue

            if image is None or image.size == 0:
                self.frames_dropped += 1
                continue

            self.frames_read += 1
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
    def __init__(
        self,
        path: str | Path,
        frame_size: tuple[int, int],
        fps: float,
        fourcc: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = frame_size
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
            raise IOError(f"could not open a video writer for {self.path} with any codec in {candidates}")
        if self._used_fourcc != fourcc:
            logger.warning("codec '%s' unavailable, using '%s' instead", fourcc, self._used_fourcc)

        self.frames_written = 0

    def write(self, image: np.ndarray) -> None:
        if image.shape[1::-1] != self.frame_size:
            image = cv2.resize(image, self.frame_size)
        self._writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("wrote %d frames to %s (codec=%s)", self.frames_written, self.path, self._used_fourcc)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
