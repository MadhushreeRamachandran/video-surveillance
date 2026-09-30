from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from pipeline.pipeline import FrameContext, Stage

logger = logging.getLogger(__name__)

_ANALYSIS_WIDTH = 640  


def _small_gray(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = min(1.0, _ANALYSIS_WIDTH / float(w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray, scale


class LowLightStage(Stage):
    def __init__(
        self,
        on_below: float = 85.0,
        off_above: float = 100.0,
        gamma_below: float = 50.0,
        target_mean: float = 110.0,
        clip_limit: float = 2.5,
        tile_grid: int = 8,
        smoothing: float = 0.9,
    ) -> None:
        self.on_below = on_below
        self.off_above = off_above
        self.gamma_below = gamma_below
        self.target_mean = target_mean
        self.smoothing = smoothing
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        self._ema: Optional[float] = None
        self._active = False
        self.frames_enhanced = 0

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.original is None:
            ctx.original = ctx.image

        gray, _ = _small_gray(ctx.image)
        mean = float(gray.mean())
        self._ema = mean if self._ema is None else self.smoothing * self._ema + (1 - self.smoothing) * mean

        if self._active and self._ema > self.off_above:
            self._active = False
            logger.info("low-light enhancement OFF at frame %d (mean=%.0f)", ctx.frame_idx, self._ema)
        elif not self._active and self._ema < self.on_below:
            self._active = True
            logger.info("low-light enhancement ON at frame %d (mean=%.0f)", ctx.frame_idx, self._ema)

        if not self._active:
            return ctx

        lab = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self._ema < self.gamma_below:
            gamma = float(np.clip(
                np.log(self.target_mean / 255.0) / np.log(max(self._ema, 1.0) / 255.0), 0.4, 1.0
            ))
            lut = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
            out = cv2.LUT(out, lut)

        ctx.image = out
        self.frames_enhanced += 1
        return ctx

    def reset(self) -> None:
        self._ema = None
        self._active = False


class StabilizationStage(Stage):
    def __init__(
        self,
        max_shift_frac: float = 0.05,
        decay: float = 0.98,
        max_corners: int = 300,
        min_inliers: int = 12,
    ) -> None:
        self.max_shift_frac = max_shift_frac
        self.decay = decay
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self._prev: Optional[np.ndarray] = None
        self._offset = np.zeros(2, dtype=np.float64)
        self.frames_corrected = 0
        self.frames_unreliable = 0

    def process(self, ctx: FrameContext) -> FrameContext:
        gray, scale = _small_gray(ctx.image)
        delta = self._estimate(gray)
        self._prev = gray

        self._offset = self._offset * self.decay + delta / scale
        h, w = ctx.image.shape[:2]
        limit = np.array([w, h]) * self.max_shift_frac
        self._offset = np.clip(self._offset, -limit, limit)

        if np.abs(self._offset).max() >= 0.5:
            m = np.float32([[1, 0, -self._offset[0]], [0, 1, -self._offset[1]]])
            ctx.image = cv2.warpAffine(
                ctx.image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            self.frames_corrected += 1

        ctx.original = ctx.image
        return ctx

    def _estimate(self, gray: np.ndarray) -> np.ndarray:
        if self._prev is None or self._prev.shape != gray.shape:
            return np.zeros(2)

        pts = cv2.goodFeaturesToTrack(self._prev, self.max_corners, 0.01, 8)
        if pts is None or len(pts) < self.min_inliers:
            self.frames_unreliable += 1
            return np.zeros(2)

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._prev, gray, pts, None)
        ok = status.ravel() == 1
        if ok.sum() < self.min_inliers:
            self.frames_unreliable += 1
            return np.zeros(2)

        m, inliers = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if m is None or inliers is None or int(inliers.sum()) < self.min_inliers:
            self.frames_unreliable += 1
            return np.zeros(2)
        return np.array([m[0, 2], m[1, 2]], dtype=np.float64)

    def reset(self) -> None:
        self._prev = None
        self._offset = np.zeros(2)