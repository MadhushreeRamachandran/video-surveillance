"""YOLOv8 person detector (Ultralytics)."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
from ultralytics import YOLO

from detectors.base import Detection, Detector
from detectors.factory import register_detector
from utils.device import describe_device, is_cuda, resolve_device

logger = logging.getLogger(__name__)


@register_detector("yolo")
@register_detector("yolov8")
class YoloDetector(Detector):
    """Person-only detector built on pretrained COCO YOLOv8 weights.

    Weights are downloaded automatically on first use (e.g. yolov8n.pt).
    Swap model size by changing `weights` (yolov8s.pt, yolov8m.pt, ...).
    """

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        device: str = "auto",
        half: Optional[bool] = None,
        max_det: int = 100,
        min_box_area: float = 100.0,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.min_box_area = min_box_area

        self.device = resolve_device(device)
        # FP16 only helps (and is only reliable) on CUDA.
        self.half = is_cuda(self.device) if half is None else (half and is_cuda(self.device))

        logger.info("Loading %s on %s (half=%s)", weights, describe_device(self.device), self.half)
        self._model = YOLO(weights)
        self._warmup()

    # ------------------------------------------------------------------ API
    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:  # empty/corrupt frame -> no detections
            return []
        result = self._model.predict(frame, **self._predict_kwargs())[0]
        return self._to_detections(result)

    def detect_batch(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Run one forward pass on several frames (better GPU utilisation).
        Output stays index-aligned with the input, even if some frames are empty."""
        valid = [i for i, f in enumerate(frames) if f is not None and f.size > 0]
        outputs: List[List[Detection]] = [[] for _ in frames]
        if not valid:
            return outputs

        results = self._model.predict([frames[i] for i in valid], **self._predict_kwargs())
        for idx, result in zip(valid, results):
            outputs[idx] = self._to_detections(result)
        return outputs

    # ------------------------------------------------------------- internals
    def _predict_kwargs(self) -> dict:
        return dict(
            classes=[self.PERSON_CLASS_ID],  # filter to people inside the model
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            max_det=self.max_det,
            verbose=False,
        )

    def _to_detections(self, result) -> List[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        detections: List[Detection] = []
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
            if (x2 - x1) * (y2 - y1) < self.min_box_area:  # drop tiny noise boxes
                continue
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(c),
                    class_id=self.PERSON_CLASS_ID,
                    class_name="person",
                )
            )
        return detections

    def _warmup(self) -> None:
        """One dummy inference so the first real frame isn't slow (this matters
        for FPS benchmarks)."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._model.predict(dummy, **self._predict_kwargs())