"""Computer-vision engine: a thin wrapper around Ultralytics YOLOv8.

Inference uses ``stream=True`` (lazy generator output) so per-frame results
are released immediately instead of accumulating in memory.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import DEFAULT_CONF, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# The 80 COCO labels YOLOv8 is trained on (used by the UI class filter).
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# BGR colour palette; a class always maps to the same colour.
_PALETTE: List[Tuple[int, int, int]] = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]

Detection = Dict[str, object]


class YOLOObjectDetector:
    """Loads a YOLOv8 model and turns BGR frames into annotated frames."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        # Imported lazily: ultralytics pulls in PyTorch, which the UI and
        # the unit tests do not need just to import this module.
        from ultralytics import YOLO

        logger.info("Loading YOLO weights: %s", model_path)
        self.model = YOLO(model_path)  # downloads yolov8n.pt on first use
        self.tracker_config = tracker_config
        self.class_names: Dict[int, str] = dict(self.model.names)
        self._name_to_id = {
            name.lower(): idx for idx, name in self.class_names.items()
        }
        self._warmup()

    # --------------------------------------------------------------- helpers
    def _warmup(self) -> None:
        """Run one dummy inference so the first real frame is not slow."""
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in self.model.predict(dummy, verbose=False, stream=True):
                pass
        except Exception as exc:  # warm-up must never block start-up
            logger.warning("Model warm-up skipped: %s", exc)

    def reset_tracking(self) -> None:
        """Forget tracker state (call when the video source changes)."""
        try:
            self.model.predictor = None
        except Exception as exc:
            logger.warning("Could not reset tracker: %s", exc)

    def _resolve_class_ids(
        self, allowed_classes: Optional[Sequence[str]]
    ) -> Optional[List[int]]:
        """Map class names to model ids; ``None`` means "all classes"."""
        if not allowed_classes:
            return None
        return [
            self._name_to_id[name.lower()]
            for name in allowed_classes
            if name.lower() in self._name_to_id
        ]

    # ------------------------------------------------------------- inference
    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = DEFAULT_CONF,
        allowed_classes: Optional[Sequence[str]] = None,
        track: bool = False,
    ) -> Tuple[np.ndarray, List[Detection]]:
        """Detect objects in a BGR frame.

        Args:
            frame: BGR image as produced by ``cv2.VideoCapture.read``.
            conf_threshold: minimum confidence (0-1) to keep a prediction.
            allowed_classes: class names to keep; empty/None keeps all.
            track: enable ByteTrack so each detection carries a track id.

        Returns:
            ``(annotated_rgb_frame, detections)`` where each detection is a
            dict with ``class_id``, ``class_name``, ``confidence``, pixel
            ``x``/``y`` (top-left corner), ``w``/``h`` and ``track_id``.
        """
        if frame is None or frame.size == 0:
            raise ValueError("process_frame received an empty frame")

        annotated = frame.copy()
        detections: List[Detection] = []
        class_ids = self._resolve_class_ids(allowed_classes)

        # Filter requested but none of the names exist in the model.
        if class_ids is not None and not class_ids:
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), detections

        options = dict(
            conf=float(conf_threshold),
            classes=class_ids,
            verbose=False,
            stream=True,
        )
        if track:
            results = self.model.track(
                frame, persist=True, tracker=self.tracker_config, **options
            )
        else:
            results = self.model.predict(frame, **options)

        for result in results:  # generator: one item per image
            detections.extend(self._extract_detections(result))

        for det in detections:
            self._draw_detection(annotated, det)

        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), detections

    def _extract_detections(self, result) -> List[Detection]:
        """Convert one Ultralytics result into plain dictionaries."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            boxes.id.cpu().numpy().astype(int)
            if getattr(boxes, "id", None) is not None
            else [None] * len(xyxy)
        )

        extracted: List[Detection] = []
        for (x1, y1, x2, y2), conf, cid, tid in zip(
            xyxy, confs, class_ids, track_ids
        ):
            extracted.append(
                {
                    "class_id": int(cid),
                    "class_name": self.class_names.get(int(cid), str(cid)),
                    "confidence": float(conf),
                    "x": float(x1),
                    "y": float(y1),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                    "track_id": None if tid is None else int(tid),
                }
            )
        return extracted

    @staticmethod
    def _draw_detection(image: np.ndarray, det: Detection) -> None:
        """Draw a styled box plus a filled label chip onto ``image``."""
        height, width = image.shape[:2]
        x1 = max(0, int(det["x"]))
        y1 = max(0, int(det["y"]))
        x2 = min(width - 1, int(det["x"] + det["w"]))
        y2 = min(height - 1, int(det["y"] + det["h"]))
        color = _PALETTE[int(det["class_id"]) % len(_PALETTE)]

        thickness = max(2, width // 400)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = f"{det['class_name']} {float(det['confidence']):.0%}"
        if det.get("track_id") is not None:
            label = f"#{det['track_id']} {label}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, width / 1600)
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, 1)
        chip_top = max(0, y1 - text_h - baseline - 6)
        cv2.rectangle(
            image,
            (x1, chip_top),
            (min(width - 1, x1 + text_w + 8), chip_top + text_h + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 4, chip_top + text_h + 2),
            font,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
