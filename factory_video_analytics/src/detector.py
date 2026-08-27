"""High-performance YOLO26 Person/Worker Detector for Factory Activity Recognition across 5 scales."""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)

# YOLO26 Scale Mapping & Specialized Factory Model
YOLO26_SCALES = {
    "factory_person_yolo26n_best.pt": {
        "name": "★ Specialized Factory YOLO26 (Fine-Tuned)",
        "scale": "SPECIALIZED",
        "fallback": "yolo26n.pt",
        "desc": "Custom fine-tuned weights optimized specifically for factory workers",
        "is_specialized": True,
    },
    "yolo26n.pt": {"name": "YOLO26 Nano", "scale": "NANO", "fallback": "yolo11n.pt", "desc": "Lightweight model for maximum FPS", "is_specialized": False},
    "yolo26s.pt": {"name": "YOLO26 Small", "scale": "SMALL", "fallback": "yolo11s.pt", "desc": "Optimal balance of speed and accuracy", "is_specialized": False},
    "yolo26m.pt": {"name": "YOLO26 Medium", "scale": "MEDIUM", "fallback": "yolo11m.pt", "desc": "Higher precision for busy factory floors", "is_specialized": False},
    "yolo26l.pt": {"name": "YOLO26 Large", "scale": "LARGE", "fallback": "yolo11l.pt", "desc": "High capacity model for dense environments", "is_specialized": False},
    "yolo26x.pt": {"name": "YOLO26 X-Large", "scale": "XLARGE", "fallback": "yolo11x.pt", "desc": "Maximum accuracy and detection range", "is_specialized": False},
}


@dataclass
class Detection:
    """Represents a single detected person/worker in a frame."""
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[float, float]
    is_person: bool = True

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class YOLO26PersonDetector:
    """YOLO26 detector focused exclusively on robust person/worker detection and activity recognition across scales."""

    def __init__(
        self,
        checkpoint: str = "factory_person_yolo26n_best.pt",
        fallback_checkpoint: str = "yolo26s.pt",
        person_confidence: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "auto",
        **kwargs,
    ):
        self.checkpoint = checkpoint if checkpoint in YOLO26_SCALES else "factory_person_yolo26n_best.pt"
        self.fallback_checkpoint = fallback_checkpoint
        self.person_confidence = person_confidence
        self.iou_threshold = iou_threshold
        self.device = self._resolve_device(device)

        # Diagnostics
        self.total_raw_detections = 0
        self.total_accepted_detections = 0

        self.model = self._load_model()

    def _resolve_device(self, device_str: str) -> str:
        if device_str == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        elif str(device_str).lower() in ("cuda", "gpu", "0"):
            return "0" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _find_checkpoint(self, name: str) -> str:
        p = Path(name)
        if p.exists():
            return str(p.resolve())
        p_parent = Path("..") / name
        if p_parent.exists():
            return str(p_parent.resolve())
        return name

    def _load_model(self) -> Any:
        from ultralytics import YOLO

        scale_info = YOLO26_SCALES.get(self.checkpoint, YOLO26_SCALES["factory_person_yolo26n_best.pt"])
        target = self._find_checkpoint(self.checkpoint)
        fallback = scale_info["fallback"]

        # 1. Try local target checkpoint (e.g. factory_person_yolo26n_best.pt)
        if Path(target).exists():
            try:
                logger.info(f"Loading local checkpoint: {target}")
                model = YOLO(target)
                logger.info(f"Successfully loaded {scale_info['name']} on device {self.device}")
                return model
            except Exception as e:
                logger.warning(f"Could not load local {target}: {e}")

        # 2. Try standard weights or fallback
        try:
            logger.info(f"Loading {scale_info['name']} ({self.checkpoint})...")
            model = YOLO(self.checkpoint)
            logger.info(f"Successfully loaded {scale_info['name']} ({self.checkpoint}) on device {self.device}")
            return model
        except Exception:
            logger.info(f"Falling back to scale weights: {fallback}")
            model = YOLO(fallback)
            logger.info(f"Successfully loaded {scale_info['name']} ({fallback}) on device {self.device}")
            return model

    def switch_model(self, checkpoint: str) -> bool:
        """Dynamically switches active detector model."""
        if checkpoint not in YOLO26_SCALES:
            checkpoint = "factory_person_yolo26n_best.pt"

        self.checkpoint = checkpoint
        scale_info = YOLO26_SCALES[checkpoint]
        logger.info(f"Switching to {scale_info['name']} ({checkpoint})...")

        try:
            self.model = self._load_model()
            logger.info(f"Successfully activated {scale_info['name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch model to {checkpoint}: {e}")
            raise e

    def set_confidence(self, confidence: float) -> float:
        """Dynamically updates the person detection confidence threshold (0.05 to 0.95)."""
        self.person_confidence = max(0.05, min(0.95, float(confidence)))
        logger.info(f"Updated person detection confidence threshold to: {self.person_confidence:.2f}")
        return self.person_confidence

    def get_model_info(self) -> Dict[str, Any]:
        """Returns active model metadata with specialized status."""
        scale_info = YOLO26_SCALES.get(self.checkpoint, YOLO26_SCALES["factory_person_yolo26n_best.pt"])
        return {
            "model_family": "YOLO26",
            "active_model": self.checkpoint,
            "name": scale_info["name"],
            "scale": scale_info["scale"],
            "is_specialized": scale_info.get("is_specialized", False),
            "target_class": "person",
            "device": self.device,
            "person_confidence": round(self.person_confidence, 2),
            "supported_scales": [
                {
                    "id": k,
                    "name": v["name"],
                    "scale": v["scale"],
                    "is_specialized": v.get("is_specialized", False),
                    "description": v["desc"],
                }
                for k, v in YOLO26_SCALES.items()
            ]
        }

    def detect_and_filter(
        self,
        frame: np.ndarray,
        zone_engine: Optional[Any] = None,
    ) -> List[Detection]:
        """Runs fast person-only detection with confidence filtering."""
        h, w = frame.shape[:2]

        try:
            # classes=[0] filters inference strictly to 'person'
            results = self.model.predict(
                source=frame,
                classes=[0],
                conf=self.person_confidence,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Person detection error: {e}")
            return []

        if not results or len(results) == 0:
            return []

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        detections: List[Detection] = []

        for i in range(len(xyxy)):
            box = (float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3]))
            conf = float(confs[i])
            cls_id = int(cls_ids[i])

            if conf < self.person_confidence:
                continue

            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            det = Detection(
                box=box,
                confidence=conf,
                class_id=cls_id,
                class_name="person",
                center=(cx, cy),
                is_person=True,
            )

            self.total_raw_detections += 1
            self.total_accepted_detections += 1
            detections.append(det)

        return detections


# Backwards compatibility aliases
YOLOEDetector = YOLO26PersonDetector
YOLODetector = YOLO26PersonDetector
