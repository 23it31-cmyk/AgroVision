"""Lazy YOLO loading; no generic model fallback or weight downloads in inference."""
from pathlib import Path
from threading import Lock
from typing import Any
from PIL import Image


class PestDetector:
    """Serialize inference on a cached, shared Ultralytics model."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError("Custom pest weights are missing: models/pest_yolov8.pt")
        from ultralytics import YOLO
        self.model = YOLO(str(path), task="detect")
        if self.model.task != "detect":
            raise ValueError("A YOLO object detection checkpoint is required.")
        self.lock = Lock()

    def detect(self, image: Image.Image, confidence: float = 0.25) -> tuple[Image.Image, list[dict[str, Any]]]:
        """Return an RGB annotated image and one record per detected object."""
        if not 0.0 < confidence <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        with self.lock:
            result = self.model.predict(source=image, conf=confidence, verbose=False)[0]
            records = []
            if result.boxes is not None:
                for box in result.boxes:
                    records.append({"Pest class": result.names[int(box.cls.item())],
                                    "Confidence (%)": round(float(box.conf.item()) * 100, 2)})
            # Ultralytics plot() returns BGR; Pillow expects RGB.
            annotated = Image.fromarray(result.plot()[..., ::-1].copy())
        return annotated, records
