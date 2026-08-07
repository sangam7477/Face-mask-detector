
import os
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from .alert import AlertManager
from .face_detector import FaceDetector
from .preprocess import preprocess_face


@dataclass
class Detection:
    box: Tuple[int, int, int, int]
    label: str
    confidence: float


class MaskDetector:
    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str] = None,
        config_path: Optional[str] = None,
        cascade_path: Optional[str] = None,
        input_size: Tuple[int, int] = (224, 224),
        alert: bool = True,
        alert_cooldown: float = 2.0,
        screenshot_dir: Optional[str] = None,
        screenshot_cooldown: float = 5.0,
    ):
    
       
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}. Run train_model.py first."
            )

        if os.path.getsize(model_path) == 0:
            raise ValueError(
                f"Model file is empty at: {model_path}."
            )

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        config = _load_config(config_path, model_path)

        self.labels = _load_labels(labels_path, config)

        self.face_detector = FaceDetector(cascade_path)

        self.input_size = _resolve_input_size(
            config,
            input_size
        )

        self.alert_manager = AlertManager(
            enabled=alert,
            cooldown=alert_cooldown
        )

        self.screenshot_dir = screenshot_dir
        self.screenshot_cooldown = screenshot_cooldown
        self._last_screenshot = 0

        self.total_mask = 0
        self.total_no_mask = 0

        # CSV Log
        self.log_file = "logs/attendance.csv"
        os.makedirs("logs", exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Date", "Time", "Status", "Confidence"]
                )

        self._last_log = 0
        self.log_cooldown = 5

        if self.screenshot_dir:
            os.makedirs(
                self.screenshot_dir,
                exist_ok=True
            )



    def save_log(self, status, confidence):
        now = datetime.now()

        if time.time() - self._last_log < self.log_cooldown:
            return

        self._last_log = time.time()

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                status,
                f"{confidence*100:.2f}%"
            ])

    def _maybe_save_screenshot(self, frame):
        if self.screenshot_dir is None:
            return

        now = time.time()

        if now - self._last_screenshot < self.screenshot_cooldown:
            return

        self._last_screenshot = now

        filename = datetime.now().strftime(
            "alert_%Y%m%d_%H%M%S.jpg"
        )

        path = os.path.join(
            self.screenshot_dir,
            filename
        )

        cv2.imwrite(
            path,
            frame
        )
        
    def annotate_frame(self, frame):
        detections = self._predict(frame)

        current_mask = 0
        current_no_mask = 0

        for i, det in enumerate(detections):

            x, y, w, h = det.box

            if det.label == "Mask":
                color = (0, 255, 0)
                current_mask += 1
            else:
                color = (0, 0, 255)
                current_no_mask += 1

            label_text = (
                f"ID {i+1} | "
                f"{det.label} "
                f"{det.confidence*100:.1f}%"
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            cv2.putText(
                frame,
                label_text,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        if current_no_mask > 0:

            self.alert_manager.trigger()

            self._maybe_save_screenshot(frame)

            for det in detections:
                if det.label == "No Mask":
                    self.save_log(
                        det.label,
                        det.confidence
                    )
        self.total_mask += current_mask
        self.total_no_mask += current_no_mask

        total = current_mask + current_no_mask

        if total > 0:
            compliance = (current_mask / total) * 100
        else:
            compliance = 0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(
            frame,
            f"Faces : {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Mask : {current_mask}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"No Mask : {current_no_mask}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Compliance : {compliance:.1f}%",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            timestamp,
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return frame, detections

    def _predict(self, frame) -> List[Detection]:
        faces = self.face_detector.detect(frame)

        if len(faces) == 0:
            return []

        face_batch = []
        boxes = []

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = preprocess_face(face, self.input_size)
            face_batch.append(face)
            boxes.append((x, y, w, h))

        if not face_batch:
            return []

        preds = self.model.predict(np.array(face_batch), verbose=0)

        detections = []

        for box, pred in zip(boxes, preds):
            pred = np.asarray(pred).reshape(-1)

            if pred.size == 1:
                pred = np.array(
                    [1.0 - float(pred[0]), float(pred[0])],
                    dtype="float32"
                )

            label_idx = int(np.argmax(pred))

            if self.labels and label_idx < len(self.labels):
                label_raw = self.labels[label_idx]
            else:
                label_raw = "with_mask" if label_idx == 0 else "without_mask"

            label = _normalize_label(label_raw)
            confidence = float(pred[label_idx])

            detections.append(
                Detection(
                    box=box,
                    label=label,
                    confidence=confidence
                )
            )

        return detections

def _load_config(config_path, model_path):
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    # Try default config location
    model_dir = os.path.dirname(model_path)
    default_config = os.path.join(
        model_dir,
        "model_config.json"
    )

    if os.path.exists(default_config):
        try:
            with open(default_config, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {}


def _load_labels(labels_path, config):
    labels = None

    if labels_path and os.path.exists(labels_path):
        try:
            with open(labels_path, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                labels = data

            elif isinstance(data, dict):
                labels = list(data.values())

        except Exception:
            pass


    if labels is None:
        labels = config.get(
            "labels",
            [
                "Mask",
                "No Mask"
            ]
        )

    return labels



def _resolve_input_size(config, default_size):
    try:
        size = config.get("input_size")

        if size:
            if isinstance(size, list):
                return tuple(size)

            if isinstance(size, tuple):
                return size

    except Exception:
        pass

    return default_size



def _normalize_label(label):
    label = str(label).lower().strip()

    if label in [
        "mask",
        "with_mask",
        "with mask",
        "masked"
    ]:
        return "Mask"

    if label in [
        "no_mask",
        "without_mask",
        "without mask",
        "no mask",
        "nomask"
    ]:
        return "No Mask"

    return label.title()

        