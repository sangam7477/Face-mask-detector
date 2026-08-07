import os
from typing import Optional, Tuple

import cv2


class FaceDetector:
    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Haar cascade not found at: {cascade_path}. "
                "Please provide a valid --face-cascade path."
            )
        self.cascade_path = cascade_path
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(
        self,
        frame,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (60, 60),
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        return faces
