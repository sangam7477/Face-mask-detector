from typing import Tuple

import cv2


def preprocess_face(face_bgr, target_size: Tuple[int, int] = (224, 224)):
    """Convert BGR face to normalized RGB tensor."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, target_size)
    face = face_rgb.astype("float32") / 255.0
    return face
