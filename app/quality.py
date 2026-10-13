from __future__ import annotations

import numpy as np

from .config import QualityConfig
from .types import QualityReport


class FaceQualityGate:
    def __init__(self, config: QualityConfig):
        self.config = config

    @staticmethod
    def _crop(image: np.ndarray, face: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        x, y, w, h = (float(v) for v in face[:4])
        x1, y1 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        x2 = max(0, min(width, int(np.ceil(x + w))))
        y2 = max(0, min(height, int(np.ceil(y + h))))
        return image[y1:y2, x1:x2]

    def evaluate(self, image: np.ndarray, faces: np.ndarray) -> QualityReport:
        count = int(len(faces))
        if count == 0:
            return QualityReport(False, "no_face", "未检测到人脸", 0)
        if count > 1:
            return QualityReport(False, "multiple_faces", "检测到多张人脸，请一次仅保留一人", count)

        face = faces[0]
        if not np.all(np.isfinite(face)) or face[2] <= 0 or face[3] <= 0:
            return QualityReport(False, "invalid_box", "人脸框无效", 1)
        frame_h, frame_w = image.shape[:2]
        face_ratio = min(float(face[2]) / frame_w, float(face[3]) / frame_h)
        crop = self._crop(image, face)
        if crop.size == 0:
            return QualityReport(False, "invalid_box", "人脸框超出图像范围", 1, face_ratio)

        import cv2

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        values = dict(
            face_count=1,
            face_ratio=face_ratio,
            blur_variance=blur,
            brightness=brightness,
        )
        if face_ratio < self.config.min_face_ratio:
            return QualityReport(False, "face_too_small", "人脸过小，请靠近摄像头", **values)
        if blur < self.config.min_blur_variance:
            return QualityReport(False, "too_blurry", "画面模糊，请保持稳定后重试", **values)
        if brightness < self.config.min_brightness:
            return QualityReport(False, "too_dark", "人脸区域过暗，请改善光照", **values)
        if brightness > self.config.max_brightness:
            return QualityReport(False, "too_bright", "人脸区域过亮，请避免强光或逆光", **values)
        return QualityReport(True, "accepted", "质量检查通过", **values)
