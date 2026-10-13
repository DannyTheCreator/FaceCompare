from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalize(feature: np.ndarray) -> np.ndarray:
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("模型输出包含非有限值或为空")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("模型输出为零向量")
    return vector / norm


class FaceRecognizer:
    """SFace alignment and feature extraction wrapper."""

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"SFace 模型不存在: {model_path}")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("缺少 opencv-contrib-python，请先安装 requirements.txt") from exc
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("当前 OpenCV 不包含 FaceRecognizerSF，请安装 contrib 版本")
        self.model_hash = sha256_file(model_path)
        self._model = cv2.FaceRecognizerSF.create(str(model_path), "")
        self._lock = Lock()

    def extract(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        row = np.asarray(face, dtype=np.float32).reshape(1, -1)
        with self._lock:
            aligned = self._model.alignCrop(image, row)
            feature = self._model.feature(aligned)
        return l2_normalize(feature)
