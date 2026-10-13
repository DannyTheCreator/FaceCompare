from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np

from .config import DetectionConfig


class FaceDetector:
    """YuNet wrapper that maps boxes and all five landmarks to source pixels."""

    def __init__(self, model_path: Path, config: DetectionConfig):
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet 模型不存在: {model_path}")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("缺少 opencv-contrib-python，请先安装 requirements.txt") from exc
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("当前 OpenCV 不包含 FaceDetectorYN，请安装 contrib 版本")
        self._cv2 = cv2
        self._config = config
        self._model = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            config.score_threshold,
            config.nms_threshold,
            config.top_k,
        )
        self._lock = Lock()

    def detect(self, image: np.ndarray, scale: float | None = None) -> np.ndarray:
        """Return YuNet rows [box, five landmarks, confidence] in source pixels."""

        if (
            image is None
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or image.size == 0
        ):
            raise ValueError("检测输入必须是非空 BGR 图像")
        applied_scale = self._config.detect_scale if scale is None else float(scale)
        if not 0 < applied_scale <= 1:
            raise ValueError("检测缩放比例必须在 (0, 1] 范围内")
        source_h, source_w = image.shape[:2]
        if applied_scale < 1:
            detect_w = max(32, int(round(source_w * applied_scale)))
            detect_h = max(32, int(round(source_h * applied_scale)))
            detect_image = self._cv2.resize(
                image, (detect_w, detect_h), interpolation=self._cv2.INTER_AREA
            )
            scale_x, scale_y = source_w / detect_w, source_h / detect_h
        else:
            detect_image = image
            detect_h, detect_w = source_h, source_w
            scale_x = scale_y = 1.0

        with self._lock:
            self._model.setInputSize((detect_w, detect_h))
            _, faces = self._model.detect(detect_image)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        mapped = np.asarray(faces, dtype=np.float32).copy()
        mapped[:, [0, 2, 4, 6, 8, 10, 12]] *= scale_x
        mapped[:, [1, 3, 5, 7, 9, 11, 13]] *= scale_y
        return mapped
