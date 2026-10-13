from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraConfig:
    default_index: int
    probe_indices: tuple[int, ...]
    width: int
    height: int
    preview_fps: int
    recognition_interval_ms: int
    result_ttl_ms: int = 2500


@dataclass(frozen=True)
class ModelConfig:
    detector: Path
    recognizer: Path


@dataclass(frozen=True)
class DetectionConfig:
    score_threshold: float
    nms_threshold: float
    top_k: int
    detect_scale: float


@dataclass(frozen=True)
class QualityConfig:
    min_face_ratio: float
    min_blur_variance: float
    min_brightness: float
    max_brightness: float


@dataclass(frozen=True)
class MatchingConfig:
    cosine_threshold: float
    top_k: int = 3


@dataclass(frozen=True)
class TrackingConfig:
    max_faces: int = 5
    iou_threshold: float = 0.3
    max_gap_ms: int = 2000
    confirmation_frames: int = 3


@dataclass(frozen=True)
class EnrollmentConfig:
    minimum_samples: int
    target_samples: int


@dataclass(frozen=True)
class StorageConfig:
    database: Path
    samples_dir: Path


@dataclass(frozen=True)
class AppConfig:
    root: Path
    camera: CameraConfig
    models: ModelConfig
    detection: DetectionConfig
    quality: QualityConfig
    matching: MatchingConfig
    enrollment: EnrollmentConfig
    storage: StorageConfig
    preprocess_version: str
    tracking: TrackingConfig = TrackingConfig()


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"配置缺少字段: {key}")
    return data[key]


def _resolve(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("配置中的文件路径必须是非空字符串")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _integer(data: dict[str, Any], key: str) -> int:
    value = _required(data, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} 必须是整数")
    return value


def _number(data: dict[str, Any], key: str) -> float:
    value = _required(data, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} 必须是数值")
    if not math.isfinite(value):
        raise ValueError(f"{key} 必须是有限数值")
    return float(value)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the JSON configuration.

    Relative file paths are interpreted from the directory containing config.json,
    so the application does not depend on the process working directory.
    """

    config_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else Path(__file__).resolve().parents[1] / "config.json"
    )
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("配置根节点必须是对象")
    for key in ("camera", "models", "detection", "quality", "matching", "enrollment", "storage"):
        if not isinstance(_required(raw, key), dict):
            raise ValueError(f"配置 {key} 必须是对象")
    root = config_path.parent

    camera = _required(raw, "camera")
    models = _required(raw, "models")
    detection = _required(raw, "detection")
    quality = _required(raw, "quality")
    matching = _required(raw, "matching")
    enrollment = _required(raw, "enrollment")
    storage = _required(raw, "storage")
    tracking = raw.get("tracking", {})
    if not isinstance(tracking, dict):
        raise ValueError("tracking 必须是对象")
    tracking = {
        "max_faces": 5,
        "iou_threshold": 0.3,
        "max_gap_ms": 2000,
        "confirmation_frames": 3,
        **tracking,
    }
    indices = _required(camera, "probe_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("camera.probe_indices 必须是非空且不重复的非负整数列表")
    preprocess = _required(raw, "preprocess_version")
    if not isinstance(preprocess, str) or not preprocess.strip():
        raise ValueError("preprocess_version 必须是非空字符串")

    cfg = AppConfig(
        root=root,
        camera=CameraConfig(
            default_index=_integer(camera, "default_index"),
            probe_indices=tuple(indices),
            width=_integer(camera, "width"),
            height=_integer(camera, "height"),
            preview_fps=_integer(camera, "preview_fps"),
            recognition_interval_ms=_integer(camera, "recognition_interval_ms"),
            result_ttl_ms=_integer(
                {"result_ttl_ms": camera.get("result_ttl_ms", 2500)}, "result_ttl_ms"
            ),
        ),
        models=ModelConfig(
            detector=_resolve(root, _required(models, "detector")),
            recognizer=_resolve(root, _required(models, "recognizer")),
        ),
        detection=DetectionConfig(
            score_threshold=_number(detection, "score_threshold"),
            nms_threshold=_number(detection, "nms_threshold"),
            top_k=_integer(detection, "top_k"),
            detect_scale=_number(detection, "detect_scale"),
        ),
        quality=QualityConfig(
            min_face_ratio=_number(quality, "min_face_ratio"),
            min_blur_variance=_number(quality, "min_blur_variance"),
            min_brightness=_number(quality, "min_brightness"),
            max_brightness=_number(quality, "max_brightness"),
        ),
        matching=MatchingConfig(
            cosine_threshold=_number(matching, "cosine_threshold"),
            top_k=_integer({"top_k": matching.get("top_k", 3)}, "top_k"),
        ),
        enrollment=EnrollmentConfig(
            minimum_samples=_integer(enrollment, "minimum_samples"),
            target_samples=_integer(enrollment, "target_samples"),
        ),
        storage=StorageConfig(
            database=_resolve(root, _required(storage, "database")),
            samples_dir=_resolve(root, _required(storage, "samples_dir")),
        ),
        preprocess_version=preprocess.strip(),
        tracking=TrackingConfig(
            max_faces=_integer(tracking, "max_faces"),
            iou_threshold=_number(tracking, "iou_threshold"),
            max_gap_ms=_integer(tracking, "max_gap_ms"),
            confirmation_frames=_integer(tracking, "confirmation_frames"),
        ),
    )
    if not 0 < cfg.detection.detect_scale <= 1:
        raise ValueError("detection.detect_scale 必须在 (0, 1] 范围内")
    if not -1 <= cfg.matching.cosine_threshold <= 1:
        raise ValueError("matching.cosine_threshold 必须在 [-1, 1] 范围内")
    if cfg.enrollment.minimum_samples < 1:
        raise ValueError("enrollment.minimum_samples 必须至少为 1")
    if cfg.enrollment.target_samples < cfg.enrollment.minimum_samples:
        raise ValueError("enrollment.target_samples 不能小于 minimum_samples")
    if cfg.camera.default_index not in cfg.camera.probe_indices:
        raise ValueError("camera.default_index 必须包含在 probe_indices 中")
    for key in ("width", "height", "preview_fps", "recognition_interval_ms", "result_ttl_ms"):
        if getattr(cfg.camera, key) <= 0:
            raise ValueError(f"camera.{key} 必须大于 0")
    if cfg.detection.top_k < 1:
        raise ValueError("detection.top_k 必须大于 0")
    for key in ("score_threshold", "nms_threshold"):
        if not 0 <= getattr(cfg.detection, key) <= 1:
            raise ValueError(f"detection.{key} 必须在 [0, 1] 范围内")
    if not 0 < cfg.quality.min_face_ratio <= 1 or cfg.quality.min_blur_variance < 0:
        raise ValueError("人脸尺寸比例必须在 (0, 1] 内，清晰度下限不能为负数")
    if not 0 <= cfg.quality.min_brightness < cfg.quality.max_brightness <= 255:
        raise ValueError("亮度范围必须满足 0 <= min_brightness < max_brightness <= 255")
    if cfg.models.detector == cfg.models.recognizer:
        raise ValueError("检测模型和识别模型不能指向同一文件")
    if cfg.storage.database == cfg.storage.samples_dir:
        raise ValueError("数据库和样本目录不能使用同一路径")
    if not 1 <= cfg.matching.top_k <= 20:
        raise ValueError("matching.top_k 必须在 [1, 20] 内")
    if not 1 <= cfg.tracking.max_faces <= 20 or not 1 <= cfg.tracking.confirmation_frames <= 30:
        raise ValueError("tracking.max_faces 和 confirmation_frames 超出允许范围")
    if not 0 < cfg.tracking.iou_threshold <= 1 or cfg.tracking.max_gap_ms <= 0:
        raise ValueError("tracking.iou_threshold 必须在 (0, 1] 内，max_gap_ms 必须为正数")
    return cfg
