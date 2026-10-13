from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .matcher import Candidate


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    captured_monotonic: float
    image: np.ndarray
    stream_id: int = 0
    mode: str = "single"
    mode_generation: int = 0
    submitted_monotonic: float | None = None


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    code: str
    message: str
    face_count: int
    face_ratio: float | None = None
    blur_variance: float | None = None
    brightness: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "code": self.code,
            "message": self.message,
            "face_count": self.face_count,
            "face_ratio": self.face_ratio,
            "blur_variance": self.blur_variance,
            "brightness": self.brightness,
        }


@dataclass(frozen=True)
class PreparedSample:
    image: np.ndarray
    feature: np.ndarray
    face: np.ndarray
    quality: QualityReport


@dataclass(frozen=True)
class RecognitionResult:
    frame_id: int
    captured_monotonic: float
    completed_monotonic: float
    status: str
    message: str
    threshold: float
    person_id: int | None = None
    person_name: str | None = None
    candidate_person_id: int | None = None
    candidate_name: str | None = None
    score: float | None = None
    face: np.ndarray | None = None
    quality: QualityReport | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    stream_id: int = 0
    gallery_version: int = 0
    candidates: tuple[Candidate, ...] = ()
    track_id: int | None = None
    confirmation_hits: int = 0
    confirmed: bool = False
    mode_generation: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def end_to_end_ms(self) -> float:
        return max(0.0, (self.completed_monotonic - self.captured_monotonic) * 1000.0)


@dataclass(frozen=True)
class SceneResult:
    frame_id: int
    captured_monotonic: float
    completed_monotonic: float
    faces: tuple[RecognitionResult, ...]
    detected_count: int
    threshold: float
    stream_id: int
    gallery_version: int
    timings_ms: dict[str, float]
    mode_generation: int = 0

    @property
    def end_to_end_ms(self) -> float:
        return max(0.0, (self.completed_monotonic - self.captured_monotonic) * 1000.0)
