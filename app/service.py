from __future__ import annotations

import time
import uuid
from dataclasses import replace
from threading import RLock
from typing import Sequence

import numpy as np

from .config import AppConfig
from .detector import FaceDetector
from .matcher import FaceMatcher, GallerySnapshot
from .quality import FaceQualityGate
from .recognizer import FaceRecognizer
from .repository import GalleryRepository
from .types import PreparedSample, QualityReport, RecognitionResult, SceneResult


class FaceComparisonService:
    """Single entry point shared by UI, evaluation, and benchmarks."""

    def __init__(self, config: AppConfig, *, read_only: bool = False):
        self.config = config
        self.repository = GalleryRepository(
            config.storage.database, config.storage.samples_dir, read_only=read_only
        )
        self.detector = FaceDetector(config.models.detector, config.detection)
        self.recognizer = FaceRecognizer(config.models.recognizer)
        self.quality_gate = FaceQualityGate(config.quality)
        self.matcher = FaceMatcher(config.matching.cosine_threshold)
        self._operation_lock = RLock()
        self.refresh_gallery()

    @property
    def model_hash(self) -> str:
        return self.recognizer.model_hash

    def refresh_gallery(self) -> GallerySnapshot:
        with self._operation_lock:
            records, version = self.repository.load_feature_snapshot(
                self.model_hash, self.config.preprocess_version
            )
            snapshot = GallerySnapshot.from_records(records)
            self.matcher.publish(snapshot)
            self._gallery_version = version
            return snapshot

    def _base_result(
        self,
        frame_id: int,
        captured_monotonic: float,
        status: str,
        message: str,
        timings: dict[str, float],
        **kwargs,
    ) -> RecognitionResult:
        return RecognitionResult(
            frame_id=frame_id,
            captured_monotonic=captured_monotonic,
            completed_monotonic=time.perf_counter(),
            status=status,
            message=message,
            threshold=self.matcher.threshold,
            timings_ms=timings,
            gallery_version=self._gallery_version,
            **kwargs,
        )

    def recognize(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        captured_monotonic: float | None = None,
        detect_scale: float | None = None,
        match_mode: str = "matrix",
        write_log: bool = True,
        stream_id: int = 0,
        _faces: np.ndarray | None = None,
    ) -> RecognitionResult:
        captured = time.perf_counter() if captured_monotonic is None else captured_monotonic
        timings: dict[str, float] = {}
        waiting = time.perf_counter()
        with self._operation_lock:
            timings["lock_wait"] = (time.perf_counter() - waiting) * 1000
            started = time.perf_counter()
            stage = time.perf_counter()
            faces = self.detector.detect(image, detect_scale) if _faces is None else _faces
            timings["detect"] = (time.perf_counter() - stage) * 1000

            stage = time.perf_counter()
            quality = self.quality_gate.evaluate(image, faces)
            timings["quality"] = (time.perf_counter() - stage) * 1000
            face = faces[0].copy() if len(faces) == 1 else None
            if not quality.accepted:
                result = self._base_result(
                    frame_id,
                    captured,
                    quality.code,
                    quality.message,
                    timings,
                    face=face,
                    quality=quality,
                    stream_id=stream_id,
                )
            else:
                stage = time.perf_counter()
                feature = self.recognizer.extract(image, faces[0])
                timings["align_and_feature"] = (time.perf_counter() - stage) * 1000

                stage = time.perf_counter()
                candidates = self.matcher.rank(
                    feature, k=self.config.matching.top_k, mode=match_mode
                )
                candidate = candidates[0] if candidates else None
                timings["match"] = (time.perf_counter() - stage) * 1000
                if candidate is None:
                    result = self._base_result(
                        frame_id,
                        captured,
                        "empty_gallery",
                        "标准库为空，请先录入人员",
                        timings,
                        face=face,
                        quality=quality,
                        stream_id=stream_id,
                    )
                elif candidate.score >= self.matcher.threshold:
                    result = self._base_result(
                        frame_id,
                        captured,
                        "matched",
                        f"匹配成功：{candidate.person_name}",
                        timings,
                        person_id=candidate.person_id,
                        person_name=candidate.person_name,
                        candidate_person_id=candidate.person_id,
                        candidate_name=candidate.person_name,
                        score=candidate.score,
                        candidates=candidates,
                        face=face,
                        quality=quality,
                        stream_id=stream_id,
                    )
                else:
                    result = self._base_result(
                        frame_id,
                        captured,
                        "unknown",
                        "未知人员",
                        timings,
                        candidate_person_id=candidate.person_id,
                        candidate_name=candidate.person_name,
                        score=candidate.score,
                        candidates=candidates,
                        face=face,
                        quality=quality,
                        stream_id=stream_id,
                    )
            timings["algorithm_total"] = (time.perf_counter() - started) * 1000
            warnings = ()
            if write_log:
                stage = time.perf_counter()
                try:
                    self.repository.log_recognition(result)
                except Exception as exc:
                    warnings = (f"识别已完成，但日志保存失败: {exc}",)
                timings["log_write"] = (time.perf_counter() - stage) * 1000
            return replace(result, completed_monotonic=time.perf_counter(), warnings=warnings)

    def recognize_scene(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        captured_monotonic: float | None = None,
        stream_id: int = 0,
    ) -> SceneResult:
        """Detect once and evaluate each face independently; enrollment stays single-face."""
        captured = time.perf_counter() if captured_monotonic is None else captured_monotonic
        with self._operation_lock:
            started = time.perf_counter()
            faces = self.detector.detect(image)
            detection_ms = (time.perf_counter() - started) * 1000
            # Bound CPU work and explicitly report total versus processed detections.
            selected = sorted(faces, key=lambda row: -float(row[14]))[
                : self.config.tracking.max_faces
            ]
            results = []
            for face in selected:
                try:
                    results.append(
                        self.recognize(
                            image,
                            frame_id,
                            captured,
                            stream_id=stream_id,
                            write_log=False,
                            _faces=np.asarray([face], dtype=np.float32),
                        )
                    )
                except Exception as exc:
                    results.append(
                        self._base_result(
                            frame_id,
                            captured,
                            "face_error",
                            f"该人脸处理失败: {exc}",
                            {},
                            face=face.copy(),
                            stream_id=stream_id,
                        )
                    )
            timings = {
                "detect": detection_ms,
                "algorithm_total": (time.perf_counter() - started) * 1000,
            }
            return SceneResult(
                frame_id,
                captured,
                time.perf_counter(),
                tuple(results),
                len(faces),
                self.matcher.threshold,
                stream_id,
                self._gallery_version,
                timings,
            )

    def prepare_enrollment_frame(
        self, image: np.ndarray, detect_scale: float | None = None
    ) -> tuple[PreparedSample | None, QualityReport]:
        with self._operation_lock:
            faces = self.detector.detect(image, detect_scale)
            quality = self.quality_gate.evaluate(image, faces)
            if not quality.accepted:
                return None, quality
            feature = self.recognizer.extract(image, faces[0])
            return (
                PreparedSample(image.copy(), feature, faces[0].copy(), quality),
                quality,
            )

    def commit_enrollment(
        self,
        samples: Sequence[PreparedSample],
        *,
        name: str | None = None,
        external_id: str | None = None,
        existing_person_id: int | None = None,
        session_id: str | None = None,
    ) -> int:
        with self._operation_lock:
            created_person = False
            if existing_person_id is None:
                if len(samples) < self.config.enrollment.minimum_samples:
                    raise ValueError(
                        f"新人员至少需要 {self.config.enrollment.minimum_samples} 张合格样本"
                    )
                if not name:
                    raise ValueError("新增人员时姓名不能为空")
                person_id = self.repository.create_person(name, external_id)
                created_person = True
            else:
                person_id = int(existing_person_id)
            try:
                self.repository.add_samples(
                    person_id,
                    samples,
                    session_id or uuid.uuid4().hex,
                    self.model_hash,
                    self.config.preprocess_version,
                )
            except Exception:
                if created_person:
                    self.repository.delete_person(person_id)
                raise
            self.refresh_gallery()
            return person_id

    def delete_sample(self, sample_id: str) -> None:
        with self._operation_lock:
            self.repository.delete_sample(sample_id)
            self.refresh_gallery()

    def delete_person(self, person_id: int) -> None:
        with self._operation_lock:
            self.repository.delete_person(person_id)
            self.refresh_gallery()
