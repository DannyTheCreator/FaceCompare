from dataclasses import replace

import numpy as np
import pytest

from app.config import StorageConfig, TrackingConfig, load_config
from app.matcher import FeatureRecord, GallerySnapshot
from app.service import FaceComparisonService
from app.types import QualityReport


@pytest.fixture
def scene_service(tmp_path, monkeypatch):
    import app.service as module

    class Detector:
        def __init__(self, *args):
            self.calls = 0
            self.faces = np.zeros((2, 15), dtype=np.float32)
            self.faces[:, :4] = [[10, 10, 50, 50], [100, 10, 50, 50]]
            self.faces[:, 14] = [0.99, 0.98]

        def detect(self, image, *args):
            self.calls += 1
            return self.faces

    class Recognizer:
        model_hash = "synthetic-test-model"

        def __init__(self, *args):
            pass

        def extract(self, image, face):
            return np.array([1, 0] if face[0] < 50 else [0, 1], dtype=np.float32)

    class Quality:
        def __init__(self, *args):
            pass

        def evaluate(self, image, faces):
            single = len(faces) == 1
            return QualityReport(
                single, "accepted" if single else "multiple_faces", "quality", len(faces)
            )

    monkeypatch.setattr(module, "FaceDetector", Detector)
    monkeypatch.setattr(module, "FaceRecognizer", Recognizer)
    monkeypatch.setattr(module, "FaceQualityGate", Quality)
    config = replace(
        load_config(), storage=StorageConfig(tmp_path / "gallery.db", tmp_path / "samples")
    )
    service = FaceComparisonService(config)
    service.matcher.publish(
        GallerySnapshot.from_records(
            [
                FeatureRecord(1, "A", np.array([1, 0])),
                FeatureRecord(2, "B", np.array([0, 1])),
            ]
        )
    )
    return service


def test_scene_detects_once_and_matches_faces_independently(scene_service):
    result = scene_service.recognize_scene(np.zeros((200, 200, 3), dtype=np.uint8))
    assert scene_service.detector.calls == 1
    assert result.detected_count == 2
    assert [face.person_id for face in result.faces] == [1, 2]
    assert all(len(face.candidates) == 2 for face in result.faces)


def test_scene_limit_is_explicit_and_single_enrollment_remains_strict(scene_service):
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    scene_service.config = replace(scene_service.config, tracking=TrackingConfig(max_faces=1))
    result = scene_service.recognize_scene(image)
    assert result.detected_count == 2 and len(result.faces) == 1
    assert scene_service.recognize(image, write_log=False).status == "multiple_faces"
    sample, report = scene_service.prepare_enrollment_frame(image)
    assert sample is None and report.code == "multiple_faces"


def test_one_face_error_does_not_hide_other_faces(scene_service, monkeypatch):
    extract = scene_service.recognizer.extract

    def sometimes_fail(image, face):
        if face[0] < 50:
            raise ValueError("bad alignment")
        return extract(image, face)

    monkeypatch.setattr(scene_service.recognizer, "extract", sometimes_fail)
    result = scene_service.recognize_scene(np.zeros((200, 200, 3), dtype=np.uint8))
    assert [face.status for face in result.faces] == ["face_error", "matched"]
