from dataclasses import replace

import numpy as np
import pytest

from app.config import StorageConfig, load_config
from app.model_manager import verify_models
from app.service import FaceComparisonService


@pytest.mark.integration
def test_real_models_handle_blank_frame_without_touching_local_gallery(tmp_path):
    config = load_config()
    if not config.models.detector.exists() or not config.models.recognizer.exists():
        pytest.skip("Download official ONNX models to run integration tests")
    verify_models(config)
    config = replace(config, storage=StorageConfig(tmp_path / "gallery.db", tmp_path / "samples"))
    service = FaceComparisonService(config)
    result = service.recognize(np.zeros((480, 640, 3), dtype=np.uint8), write_log=False)
    assert result.status == "no_face"
    assert result.score is None
    assert result.end_to_end_ms >= result.timings_ms["algorithm_total"]
    assert service.repository.list_persons() == []
    scene = service.recognize_scene(np.zeros((480, 640, 3), dtype=np.uint8))
    assert scene.detected_count == 0 and scene.faces == ()
    assert scene.end_to_end_ms >= scene.timings_ms["algorithm_total"]
