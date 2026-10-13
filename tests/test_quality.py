import numpy as np
import pytest

from app.config import load_config
from app.quality import FaceQualityGate
from app.recognizer import l2_normalize


def test_negative_outside_box_does_not_crop_unrelated_pixels():
    gate = FaceQualityGate(load_config().quality)
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    face = np.zeros((1, 15), dtype=np.float32)
    face[0, :4] = [-40, 10, 10, 60]
    assert gate.evaluate(image, face).code == "invalid_box"


@pytest.mark.parametrize("feature", [[], [0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_normalization_rejects_invalid_features(feature):
    with pytest.raises(ValueError):
        l2_normalize(np.asarray(feature))


def test_large_finite_feature_keeps_unit_norm():
    np.testing.assert_allclose(np.linalg.norm(l2_normalize(np.array([1e30, 1e30]))), 1)
