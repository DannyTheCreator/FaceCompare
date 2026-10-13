import json
from pathlib import Path

import pytest

from app.config import load_config


def test_default_config_resolves_paths_from_project_root():
    config = load_config()
    assert config.root == Path(__file__).resolve().parents[1]
    assert config.models.detector.is_absolute()
    assert config.storage.database.is_absolute()
    assert 0 < config.detection.detect_scale <= 1


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("camera", "width", 0),
        ("camera", "preview_fps", True),
        ("camera", "recognition_interval_ms", -1),
        ("camera", "result_ttl_ms", 0),
        ("camera", "default_index", 100),
        ("camera", "probe_indices", []),
        ("camera", "probe_indices", [0, 0]),
        ("camera", "probe_indices", [0, "1"]),
        ("detection", "top_k", 0),
        ("detection", "detect_scale", float("nan")),
        ("detection", "score_threshold", 1.1),
        ("quality", "min_blur_variance", -1),
        ("quality", "min_brightness", 250),
        ("matching", "cosine_threshold", float("inf")),
        ("enrollment", "target_samples", 1),
        ("models", "detector", ""),
    ],
)
def test_invalid_parameters_fail_before_startup(tmp_path, section, key, value):
    source = Path(__file__).resolve().parents[1] / "config.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw[section][key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_old_config_receives_result_ttl_default(tmp_path):
    raw = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    raw["camera"].pop("result_ttl_ms", None)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    assert load_config(path).camera.result_ttl_ms == 2500


@pytest.mark.parametrize(
    "key,value",
    [("confirmation_frames", 0), ("max_faces", 21), ("iou_threshold", 0), ("max_gap_ms", -1)],
)
def test_invalid_tracking_parameters_are_rejected(tmp_path, key, value):
    raw = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    raw["tracking"][key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(path)
