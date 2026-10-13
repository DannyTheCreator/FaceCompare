from dataclasses import replace

import pytest

from experiments.common import write_predictions
from experiments.report import build_report, load_predictions, render_report
from tests.test_metrics import prediction


def test_report_preserves_failures_and_scene_denominators(tmp_path):
    rows = [
        prediction("k1", True, 1, 1, 0.8),
        prediction("u1", False, None, 1, 0.3),
        replace(prediction("u2", False, None, None, None, "no_face"), scene="dark"),
    ]
    path = tmp_path / "predictions.csv"
    write_predictions(path, rows, 0.42)
    loaded = load_predictions(path)
    report = build_report(loaded, 0.42)
    assert report["metrics"]["attempts"] == 3
    assert report["metrics"]["unknown_correct_reject_rate_all_attempts"] == 0.5
    assert report["scenes"]["dark"]["valid_unknown_queries"] == 0
    render_report(loaded, report, tmp_path / "report")
    assert (tmp_path / "report/analysis.png").stat().st_size > 1000
    assert "candidate_name" not in (tmp_path / "report/analysis.json").read_text()


def test_known_only_report_does_not_invent_unknown_far():
    report = build_report([prediction("k", True, 1, 1, 0.8)], 0.42)
    assert all(point["unknown_far_valid"] is None for point in report["curve"])


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        build_report([prediction("k", True, 1, 1, 0.8)], float("nan"))
