import pytest

from experiments.common import RawPrediction, calculate_metrics, choose_threshold


def prediction(sample_id, known, expected, candidate, score, status="matched"):
    return RawPrediction(
        sample_id,
        f"{sample_id}.jpg",
        "validation",
        known,
        expected,
        "normal",
        "session",
        status,
        candidate,
        str(candidate) if candidate else None,
        score,
        1.0,
        1.0,
        1.0,
        1.0,
        4.0,
    )


def test_metrics_keep_failures_out_of_effective_unknown_denominator():
    rows = [
        prediction("k1", True, 1, 1, 0.8),
        prediction("k2", True, 1, 1, 0.2),
        prediction("u1", False, None, 1, 0.6),
        prediction("u2", False, None, 1, 0.1),
        prediction("u3", False, None, None, None, "no_face"),
    ]
    metrics = calculate_metrics(rows, 0.5)
    assert metrics["known_end_to_end_recognition_rate"] == 0.5
    assert metrics["unknown_false_accept_rate_all_attempts"] == 1 / 3
    assert metrics["unknown_false_accept_rate_valid_queries"] == 0.5
    assert metrics["unknown_correct_reject_rate_all_attempts"] == 1 / 3


def test_threshold_selection_prioritizes_far_constraint_then_known_rate():
    rows = [
        prediction("k1", True, 1, 1, 0.85),
        prediction("k2", True, 1, 1, 0.70),
        prediction("u1", False, None, 1, 0.65),
        prediction("u2", False, None, 1, 0.30),
    ]
    threshold, metrics = choose_threshold(rows, 0.0)
    assert threshold > 0.65
    assert metrics["unknown_false_accept_rate_valid_queries"] == 0.0
    assert metrics["known_end_to_end_recognition_rate"] == 1.0


def test_nonfinite_scores_do_not_count_as_valid_rejections():
    metrics = calculate_metrics([prediction("u", False, None, 1, float("nan"))], 0.5)
    assert metrics["valid_unknown_queries"] == 0
    assert metrics["counts"] == {"processing_failure": 1}
    assert metrics["unknown_false_accept_rate_valid_queries"] is None


def test_calibration_requires_usable_known_and_unknown_queries():
    with pytest.raises(ValueError, match="已知人员"):
        choose_threshold([prediction("u", False, None, 1, 0.3)], 0.05)
