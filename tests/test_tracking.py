from dataclasses import replace

import numpy as np
import pytest

from app.config import TrackingConfig
from app.tracking import SceneTracker, box_iou
from app.types import RecognitionResult, SceneResult


def face(x=10, person=1, status="matched"):
    row = np.zeros(15, dtype=np.float32)
    row[:4] = [x, 10, 50, 50]
    return RecognitionResult(
        1,
        0,
        0,
        status,
        status,
        0.42,
        person_id=person if status == "matched" else None,
        person_name=f"Person {person}" if status == "matched" else None,
        candidate_person_id=person,
        score=0.8 if status == "matched" else 0.2,
        face=row,
    )


def scene(timestamp, faces, stream=1, version=1):
    return SceneResult(
        int(timestamp * 10),
        timestamp,
        timestamp,
        tuple(faces),
        len(faces),
        0.42,
        stream,
        version,
        {"algorithm_total": 1},
    )


def test_confirmation_requires_distinct_consecutive_observations():
    tracker = SceneTracker(TrackingConfig(confirmation_frames=3))
    first = tracker.update(scene(1, [face()]))
    assert first.faces[0].status == "pending"
    assert first.faces[0].person_name is None
    assert tracker.update(scene(1, [face()])).faces[0].confirmation_hits == 1
    second = tracker.update(scene(1.1, [face(x=12)]))
    third = tracker.update(scene(1.2, [face(x=14)]))
    assert second.faces[0].track_id == third.faces[0].track_id == 1
    assert third.faces[0].confirmed and third.faces[0].person_id == 1


def test_identity_change_and_quality_failure_clear_confirmation_immediately():
    tracker = SceneTracker(TrackingConfig(confirmation_frames=2))
    tracker.update(scene(1, [face()]))
    assert tracker.update(scene(1.1, [face()])).faces[0].confirmed
    changed = tracker.update(scene(1.2, [face(person=2)]))
    assert changed.faces[0].status == "pending" and changed.faces[0].confirmation_hits == 1
    unknown = tracker.update(scene(1.3, [face(person=2, status="unknown")]))
    assert unknown.faces[0].person_name is None and unknown.faces[0].confirmation_hits == 0
    assert tracker.update(scene(1.4, [face(status="too_blurry")])).faces[0].status == "too_blurry"


def test_absent_face_breaks_evidence_and_old_track_expires():
    tracker = SceneTracker(TrackingConfig(confirmation_frames=2, max_gap_ms=300))
    tracker.update(scene(1, [face()]))
    tracker.update(scene(1.1, []))
    assert tracker.update(scene(1.2, [face()])).faces[0].confirmation_hits == 1
    expired = tracker.update(scene(2, [face()]))
    assert expired.faces[0].track_id == 2 and not expired.faces[0].confirmed


@pytest.mark.parametrize(
    "change", [{"stream_id": 2}, {"gallery_version": 2}, {"mode_generation": 2}]
)
def test_context_change_discards_old_identity_evidence(change):
    tracker = SceneTracker(TrackingConfig(confirmation_frames=2))
    tracker.update(scene(1, [face()]))
    assert tracker.update(scene(1.1, [face()])).faces[0].confirmed
    result = tracker.update(replace(scene(1.2, [face()]), **change))
    assert not result.faces[0].confirmed and result.faces[0].confirmation_hits == 1


def test_assignment_remains_one_to_one_when_detection_order_changes():
    tracker = SceneTracker(TrackingConfig())
    tracker.update(scene(1, [face(10, 1), face(200, 2)]))
    result = tracker.update(scene(1.1, [face(202, 2), face(12, 1)]))
    assert [item.track_id for item in result.faces] == [2, 1]


def test_iou_and_out_of_order_frames():
    assert box_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1
    assert box_iou(np.array([0, 0, -10, 10]), np.array([0, 0, 10, 10])) == 0
    tracker = SceneTracker(TrackingConfig())
    tracker.update(scene(2, [face()]))
    with pytest.raises(ValueError, match="递增"):
        tracker.update(scene(1, [face()]))
