import numpy as np
import pytest

from app.matcher import FaceMatcher, FeatureRecord, GallerySnapshot


def test_gallery_aggregates_normalized_person_templates():
    snapshot = GallerySnapshot.from_records(
        [
            FeatureRecord(2, "乙", np.array([0.0, 1.0], dtype=np.float32)),
            FeatureRecord(1, "甲", np.array([1.0, 0.0], dtype=np.float32)),
            FeatureRecord(1, "甲", np.array([0.8, 0.2], dtype=np.float32)),
        ]
    )
    assert snapshot.person_ids == (1, 2)
    assert snapshot.sample_counts == (2, 1)
    np.testing.assert_allclose(np.linalg.norm(snapshot.templates, axis=1), 1.0)


def test_matrix_and_loop_match_are_equivalent():
    snapshot = GallerySnapshot.from_records(
        [
            FeatureRecord(1, "甲", np.array([1.0, 0.0], dtype=np.float32)),
            FeatureRecord(2, "乙", np.array([0.0, 1.0], dtype=np.float32)),
        ]
    )
    matcher = FaceMatcher(0.5)
    matcher.publish(snapshot)
    query = np.array([0.9, 0.1], dtype=np.float32)
    matrix = matcher.top_match(query, "matrix")
    loop = matcher.top_match(query, "loop")
    assert matrix is not None and loop is not None
    assert matrix.person_id == loop.person_id == 1
    assert matrix.score == pytest.approx(loop.score, abs=1e-7)


def test_empty_gallery_returns_none():
    assert FaceMatcher(0.4).top_match(np.array([1.0, 0.0])) is None


def test_rank_is_identity_level_bounded_and_deterministic_on_ties():
    matcher = FaceMatcher(0.4)
    matcher.publish(
        GallerySnapshot.from_records(
            [
                FeatureRecord(2, "B", np.array([1, 0])),
                FeatureRecord(1, "A", np.array([1, 0])),
                FeatureRecord(1, "A", np.array([1, 0])),
            ]
        )
    )
    ranked = matcher.rank(np.array([1, 0]), k=20)
    assert [candidate.person_id for candidate in ranked] == [1, 2]
    assert len(matcher.rank(np.array([1, 0]), k=1)) == 1


@pytest.mark.parametrize("tied", [False, True])
def test_partial_topk_matches_full_stable_sort(tied):
    rng = np.random.default_rng(31)
    matcher = FaceMatcher(0.42)
    matcher.publish(
        GallerySnapshot.from_records(
            FeatureRecord(i, str(i), np.array([1, 0, 0, 0]) if tied else rng.normal(size=4))
            for i in range(300)
        )
    )
    from app.recognizer import l2_normalize

    query = l2_normalize(rng.normal(size=4))
    scores = matcher.snapshot.templates @ l2_normalize(query)
    for k in (1, 3, 20, 300, 400):
        expected = np.argsort(-scores, kind="stable")[:k].tolist()
        actual = matcher.rank(query, k)
        assert [c.person_id for c in actual] == expected
