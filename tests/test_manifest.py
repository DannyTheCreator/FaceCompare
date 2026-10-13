import csv
from types import SimpleNamespace

import pytest

from experiments.common import load_manifest, validate_gallery_isolation


def manifest(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "image_path", "split", "is_known", "expected_person_id", "session_id"]
        )
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    "rows",
    [
        [["a", "a.jpg", "test", "true", "", "s1"]],
        [["a", "a.jpg", "test", "false", "1", "s1"]],
        [["a", "", "test", "false", "", "s1"]],
        [["a", "a.jpg", "test", "false", "", ""]],
        [
            ["a", "a.jpg", "validation", "false", "", "s1"],
            ["a", "b.jpg", "test", "false", "", "s2"],
        ],
        [
            ["a", "a.jpg", "validation", "false", "", "same"],
            ["b", "b.jpg", "test", "false", "", "same"],
        ],
        [
            ["a", "a.jpg", "validation", "false", "", "s1"],
            ["b", "a.jpg", "test", "false", "", "s2"],
        ],
    ],
)
def test_manifest_rejects_bad_labels_and_split_leakage(tmp_path, rows):
    with pytest.raises(ValueError):
        load_manifest(manifest(tmp_path, rows), split="test")


def test_registration_overlap_is_rejected(tmp_path):
    rows = [["a", "a.jpg", "test", "true", "1", "registered-session"]]
    items = load_manifest(manifest(tmp_path, rows))
    registered = [
        SimpleNamespace(image_path=tmp_path / "other.jpg", session_id="registered-session")
    ]
    with pytest.raises(ValueError, match="重叠"):
        validate_gallery_isolation(items, [1], registered)


def test_known_label_must_have_compatible_template(tmp_path):
    items = load_manifest(manifest(tmp_path, [["a", "a.jpg", "test", "true", "1", "s"]]))
    with pytest.raises(ValueError, match="兼容标准库"):
        validate_gallery_isolation(items, [], [])
