import csv
from types import SimpleNamespace

import pytest

from experiments.common import load_manifest, validate_gallery_isolation
from tools.prepare_orl import download


def test_image_split_is_explicit_and_preserves_unknown_session_metadata(tmp_path):
    path = tmp_path / "public.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "split", "is_known", "session_id", "split_unit"])
        writer.writerow(["a.png", "validation", "false", "unavailable", "image"])
        writer.writerow(["b.png", "test", "false", "unavailable", "image"])
    items = load_manifest(path)
    assert {item.split_unit for item in items} == {"image"}
    assert {item.session_id for item in items} == {"unavailable"}
    validate_gallery_isolation(
        items, [], [SimpleNamespace(image_path=tmp_path / "enrolled.png", session_id="unavailable")]
    )
    path.write_text(path.read_text().replace(",image", ",session"))
    with pytest.raises(ValueError, match="会话"):
        load_manifest(path)


def test_public_download_rejects_bad_cached_archive_without_network(tmp_path, monkeypatch):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not the dataset")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("Unexpected network"))
    with pytest.raises(ValueError, match="checksum"):
        download(path)
