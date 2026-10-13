from pathlib import Path

import numpy as np
import pytest

from app.repository import GalleryRepository
from experiments.common import load_manifest
from tests.test_manifest import manifest
from tests.test_scene_service import scene_service  # noqa: F401


def test_delete_permission_failure_commits_and_records_cleanup(
    tmp_path, prepared_sample, monkeypatch
):
    repo = GalleryRepository(tmp_path / "gallery.db", tmp_path / "samples")
    person = repo.create_person("Synthetic")
    sample = repo.add_samples(person, [prepared_sample], "s", "m", "v")[0]
    image = repo.list_samples()[0].image_path
    original = Path.unlink

    def locked(path, *args, **kwargs):
        if path == image:
            raise PermissionError("locked image")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    repo.delete_sample(sample)
    assert repo.list_samples() == []
    assert repo.cleanup_pending() == 1
    assert image.exists()
    monkeypatch.setattr(Path, "unlink", original)
    assert repo.cleanup_pending() == 0
    assert not image.exists()


def test_delete_rejects_path_outside_sample_root(tmp_path, prepared_sample):
    repo = GalleryRepository(tmp_path / "gallery.db", tmp_path / "samples")
    person = repo.create_person("Synthetic")
    sample = repo.add_samples(person, [prepared_sample], "s", "m", "v")[0]
    outside = tmp_path / "keep.txt"
    outside.write_text("keep")
    with repo._connect() as connection:
        connection.execute("UPDATE samples SET image_path='keep.txt'")
        connection.commit()
    with pytest.raises(ValueError, match="样本目录"):
        repo.delete_sample(sample)
    assert outside.read_text() == "keep"
    assert len(repo.list_samples()) == 1


def test_manifest_detects_renamed_copy_across_splits(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"same image bytes")
    (tmp_path / "b.jpg").write_bytes(b"same image bytes")
    path = manifest(
        tmp_path,
        [
            ["a", "a.jpg", "validation", "false", "", "s1"],
            ["b", "b.jpg", "test", "false", "", "s2"],
        ],
    )
    with pytest.raises(ValueError, match="内容重复"):
        load_manifest(path, "test")


def test_completion_timestamp_includes_log_write(scene_service, monkeypatch):  # noqa: F811
    import app.service as module

    service = scene_service
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    clock = [10.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        service.repository, "log_recognition", lambda result: clock.__setitem__(0, 11.0)
    )
    result = service.recognize(image)
    assert result.end_to_end_ms == 1000
    assert "log_write" in result.timings_ms


def test_logging_failure_does_not_discard_recognition(scene_service, monkeypatch):  # noqa: F811
    def fail(result):
        raise OSError("full disk")

    monkeypatch.setattr(scene_service.repository, "log_recognition", fail)
    result = scene_service.recognize(np.zeros((200, 200, 3), dtype=np.uint8))
    assert result.status == "multiple_faces"
    assert "full disk" in result.warnings[0]


def test_integrity_report_detects_missing_orphan_and_incompatible(tmp_path, prepared_sample):
    repo = GalleryRepository(tmp_path / "g.db", tmp_path / "samples")
    person = repo.create_person("Synthetic")
    repo.add_samples(person, [prepared_sample], "s", "old-model", "v1")
    repo.list_samples()[0].image_path.unlink()
    (repo.samples_dir / "orphan.jpg").write_bytes(b"orphan")
    report = repo.integrity_report("new-model", "v1")
    assert (
        report["missing_images"] == report["orphan_images"] == report["incompatible_samples"] == 1
    )
