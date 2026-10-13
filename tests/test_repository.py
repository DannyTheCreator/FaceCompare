import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.repository import GalleryRepository


def test_person_crud_and_gallery_version(tmp_path: Path):
    repository = GalleryRepository(tmp_path / "gallery.db", tmp_path / "samples")
    initial_version = repository.gallery_version()
    person_id = repository.create_person("测试人员", "student-1")
    persons = repository.list_persons()
    assert len(persons) == 1
    assert persons[0].person_id == person_id
    assert persons[0].sample_count == 0
    assert repository.gallery_version() == initial_version + 1
    repository.delete_person(person_id)
    assert repository.list_persons() == []


def test_external_id_is_unique(tmp_path: Path):
    repository = GalleryRepository(tmp_path / "gallery.db", tmp_path / "samples")
    repository.create_person("甲", "same-id")
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_person("乙", "same-id")


def test_samples_persist_with_unicode_paths_and_version_filter(tmp_path, prepared_sample):
    repository = GalleryRepository(tmp_path / "数据库/gallery.db", tmp_path / "人脸样本")
    person = repository.create_person("测试人员")
    ids = repository.add_samples(person, [prepared_sample], "session-a", "model-a", "v1")
    records = repository.list_samples(person)
    assert len(records) == 1 and records[0].image_path.is_file()
    from app.image_io import read_image

    assert read_image(records[0].image_path).shape == prepared_sample.image.shape
    assert repository.load_features("different-model", "v1") == []
    reopened = GalleryRepository(repository.database, repository.samples_dir)
    features = reopened.load_features("model-a", "v1")
    np.testing.assert_allclose(np.linalg.norm(features[0].feature), 1)
    reopened.delete_sample(ids[0])
    assert not records[0].image_path.exists()
    assert reopened.list_persons()[0].sample_count == 0


def test_write_failure_rolls_back_files_and_rows(tmp_path, prepared_sample, monkeypatch):
    import app.repository as module

    repository = GalleryRepository(tmp_path / "gallery.db", tmp_path / "samples")
    person = repository.create_person("测试")
    before = repository.gallery_version()

    def fail_after_partial_write(path, image):
        path.write_bytes(b"incomplete")
        raise OSError("disk full")

    monkeypatch.setattr(module, "write_image", fail_after_partial_write)
    with pytest.raises(OSError):
        repository.add_samples(person, [prepared_sample], "s", "m", "v1")
    assert repository.list_samples() == []
    assert list(repository.samples_dir.rglob("*.jpg")) == []
    assert repository.gallery_version() == before


def test_invalid_feature_never_enters_database(tmp_path, prepared_sample):
    repository = GalleryRepository(tmp_path / "gallery.db", tmp_path / "samples")
    person = repository.create_person("测试")
    invalid = replace(prepared_sample, feature=np.array([float("nan"), 1]))
    with pytest.raises(ValueError):
        repository.add_samples(person, [invalid], "s", "m", "v1")
    assert repository.list_samples() == []


def test_dimension_mismatch_does_not_corrupt_existing_gallery(tmp_path, prepared_sample):
    repository = GalleryRepository(tmp_path / "gallery.db", tmp_path / "samples")
    person = repository.create_person("测试")
    repository.add_samples(person, [prepared_sample], "s1", "m", "v1")
    wrong_dim = replace(prepared_sample, feature=np.array([1, 0, 0], dtype=np.float32))
    with pytest.raises(ValueError, match="维度"):
        repository.add_samples(person, [wrong_dim], "s2", "m", "v1")
    assert len(repository.list_samples()) == 1
    assert len(list(repository.samples_dir.rglob("*.jpg"))) == 1
