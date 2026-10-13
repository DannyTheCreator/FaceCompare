from dataclasses import replace

import pytest

from app.config import StorageConfig, load_config
from app.repository import GalleryRepository
from experiments.protocol import fingerprint, verify_lock, write_lock
from tests.test_manifest import manifest


def setup_protocol(tmp_path):
    cfg = load_config()
    detector, recognizer = tmp_path / "detector", tmp_path / "recognizer"
    detector.write_bytes(b"fake-detector")
    recognizer.write_bytes(b"fake-recognizer")
    cfg = replace(
        cfg,
        storage=StorageConfig(tmp_path / "g.db", tmp_path / "samples"),
        models=replace(cfg.models, detector=detector, recognizer=recognizer),
    )
    repo = GalleryRepository(cfg.storage.database, cfg.storage.samples_dir)
    (tmp_path / "query.jpg").write_bytes(b"query")
    csv = manifest(tmp_path, [["q", "query.jpg", "test", "false", "", "independent-session"]])
    return cfg, repo, csv


@pytest.mark.parametrize("change", ["threshold", "gallery", "image", "model"])
def test_freeze_rejects_changed_inputs(tmp_path, change):
    cfg, repo, csv = setup_protocol(tmp_path)
    path = tmp_path / "freeze.json"
    original = fingerprint(cfg, repo, csv)
    write_lock(path, original)
    assert verify_lock(path, original) == original
    if change == "threshold":
        cfg = replace(cfg, matching=replace(cfg.matching, cosine_threshold=0.8))
    elif change == "gallery":
        repo.create_person("Synthetic")
    elif change == "image":
        (tmp_path / "query.jpg").write_bytes(b"changed")
    else:
        cfg.models.detector.write_bytes(b"changed")
    with pytest.raises(ValueError, match="冻结"):
        verify_lock(path, fingerprint(cfg, repo, csv))
    with pytest.raises(FileExistsError):
        write_lock(path, original)


def test_read_only_gallery_never_initializes_or_writes(tmp_path):
    cfg, repo, _ = setup_protocol(tmp_path)
    before = cfg.storage.database.read_bytes()
    reader = GalleryRepository(repo.database, repo.samples_dir, read_only=True)
    assert reader.list_persons() == []
    with pytest.raises(Exception, match="readonly"):
        reader.create_person("Must fail")
    with pytest.raises(ValueError, match="只读"):
        reader.cleanup_pending()
    assert cfg.storage.database.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        GalleryRepository(tmp_path / "absent/g.db", tmp_path / "absent/samples", read_only=True)
    assert not (tmp_path / "absent").exists()


def test_validation_freeze_then_final_test_cli(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    import numpy as np

    import experiments.evaluate as module
    from app.image_io import write_image
    from app.types import RecognitionResult

    cfg, repo, _ = setup_protocol(tmp_path)
    repo.create_person("Synthetic")
    csv_rows = []
    for index, (split, known) in enumerate(
        (("validation", True), ("validation", False), ("test", True), ("test", False))
    ):
        image = tmp_path / f"q{index}.png"
        write_image(image, np.full((8, 8, 3), 10 + index, dtype=np.uint8))
        csv_rows.append([str(index), image.name, split, str(known), "1" if known else "", split])
    csv_path = manifest(tmp_path, csv_rows)
    current = [cfg]
    monkeypatch.setattr(module, "load_config", lambda path: current[0])

    def service(config, *, read_only):
        assert read_only

        def recognize(image, **kwargs):
            score = 0.9 if int(image[0, 0, 0]) % 2 == 0 else 0.1
            return RecognitionResult(
                1, 0, 0, "matched", "synthetic", 0.42, candidate_person_id=1, score=score
            )

        return SimpleNamespace(
            repository=repo,
            matcher=SimpleNamespace(
                threshold=config.matching.cosine_threshold,
                snapshot=SimpleNamespace(person_ids=(1,)),
            ),
            model_hash="synthetic",
            _gallery_version=repo.gallery_version(),
            recognize=recognize,
        )

    monkeypatch.setattr(module, "FaceComparisonService", service)
    lock = tmp_path / "freeze.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            str(csv_path),
            "--split",
            "validation",
            "--calibrate-target-far",
            "0.05",
            "--freeze-output",
            str(lock),
            "--output",
            str(tmp_path / "validation"),
        ],
    )
    assert module.main() == 0
    threshold = json.loads(lock.read_text())["settings"]["matching"]["cosine_threshold"]
    current[0] = replace(cfg, matching=replace(cfg.matching, cosine_threshold=threshold))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            str(csv_path),
            "--split",
            "test",
            "--freeze",
            str(lock),
            "--output",
            str(tmp_path / "test"),
        ],
    )
    assert module.main() == 0
    assert json.loads((tmp_path / "test/summary.json").read_text())["frozen"] is True
    current[0] = cfg
    with pytest.raises(ValueError, match="冻结"):
        module.main()
