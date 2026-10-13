"""Local reproducibility locks. Hash logical gallery contents, not volatile WAL bytes."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from app.recognizer import sha256_file
from experiments.common import load_manifest


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "opencv_threads": cv2.getNumThreads(),
    }


def fingerprint(config, repository, manifest: Path | None = None) -> dict:
    settings = asdict(config)
    for key in ("root", "storage", "models"):
        settings.pop(key)
    with repository._connect() as connection:
        connection.execute("BEGIN")
        persons = [
            tuple(row) for row in connection.execute("SELECT * FROM persons ORDER BY person_id")
        ]
        samples = [
            tuple(row)
            for row in connection.execute(
                "SELECT sample_id, person_id, session_id, hex(feature), feature_dim, quality_json, model_hash, preprocess_version FROM samples ORDER BY sample_id"
            )
        ]
        version = connection.execute(
            "SELECT value FROM metadata WHERE key='gallery_version'"
        ).fetchone()[0]
        images = [
            (row[0], sha256_file(repository._sample_path(row[1])))
            for row in connection.execute(
                "SELECT sample_id, image_path FROM samples ORDER BY sample_id"
            )
        ]
    root = Path(__file__).resolve().parents[1]
    source = {
        str(path.relative_to(root)): sha256_file(path)
        for directory in ("app", "experiments")
        for path in sorted((root / directory).rglob("*.py"))
    }
    result = {
        "schema": 1,
        "settings": settings,
        "gallery_digest": digest([persons, samples, images, version]),
        "gallery_version": int(version),
        "detector_sha256": sha256_file(config.models.detector),
        "recognizer_sha256": sha256_file(config.models.recognizer),
        "source_digest": digest(source),
        "dependencies_digest": sha256_file(root / "requirements.txt"),
        "environment": environment(),
    }
    if manifest is not None:
        items = load_manifest(manifest)
        result["dataset_digest"] = digest(
            [
                [
                    item.sample_id,
                    item.split,
                    item.is_known,
                    item.expected_person_id,
                    item.scene,
                    item.session_id,
                    item.split_unit,
                    sha256_file(item.image_path),
                ]
                for item in items
            ]
        )
    # JSON round-trip normalizes tuple config fields for comparisons after reloading a lock.
    if manifest is not None:
        result["split_units"] = sorted({item.split_unit for item in items})
    return json.loads(json.dumps(result))


def verify_lock(path: Path, current: dict) -> dict:
    locked = json.loads(path.read_text(encoding="utf-8"))
    if locked != current:
        changed = sorted(
            key for key in set(locked) | set(current) if locked.get(key) != current.get(key)
        )
        raise ValueError(f"冻结核验失败，发生变化: {', '.join(changed)}")
    return locked


def write_lock(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A frozen run must not silently overwrite earlier evidence.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
