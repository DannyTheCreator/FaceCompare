import hashlib
import io
from dataclasses import replace

import pytest

from app.config import load_config
from app.model_manager import ModelDownload, download_models, verify_model


def test_cached_model_must_match_expected_hash(tmp_path):
    target = tmp_path / "model.onnx"
    target.write_bytes(b"valid content")
    item = ModelDownload(
        "test",
        "https://example.test/model",
        target,
        5,
        hashlib.sha256(b"valid content").hexdigest(),
    )
    assert verify_model(item) == item.expected_sha256
    target.write_bytes(b"corrupt but same size")
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_model(item)


def test_failed_download_preserves_previous_model(tmp_path, monkeypatch):
    import app.model_manager as module

    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    item = ModelDownload("test", "https://example.test/model", target, 1, "0" * 64)
    monkeypatch.setattr(module, "model_downloads", lambda config: (item,))
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong content")
    )
    config = replace(load_config(), root=tmp_path)
    with pytest.raises(RuntimeError):
        download_models(config, progress=lambda text: None, overwrite=True)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.glob("*.download")) == []


def test_successful_download_supports_models_outside_config_root(tmp_path, monkeypatch):
    import app.model_manager as module

    payload = b"valid model"
    target = tmp_path / "cache/model.onnx"
    item = ModelDownload(
        "test", "https://example.test/model", target, 1, hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(module, "model_downloads", lambda config: (item,))
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload)
    )
    config = replace(load_config(), root=tmp_path / "config-location")
    manifest = download_models(config, progress=lambda text: None)
    assert target.read_bytes() == payload
    assert manifest.is_file()
