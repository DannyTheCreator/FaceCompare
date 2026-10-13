from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import __version__
from .config import AppConfig


@dataclass(frozen=True)
class ModelDownload:
    name: str
    url: str
    destination: Path
    minimum_bytes: int
    expected_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_downloads(config: AppConfig) -> tuple[ModelDownload, ...]:
    base = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
    return (
        ModelDownload(
            "YuNet 2023mar",
            f"{base}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            config.models.detector,
            100_000,
            "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        ),
        ModelDownload(
            "SFace 2021dec",
            f"{base}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            config.models.recognizer,
            10_000_000,
            "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        ),
    )


def verify_model(item: ModelDownload, path: Path | None = None) -> str:
    """Check size and the expected hash, including for already cached files."""
    model_path = item.destination if path is None else path
    if not model_path.is_file() or model_path.stat().st_size < item.minimum_bytes:
        raise RuntimeError(f"模型缺失或不完整: {item.destination.name}")
    actual = _sha256(model_path)
    if actual != item.expected_sha256:
        raise RuntimeError(f"模型 SHA-256 不匹配: {item.destination.name}，请重新下载官方版本")
    return actual


def verify_models(config: AppConfig) -> None:
    for item in model_downloads(config):
        verify_model(item)


def download_models(
    config: AppConfig, progress: Callable[[str], None] = print, overwrite: bool = False
) -> Path:
    """Download content-pinned OpenCV Zoo models and verify expected hashes.

    Files are first downloaded to a sibling temporary path and atomically moved
    into place, preventing interrupted downloads from looking like valid models.
    """

    manifest: dict[str, object] = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": [],
    }
    for item in model_downloads(config):
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        if item.destination.exists() and not overwrite:
            progress(f"已存在，跳过: {item.destination.name}")
        else:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{item.destination.name}.", suffix=".download", dir=item.destination.parent
            )
            os.close(fd)
            temporary = Path(temporary_name)
            progress(f"下载 {item.name} ...")
            try:
                request = urllib.request.Request(
                    item.url, headers={"User-Agent": f"FaceCompare/{__version__}"}
                )
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    temporary.open("wb") as handle,
                ):
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                verify_model(item, temporary)
                os.replace(temporary, item.destination)
            finally:
                temporary.unlink(missing_ok=True)
        size = item.destination.stat().st_size
        actual_hash = verify_model(item)
        try:
            file_label = item.destination.relative_to(config.root).as_posix()
        except ValueError:
            file_label = str(item.destination)
        manifest["models"].append(
            {
                "name": item.name,
                "file": file_label,
                "source": item.url,
                "bytes": size,
                "sha256": actual_hash,
                "expected_sha256": item.expected_sha256,
            }
        )
        progress(f"校验完成: {item.destination.name} ({size} bytes)")
    manifest_path = config.root / "models" / "model_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, manifest_temp = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(manifest_temp).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(manifest_temp, manifest_path)
    finally:
        Path(manifest_temp).unlink(missing_ok=True)
    return manifest_path
