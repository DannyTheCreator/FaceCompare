"""Exercise existing B0-B4 CLIs with constant images/video and an isolated empty gallery."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import load_config
from app.image_io import write_image
from app.repository import GalleryRepository


def main():
    output = ROOT / "profile_output/synthetic"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="facecompare-benchmark-") as temporary:
        root = Path(temporary)
        cfg = load_config()
        raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        raw["models"] = {
            "detector": str(cfg.models.detector),
            "recognizer": str(cfg.models.recognizer),
        }
        raw["storage"] = {
            "database": str(root / "gallery.db"),
            "samples_dir": str(root / "samples"),
        }
        config = root / "config.json"
        config.write_text(json.dumps(raw), encoding="utf-8")
        GalleryRepository(root / "gallery.db", root / "samples")
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sample_id",
                    "image_path",
                    "split",
                    "is_known",
                    "expected_person_id",
                    "scene",
                    "session_id",
                ]
            )
            for i in range(4):
                image = root / f"constant-{i}.png"
                write_image(image, np.full((480, 640, 3), 50 + i * 40, dtype=np.uint8))
                writer.writerow(
                    [
                        f"synthetic-{i}",
                        image.name,
                        "test",
                        "false",
                        "",
                        "synthetic-no-face",
                        "synthetic-session",
                    ]
                )
        video = root / "constant.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 120, (640, 480))
        if not writer.isOpened():
            raise RuntimeError("MJPG video writer unavailable")
        for i in range(48):
            writer.write(np.full((480, 640, 3), 100 + i, dtype=np.uint8))
        writer.release()
        for script, args in (
            ("benchmark.py", [str(manifest), "--split", "test"]),
            ("video_benchmark.py", [str(video), "--max-seconds", "1"]),
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "experiments" / script),
                    *args,
                    "--config",
                    str(config),
                    "--repeats",
                    "3",
                    "--opencv-threads",
                    "1",
                    "--output",
                    str(output / script.removesuffix(".py")),
                ],
                check=True,
            )
    print("Synthetic engineering smoke only: no accuracy or face-tracking evidence.")


if __name__ == "__main__":
    main()
