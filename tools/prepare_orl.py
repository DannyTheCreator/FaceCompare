"""Prepare a reproducible local AT&T/ORL image-split benchmark, never a session split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import load_config
from app.image_io import write_image
from app.model_manager import verify_models
from app.recognizer import sha256_file
from app.service import FaceComparisonService

URL = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
SOURCE = "https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html"
SHA256 = "509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec"


def download(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=30) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("Archive exceeds bounded size")
        import hashlib

        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Dataset archive checksum mismatch")
        with path.open("xb") as handle:
            handle.write(data)
    if sha256_file(path) != SHA256:
        raise ValueError("Dataset archive checksum mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/public/orl/att_faces.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public/orl/run-01")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; never overwrite an earlier experiment")
    download(args.archive)
    cfg = load_config()
    verify_models(cfg)
    cv2.setNumThreads(1)
    output.mkdir(parents=True)
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    raw["models"] = {"detector": str(cfg.models.detector), "recognizer": str(cfg.models.recognizer)}
    raw["storage"] = {
        "database": str(output / "gallery.db"),
        "samples_dir": str(output / "samples"),
    }
    # Dataset-specific enrollment permits a subject with any usable enrollment image;
    # every failed enrollment attempt is retained below. Application defaults stay unchanged.
    raw["enrollment"]["minimum_samples"] = 1
    # Fixed adaptation established on enrollment images only: interpolation of 92x112
    # sources invalidates the camera-resolution Laplacian cutoff. Other gates stay on.
    raw["quality"]["min_blur_variance"] = 0.0
    config = output / "config.json"
    config.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    service = FaceComparisonService(load_config(config))
    images = {}
    with zipfile.ZipFile(args.archive) as archive:
        for subject in range(1, 41):
            for index in range(1, 11):
                name = f"s{subject}/{index}.pgm"
                entry = archive.getinfo(name)
                if entry.file_size > 100000:
                    raise ValueError("Unexpected image size")
                image = cv2.imdecode(
                    np.frombuffer(archive.read(entry), dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None or image.shape[:2] != (112, 92):
                    raise ValueError(f"Unexpected ORL input: {name}")
                # Fixed before looking at test outputs; no manual boxes or identity-specific tuning.
                image = cv2.resize(image, (368, 448), interpolation=cv2.INTER_CUBIC)
                path = output / "images" / f"s{subject}" / f"{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                write_image(path, image)
                images[subject, index] = path
    enrolled = {}
    attempts = []
    for subject in range(1, 21):
        prepared = []
        for index in range(1, 6):
            image = cv2.imdecode(
                np.fromfile(images[subject, index], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            sample, quality = service.prepare_enrollment_frame(image)
            attempts.append({"subject": subject, "image": index, "status": quality.code})
            if sample is not None:
                prepared.append(sample)
        if not prepared:
            (output / "enrollment-failure.json").write_text(
                json.dumps(attempts, indent=2), encoding="utf-8"
            )
            raise ValueError(
                f"Enrollment failed for predefined subject {subject}; do not silently omit it"
            )
        enrolled[subject] = service.commit_enrollment(
            prepared, name=f"ORL-{subject:02d}", session_id="orl-session-unavailable"
        )
    manifest = output / "manifest.csv"
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
                "split_unit",
            ]
        )
        for subject in range(1, 41):
            indices = range(6, 11) if subject <= 20 else range(1, 11)
            for index in indices:
                split = (
                    ("validation" if index <= 7 else "test")
                    if subject <= 20
                    else ("validation" if subject <= 30 else "test")
                )
                writer.writerow(
                    [
                        f"orl-s{subject}-{index}",
                        str(images[subject, index].relative_to(output)),
                        split,
                        subject <= 20,
                        enrolled.get(subject, ""),
                        "ORL-gray-fixed-4x",
                        "orl-session-unavailable",
                        "image",
                    ]
                )
    provenance = {
        "source": SOURCE,
        "download_url": URL,
        "archive_sha256": SHA256,
        "credit": "AT&T Laboratories Cambridge",
        "split_unit": "image",
        "session_metadata": "unavailable",
        "resize": "fixed 4x INTER_CUBIC, all inputs",
        "known_subjects": list(range(1, 21)),
        "quality_adaptation": "Laplacian blur gate disabled for low-resolution public images; default camera config unchanged",
        "unknown_validation_subjects": list(range(21, 31)),
        "unknown_test_subjects": list(range(31, 41)),
        "enrollment_attempts": attempts,
        "accepted_enrollment": len(service.repository.list_samples()),
        "note": "Local research benchmark. Not a camera/session-isolated evaluation. Source images are not covered by the project's MIT license.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Prepared {manifest}; accepted enrollment {provenance['accepted_enrollment']}/100")


if __name__ == "__main__":
    main()
