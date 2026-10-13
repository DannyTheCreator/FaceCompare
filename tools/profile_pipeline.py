"""Bounded synthetic CPU microprofile: not a biometric accuracy experiment."""

from __future__ import annotations

import argparse
import cProfile
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.detector import FaceDetector
from app.matcher import FaceMatcher, FeatureRecord, GallerySnapshot
from app.recognizer import FaceRecognizer, sha256_file
from experiments.protocol import environment


def measure(operation, repeats):
    for _ in range(5):
        operation()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "raw_ms": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "profile_output")
    args = parser.parse_args()
    if args.repeats < 10:
        raise ValueError("At least 10 repeats required")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cv2.setNumThreads(1)
    rng = np.random.default_rng(42)
    rows, outputs = {}, {}
    for size in (3, 1000, 10000):
        matcher = FaceMatcher(cfg.matching.cosine_threshold)
        matcher.publish(
            GallerySnapshot.from_records(
                FeatureRecord(i, f"synthetic-{i}", rng.normal(size=128)) for i in range(size)
            )
        )
        query = rng.normal(size=128).astype(np.float32)
        for mode in ("loop", "matrix"):
            key = f"match_{size}_{mode}"
            rows[key] = measure(lambda: matcher.rank(query, 3, mode), args.repeats)
            outputs[key] = [[c.person_id, c.score] for c in matcher.rank(query, 3, mode)]
    detector = FaceDetector(cfg.models.detector, cfg.detection)
    recognizer = FaceRecognizer(cfg.models.recognizer)
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Deliberately artificial landmarks exercise SFace only; there is no face/identity claim.
    face = np.array(
        [100, 100, 120, 140, 130, 140, 190, 140, 160, 170, 140, 200, 180, 200, 1], dtype=np.float32
    )
    for scale in (1.0, 0.5):
        rows[f"detect_{scale}"] = measure(
            lambda: detector.detect(blank, scale), min(args.repeats, 30)
        )
        outputs[f"detect_{scale}"] = len(detector.detect(blank, scale))
    rows["align_and_feature_artificial"] = measure(
        lambda: recognizer.extract(blank, face), min(args.repeats, 30)
    )
    profiler = cProfile.Profile()
    profiler.runcall(matcher.rank, query, 3, "matrix")
    profiler.dump_stats(str(args.output / f"{args.label}.prof"))
    result = {
        "label": args.label,
        "input": "seed=42; synthetic float32 embeddings; constant 640x480 image; artificial landmarks",
        "environment": environment(),
        "models": {
            "detector": sha256_file(cfg.models.detector),
            "recognizer": recognizer.model_hash,
        },
        "matcher_source": sha256_file(ROOT / "app/matcher.py"),
        "measurements": rows,
        "outputs": outputs,
    }
    (args.output / f"{args.label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: {k: v for k, v in value.items() if k != "raw_ms"} for key, value in rows.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
