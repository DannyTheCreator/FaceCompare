"""Interleave archived baseline/current matcher on identical synthetic queries."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.matcher import FaceMatcher, FeatureRecord, GallerySnapshot
from app.recognizer import sha256_file
from experiments.protocol import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Trusted locally generated source baseline ZIP")
    parser.add_argument("--output", type=Path, default=ROOT / "profile_output/comparison.json")
    args = parser.parse_args()
    module = types.ModuleType("app._baseline_matcher")
    module.__package__ = "app"
    sys.modules[module.__name__] = module
    with zipfile.ZipFile(args.baseline) as archive:
        exec(
            compile(archive.read("app/matcher.py"), "baseline/app/matcher.py", "exec"),
            module.__dict__,
        )
    results = []
    for seed in (41, 42, 43):
        rng = np.random.default_rng(seed)
        for size in (3, 1000, 10000):
            matcher = FaceMatcher(0.42)
            matcher.publish(
                GallerySnapshot.from_records(
                    FeatureRecord(i, str(i), rng.normal(size=128)) for i in range(size)
                )
            )
            queries = rng.normal(size=(100, 128)).astype(np.float32)

            def old(q):
                return module.FaceMatcher.rank(matcher, q, 3, "matrix")

            def new(q):
                return matcher.rank(q, 3, "matrix")

            for query in queries[:10]:
                old(query)
                new(query)
            times = {"before": [], "after": []}
            changed = 0
            max_delta = 0.0
            for index, query in enumerate(queries):
                outputs = {}
                order = (
                    (("before", old), ("after", new))
                    if index % 2 == 0
                    else (("after", new), ("before", old))
                )
                for label, operation in order:
                    started = time.perf_counter_ns()
                    outputs[label] = operation(query)
                    times[label].append((time.perf_counter_ns() - started) / 1e6)
                changed += [c.person_id for c in outputs["before"]] != [
                    c.person_id for c in outputs["after"]
                ]
                max_delta = max(
                    max_delta,
                    max(
                        abs(a.score - b.score) for a, b in zip(outputs["before"], outputs["after"])
                    ),
                )
            results.append(
                {
                    "seed": seed,
                    "identities": size,
                    "changed_topk_queries": changed,
                    "max_score_delta": max_delta,
                    **{
                        label: {
                            "p50_ms": float(np.percentile(values, 50)),
                            "p95_ms": float(np.percentile(values, 95)),
                            "raw_ms": values,
                        }
                        for label, values in times.items()
                    },
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "environment": environment(),
                "baseline_sha256": sha256_file(args.baseline),
                "matcher_sha256": sha256_file(ROOT / "app/matcher.py"),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for result in results:
        print(
            {
                key: (
                    {k: v for k, v in value.items() if k != "raw_ms"}
                    if isinstance(value, dict)
                    else value
                )
                for key, value in result.items()
            }
        )


if __name__ == "__main__":
    main()
