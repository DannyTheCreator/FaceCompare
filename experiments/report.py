"""Turn saved predictions into score, operating-point and scene-analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.common import RawPrediction, calculate_metrics, write_json


def load_predictions(path: Path) -> list[RawPrediction]:
    """Read the evaluate.py schema, ignoring its already-computed decision column."""
    fields = set(RawPrediction.__dataclass_fields__)
    integer_fields = {"expected_person_id", "candidate_person_id"}
    float_fields = {
        "score",
        "detect_ms",
        "quality_ms",
        "align_and_feature_ms",
        "match_ms",
        "algorithm_total_ms",
    }
    rows = []
    ids = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if fields - set(reader.fieldnames or []):
            raise ValueError("预测CSV缺少字段，请使用evaluate.py生成的predictions.csv")
        for row in reader:
            payload = {key: row[key] for key in fields}
            if not payload["sample_id"] or payload["sample_id"] in ids:
                raise ValueError("预测样本ID不能为空或重复")
            ids.add(payload["sample_id"])
            value = payload["is_known"].strip().lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError("is_known 必须是布尔值")
            payload["is_known"] = value in {"true", "1"}
            for key in integer_fields:
                payload[key] = int(payload[key]) if payload[key] else None
            for key in float_fields:
                payload[key] = float(payload[key]) if payload[key] else None
                if payload[key] is not None and not math.isfinite(payload[key]):
                    raise ValueError(f"{key} 包含非有限值")
            if payload["score"] is not None and not -1 <= payload["score"] <= 1:
                raise ValueError("余弦分数必须在 [-1, 1] 内")
            rows.append(RawPrediction(**payload))
    if not rows:
        raise ValueError("预测CSV没有样本")
    return rows


def build_report(rows: list[RawPrediction], threshold: float) -> dict:
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("阈值必须在 [-1, 1] 内")
    curves = []
    for value in np.linspace(-1, 1, 201):
        metrics = calculate_metrics(rows, float(value))
        curves.append(
            {
                "threshold": float(value),
                "known_rate": metrics["known_end_to_end_recognition_rate"],
                "unknown_far_valid": metrics["unknown_false_accept_rate_valid_queries"],
            }
        )
    scenes = {
        scene: calculate_metrics([row for row in rows if row.scene == scene], threshold)
        for scene in sorted({row.scene for row in rows})
    }
    return {
        "metrics": calculate_metrics(rows, threshold),
        "scenes": scenes,
        "curve": curves,
        "curve_definition": "1:N known end-to-end correct-identification rate versus effective unknown FAR; not a pair-verification ROC",
        "note": "Descriptive analysis only. Select thresholds on validation and keep test parameters frozen.",
    }


def render_report(rows: list[RawPrediction], report: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    threshold = report["metrics"]["threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    valid_statuses = {"matched", "unknown"}
    for known, label, color in (
        (True, "Known queries", "#2563eb"),
        (False, "Unknown queries", "#e5892a"),
    ):
        scores = [
            row.score
            for row in rows
            if row.is_known == known
            and row.score is not None
            and row.processing_status in valid_statuses
            and row.candidate_person_id is not None
        ]
        if scores:
            axes[0].hist(scores, bins=np.linspace(-1, 1, 31), alpha=0.6, label=label, color=color)
    axes[0].axvline(
        threshold, color="#111827", linestyle="--", label=f"Configured threshold {threshold:.3f}"
    )
    axes[0].set(
        xlabel="Maximum identity cosine similarity",
        ylabel="Query count",
        title="Score distribution (valid queries)",
    )
    axes[0].legend()
    points = [
        point
        for point in report["curve"]
        if point["known_rate"] is not None and point["unknown_far_valid"] is not None
    ]
    if points:
        axes[1].plot(
            [p["unknown_far_valid"] for p in points],
            [p["known_rate"] for p in points],
            color="#2563eb",
        )
        current = report["metrics"]
        axes[1].scatter(
            [current["unknown_false_accept_rate_valid_queries"]],
            [current["known_end_to_end_recognition_rate"]],
            color="#e5892a",
            label="Configured threshold",
            zorder=5,
        )
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "Known and valid unknown queries required",
            ha="center",
            va="center",
            wrap=True,
        )
    axes[1].set(
        xlabel="Effective unknown false acceptance rate",
        ylabel="Known correct-identification rate (all attempts)",
        title="Open-set operating curve",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    axes[1].grid(alpha=0.2)
    fig.savefig(output / "analysis.png", dpi=160)
    plt.close(fig)
    write_json(output / "analysis.json", report)
    with (output / "operating_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "known_rate", "unknown_far_valid"])
        writer.writeheader()
        writer.writerows(report["curve"])
    # Aggregate counts by scene; do not export names, identity IDs or image paths.
    with (output / "scene_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        keys = [
            "scene",
            "attempts",
            "known_attempts",
            "unknown_attempts",
            "known_end_to_end_recognition_rate",
            "unknown_false_accept_rate_valid_queries",
        ]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for scene, metrics in report["scenes"].items():
            writer.writerow({"scene": scene, **{key: metrics[key] for key in keys[1:]}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Already fixed threshold; no calibration on test",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/results/report"))
    args = parser.parse_args()
    rows = load_predictions(args.predictions)
    render_report(rows, build_report(rows, args.threshold), args.output)
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
