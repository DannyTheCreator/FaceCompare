from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.image_io import read_image
from app.service import FaceComparisonService
from experiments.common import (
    RawPrediction,
    calculate_metrics,
    load_manifest,
    validate_gallery_isolation,
    write_json,
)
from experiments.protocol import fingerprint, verify_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人脸比对 B0/B1/B2/B4 性能对照")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--freeze", type=Path, help="核验已冻结实验；省略则仅为探索性性能测量")
    parser.add_argument("--opencv-threads", type=int, help="固定 OpenCV CPU 线程数")
    parser.add_argument(
        "--allow-image-split",
        action="store_true",
        help="允许公开图片划分的探索性基准，不声称会话隔离",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/results/benchmark"))
    return parser.parse_args()


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats 必须至少为 1")
    if args.opencv_threads is not None:
        if args.opencv_threads < 1:
            raise ValueError("opencv-threads 必须为正数")
        cv2.setNumThreads(args.opencv_threads)
    config = load_config(args.config)
    service = FaceComparisonService(config, read_only=True)
    before = fingerprint(config, service.repository, args.manifest)
    if before["gallery_version"] != service._gallery_version:
        raise ValueError("建快照时标准库发生变化，请停止写入后重试")
    if args.freeze:
        verify_lock(args.freeze, before)
    items = load_manifest(args.manifest, args.split)
    if any(item.split_unit == "image" for item in items) and not args.allow_image_split:
        raise ValueError("公开图片划分需显式 --allow-image-split")

    validate_gallery_isolation(
        items, service.matcher.snapshot.person_ids, service.repository.list_samples()
    )

    loaded = []
    for item in items:
        image = read_image(item.image_path)
        if image is None:
            raise ValueError(f"测试图像无法读取，停止对照以保留完整测试集: {item.image_path}")
        loaded.append((item, image))
    if not loaded:
        raise ValueError("没有可读取的测试图像")

    variants = {
        "B0_original_loop": (1.0, "loop"),
        "B1_scaled_loop": (config.detection.detect_scale, "loop"),
        "B2_original_matrix": (1.0, "matrix"),
        "B4_scaled_matrix": (config.detection.detect_scale, "matrix"),
    }
    # Warm-up is separate from measured repeats.
    run_rows: list[dict[str, object]] = []
    predictions_by_variant: dict[str, list[RawPrediction]] = {name: [] for name in variants}

    for variant, (scale, match_mode) in variants.items():
        service.recognize(loaded[0][1], detect_scale=scale, match_mode=match_mode, write_log=False)
        for repeat in range(1, args.repeats + 1):
            for index, (item, image) in enumerate(loaded, start=1):
                result = service.recognize(
                    image,
                    frame_id=index,
                    detect_scale=scale,
                    match_mode=match_mode,
                    write_log=False,
                )
                row: dict[str, object] = {
                    "variant": variant,
                    "repeat": repeat,
                    "sample_id": item.sample_id,
                    "status": result.status,
                    "score": result.score,
                    "candidate_person_id": result.candidate_person_id,
                }
                row.update({f"{key}_ms": value for key, value in result.timings_ms.items()})
                run_rows.append(row)
                if repeat == 1:
                    predictions_by_variant[variant].append(
                        RawPrediction(
                            item.sample_id,
                            str(item.image_path),
                            item.split,
                            item.is_known,
                            item.expected_person_id,
                            item.scene,
                            item.session_id,
                            result.status,
                            result.candidate_person_id,
                            result.candidate_name,
                            result.score,
                            result.timings_ms.get("detect"),
                            result.timings_ms.get("quality"),
                            result.timings_ms.get("align_and_feature"),
                            result.timings_ms.get("match"),
                            result.timings_ms.get("algorithm_total"),
                        )
                    )
        print(f"完成 {variant}")

    output = args.output.resolve()
    if fingerprint(config, service.repository, args.manifest) != before:
        raise ValueError("性能测量期间输入或标准库变化")
    output.mkdir(parents=True, exist_ok=True)
    all_fields = [
        "variant",
        "repeat",
        "sample_id",
        "status",
        "score",
        "candidate_person_id",
        "detect_ms",
        "quality_ms",
        "align_and_feature_ms",
        "match_ms",
        "algorithm_total_ms",
    ]
    with (output / "benchmark_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(run_rows)

    summary_rows: list[dict[str, object]] = []
    stages = ["detect_ms", "align_and_feature_ms", "match_ms", "algorithm_total_ms"]
    for variant in variants:
        variant_rows = [row for row in run_rows if row["variant"] == variant]
        summary: dict[str, object] = {"variant": variant}
        for stage in stages:
            values = [float(row[stage]) for row in variant_rows if row.get(stage) is not None]
            summary[f"{stage}_mean"] = statistics.fmean(values) if values else None
            summary[f"{stage}_median"] = statistics.median(values) if values else None
            summary[f"{stage}_p50"] = statistics.median(values) if values else None
            summary[f"{stage}_p95"] = percentile95(values)
        metrics = calculate_metrics(predictions_by_variant[variant], service.matcher.threshold)
        summary["known_rate"] = metrics["known_end_to_end_recognition_rate"]
        summary["unknown_far_all"] = metrics["unknown_false_accept_rate_all_attempts"]
        summary["unknown_far_valid"] = metrics["unknown_false_accept_rate_valid_queries"]
        summary_rows.append(summary)
    summary_fields = list(summary_rows[0])
    with (output / "benchmark_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(
        output / "run_metadata.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest": str(args.manifest.resolve()),
            "split": args.split,
            "repeats": args.repeats,
            "model_hash": service.model_hash,
            "gallery_version": service.repository.gallery_version(),
            "threshold": service.matcher.threshold,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "variants": variants,
            "protocol": before,
            "frozen": args.freeze is not None,
        },
    )
    print(f"结果目录: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
