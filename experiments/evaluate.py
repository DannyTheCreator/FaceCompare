from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.image_io import read_image
from app.service import FaceComparisonService
from experiments.common import (
    RawPrediction,
    calculate_metrics,
    choose_threshold,
    load_manifest,
    validate_gallery_isolation,
    write_json,
    write_predictions,
)
from experiments.protocol import fingerprint, verify_lock, write_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="开放集人脸识别离线评估")
    parser.add_argument("manifest", type=Path, help="CSV 数据清单")
    parser.add_argument("--split", default="test", help="只评估指定 split")
    parser.add_argument("--config", type=Path, help="配置文件路径")
    parser.add_argument("--opencv-threads", type=int, help="固定 OpenCV CPU 线程数")
    parser.add_argument(
        "--allow-image-split",
        action="store_true",
        help="明确允许无会话元数据的公开图片基准；不视为会话隔离证据",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/results/evaluation"))
    parser.add_argument("--threshold", type=float, help="覆盖 config.json 中的阈值")
    parser.add_argument("--freeze", type=Path, help="最终测试必需的验证阶段冻结记录")
    parser.add_argument("--freeze-output", type=Path, help="验证标定后生成冻结记录，不覆盖已有文件")
    parser.add_argument(
        "--calibrate-target-far",
        type=float,
        help="在所选 split 上标定阈值，使有效未知查询 FAR 不高于该值",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.calibrate_target_far is not None and args.split != "validation":
        raise ValueError("阈值标定只允许使用 validation split，不能在 test 集上调参")
    if args.split == "test" and (args.freeze is None or args.threshold is not None):
        raise ValueError("最终 test 必须提供 --freeze，不能覆盖冻结阈值")
    if args.freeze_output and (args.split != "validation" or args.calibrate_target_far is None):
        raise ValueError("--freeze-output 只用于 validation 阈值标定")
    if args.threshold is not None and not -1 <= args.threshold <= 1:
        raise ValueError("阈值必须在 [-1, 1] 内")
    config = load_config(args.config)
    if args.opencv_threads is not None:
        import cv2

        if args.opencv_threads < 1:
            raise ValueError("opencv-threads 必须为正数")
        cv2.setNumThreads(args.opencv_threads)
    service = FaceComparisonService(config, read_only=True)
    before = fingerprint(config, service.repository, args.manifest)
    if before["gallery_version"] != service._gallery_version:
        raise ValueError("建快照时标准库发生变化，请停止写入后重试")
    if args.freeze:
        verify_lock(args.freeze, before)
    items = load_manifest(args.manifest, args.split)
    if any(item.split_unit == "image" for item in items) and not args.allow_image_split:
        raise ValueError("公开图片划分需显式 --allow-image-split；不能冒充会话隔离")

    validate_gallery_isolation(
        items, service.matcher.snapshot.person_ids, service.repository.list_samples()
    )

    predictions: list[RawPrediction] = []
    for index, item in enumerate(items, start=1):
        image = read_image(item.image_path)
        if image is None:
            predictions.append(
                RawPrediction(
                    item.sample_id,
                    str(item.image_path),
                    item.split,
                    item.is_known,
                    item.expected_person_id,
                    item.scene,
                    item.session_id,
                    "image_read_failure",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        result = service.recognize(image, frame_id=index, write_log=False)
        timings = result.timings_ms
        predictions.append(
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
                timings.get("detect"),
                timings.get("quality"),
                timings.get("align_and_feature"),
                timings.get("match"),
                timings.get("algorithm_total"),
            )
        )
        print(f"[{index}/{len(items)}] {item.sample_id}: {result.status}")

    if args.calibrate_target_far is not None:
        threshold, metrics = choose_threshold(predictions, args.calibrate_target_far)
    else:
        threshold = args.threshold if args.threshold is not None else service.matcher.threshold
        metrics = calculate_metrics(predictions, threshold)

    output = args.output.resolve()
    if fingerprint(config, service.repository, args.manifest) != before:
        raise ValueError("运行期间配置、代码、标准库或数据变化，结果无效")
    if args.freeze_output:
        frozen = replace(config, matching=replace(config.matching, cosine_threshold=threshold))
        write_lock(args.freeze_output, fingerprint(frozen, service.repository, args.manifest))
    output.mkdir(parents=True, exist_ok=True)
    write_predictions(output / "predictions.csv", predictions, threshold)
    write_json(
        output / "summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest": str(args.manifest.resolve()),
            "split": args.split,
            "model_hash": service.model_hash,
            "preprocess_version": config.preprocess_version,
            "gallery_version": service.repository.gallery_version(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "metrics": metrics,
            "protocol": before,
            "frozen": args.freeze is not None,
            "split_unit": sorted({item.split_unit for item in items}),
        },
    )
    print(f"阈值（复制完整精度到配置）: {threshold!r}")
    print(f"结果目录: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
