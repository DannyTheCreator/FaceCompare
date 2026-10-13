from __future__ import annotations

import argparse
import csv
import math
import platform
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.recognizer import sha256_file
from app.service import FaceComparisonService
from experiments.common import write_json
from experiments.protocol import fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定视频 B0/B3/B4 队列调度对照")
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--opencv-threads", type=int, help="固定 OpenCV CPU 线程数")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/video_benchmark"))
    return parser.parse_args()


def _p95(values: list[float]) -> float | None:
    return None if not values else float(np.percentile(values, 95))


def run_variant(
    service: FaceComparisonService,
    video: Path,
    *,
    latest_only: bool,
    detect_scale: float,
    match_mode: str,
    max_seconds: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    # Warm the exact variant before starting the timed producer and then rewind.
    ok, warm_frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError("视频没有可解码帧")
    try:
        service.recognize(
            warm_frame, detect_scale=detect_scale, match_mode=match_mode, write_log=False
        )
    except Exception as exc:
        capture.release()
        raise RuntimeError("视频推理失败，预热未完成") from exc
    capture.release()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("预热后无法重新打开视频")
    task_queue: queue.Queue[tuple[int, float, np.ndarray, float]] = queue.Queue(
        maxsize=1 if latest_only else 0
    )
    producer_done = threading.Event()
    consumer_failed = threading.Event()
    errors: list[Exception] = []
    rows: list[dict[str, object]] = []
    dropped = 0

    def consumer() -> None:
        while not producer_done.is_set() or not task_queue.empty():
            try:
                frame_id, captured_at, image, queued_at = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                queue_ms = (time.perf_counter() - queued_at) * 1000
                result = service.recognize(
                    image,
                    frame_id=frame_id,
                    captured_monotonic=captured_at,
                    detect_scale=detect_scale,
                    match_mode=match_mode,
                    write_log=False,
                )
                rows.append(
                    {
                        "frame_id": frame_id,
                        "status": result.status,
                        "score": result.score,
                        "candidate_person_id": result.candidate_person_id,
                        "queue_wait_ms": queue_ms,
                        "detect_ms": result.timings_ms.get("detect"),
                        "align_and_feature_ms": result.timings_ms.get("align_and_feature"),
                        "match_ms": result.timings_ms.get("match"),
                        "algorithm_ms": result.timings_ms.get("algorithm_total"),
                        "end_to_end_ms": (time.perf_counter() - queued_at) * 1000,
                    }
                )
            except Exception as exc:
                errors.append(exc)
                consumer_failed.set()
                break
            finally:
                task_queue.task_done()

    thread = threading.Thread(target=consumer, name="video-inference", daemon=True)
    thread.start()
    started = time.perf_counter()
    next_frame_at = started
    submitted = 0
    frame_id = 0
    try:
        while time.perf_counter() - started < max_seconds and not consumer_failed.is_set():
            ok, frame = capture.read()
            if not ok:
                break
            now = time.perf_counter()
            if now < next_frame_at:
                time.sleep(next_frame_at - now)
            captured_at = time.perf_counter()
            frame_id += 1
            payload = (frame_id, captured_at, frame, time.perf_counter())
            if latest_only:
                try:
                    task_queue.put_nowait(payload)
                except queue.Full:
                    try:
                        task_queue.get_nowait()
                        task_queue.task_done()
                        dropped += 1
                    except queue.Empty:
                        pass
                    task_queue.put_nowait(payload)
            else:
                task_queue.put(payload)
            submitted += 1
            next_frame_at = started + frame_id / fps
    finally:
        capture.release()
        producer_done.set()
        thread.join()
    elapsed = time.perf_counter() - started
    if errors:
        raise RuntimeError("视频推理失败，停止输出不完整的性能结果") from errors[0]
    latencies = [float(row["end_to_end_ms"]) for row in rows]
    algorithms = [float(row["algorithm_ms"]) for row in rows if row["algorithm_ms"] is not None]
    summary = {
        "source_fps": fps,
        "submitted_frames": submitted,
        "processed_frames": len(rows),
        "dropped_frames": dropped,
        "drop_rate": dropped / submitted if submitted else None,
        "wall_seconds": elapsed,
        "result_fps": len(rows) / elapsed if elapsed else None,
        "algorithm_ms_mean": statistics.fmean(algorithms) if algorithms else None,
        "algorithm_ms_p50": statistics.median(algorithms) if algorithms else None,
        "algorithm_ms_p95": _p95(algorithms),
        "end_to_end_ms_mean": statistics.fmean(latencies) if latencies else None,
        "end_to_end_ms_p50": statistics.median(latencies) if latencies else None,
        "end_to_end_ms_p95": _p95(latencies),
    }
    for stage in ("queue_wait_ms", "detect_ms", "align_and_feature_ms", "match_ms"):
        values = [float(row[stage]) for row in rows if row[stage] is not None]
        summary[f"{stage}_p50"] = statistics.median(values) if values else None
        summary[f"{stage}_p95"] = _p95(values)
    return rows, summary


def main() -> int:
    args = parse_args()
    if args.repeats < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("repeats 和 max-seconds 必须为正数")
    config = load_config(args.config)
    if args.opencv_threads is not None:
        import cv2

        if args.opencv_threads < 1:
            raise ValueError("opencv-threads 必须为正数")
        cv2.setNumThreads(args.opencv_threads)
    service = FaceComparisonService(config, read_only=True)
    before = fingerprint(config, service.repository)
    if before["gallery_version"] != service._gallery_version:
        raise ValueError("建快照时标准库发生变化，请停止写入后重试")
    video_hash = sha256_file(args.video)
    variants = {
        "B0_queue_all": (False, 1.0, "loop"),
        "B3_latest_only": (True, 1.0, "loop"),
        "B4_combined": (True, config.detection.detect_scale, "matrix"),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for variant, (latest_only, scale, match_mode) in variants.items():
        for repeat in range(1, args.repeats + 1):
            rows, summary = run_variant(
                service,
                args.video.resolve(),
                latest_only=latest_only,
                detect_scale=scale,
                match_mode=match_mode,
                max_seconds=args.max_seconds,
            )
            all_rows.extend({"variant": variant, "repeat": repeat, **row} for row in rows)
            summaries.append({"variant": variant, "repeat": repeat, **summary})
            print(f"完成 {variant} repeat={repeat}")
    with (output / "video_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        if (
            fingerprint(config, service.repository) != before
            or sha256_file(args.video) != video_hash
        ):
            raise ValueError("视频性能测量期间输入或标准库变化")
        fields = [
            "variant",
            "repeat",
            "frame_id",
            "status",
            "score",
            "candidate_person_id",
            "queue_wait_ms",
            "detect_ms",
            "align_and_feature_ms",
            "match_ms",
            "algorithm_ms",
            "end_to_end_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (output / "video_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    write_json(
        output / "video_metadata.json",
        {
            "video": str(args.video.resolve()),
            "repeats": args.repeats,
            "max_seconds": args.max_seconds,
            "threshold": service.matcher.threshold,
            "model_hash": service.model_hash,
            "gallery_version": service.repository.gallery_version(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "variants": variants,
            "protocol": before,
            "video_sha256": video_hash,
        },
    )
    print(f"结果目录: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
