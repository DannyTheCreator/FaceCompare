from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from app.recognizer import sha256_file


@dataclass(frozen=True)
class ManifestItem:
    sample_id: str
    image_path: Path
    split: str
    is_known: bool
    expected_person_id: int | None
    scene: str
    session_id: str
    split_unit: str = "session"


@dataclass(frozen=True)
class RawPrediction:
    sample_id: str
    image_path: str
    split: str
    is_known: bool
    expected_person_id: int | None
    scene: str
    session_id: str
    processing_status: str
    candidate_person_id: int | None
    candidate_name: str | None
    score: float | None
    detect_ms: float | None
    quality_ms: float | None
    align_and_feature_ms: float | None
    match_ms: float | None
    algorithm_total_ms: float | None


TRUE_VALUES = {"1", "true", "yes", "y", "known", "是"}
FALSE_VALUES = {"0", "false", "no", "n", "unknown", "否"}


def _parse_bool(value: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"清单第 {row_number} 行 is_known 无法解析: {value!r}")


def load_manifest(path: Path, split: str | None = None) -> list[ManifestItem]:
    """Load a CSV manifest and resolve image paths from the manifest directory."""

    path = path.resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "split", "is_known", "session_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"数据清单缺少列: {', '.join(sorted(missing))}")
        items: list[ManifestItem] = []
        sample_ids: set[str] = set()
        image_splits: dict[Path, str] = {}
        session_splits: dict[str, str] = {}
        content_splits: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            row_split = (row.get("split") or "").strip()
            image_value = (row.get("image_path") or "").strip()
            session_id = (row.get("session_id") or "").strip()
            split_unit = (row.get("split_unit") or "session").strip()
            if split_unit not in {"session", "image"}:
                raise ValueError("split_unit 只允许 session 或 image")
            if not image_value or not row_split or not session_id:
                raise ValueError(f"清单第 {row_number} 行的图像、split 和 session_id 不能为空")
            if row_split not in {"validation", "test"}:
                raise ValueError("split 只允许 validation 或 test")
            raw_image = Path(image_value)
            image_path = raw_image if raw_image.is_absolute() else path.parent / raw_image
            image_path = image_path.resolve()
            if image_path.is_file():
                digest = sha256_file(image_path)
                if digest in content_splits:
                    raise ValueError(f"图像内容重复: 第 {row_number} 行")
                content_splits[digest] = row_split
            sample_id = (row.get("sample_id") or f"row-{row_number}").strip()
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"清单第 {row_number} 行 sample_id 为空或重复")
            sample_ids.add(sample_id)
            if image_path in image_splits and image_splits[image_path] != row_split:
                raise ValueError(f"图像跨 split 重复: 第 {row_number} 行")
            if (
                split_unit == "session"
                and session_id in session_splits
                and session_splits[session_id] != row_split
            ):
                raise ValueError(f"拍摄会话跨 split 重复: {session_id}")
            image_splits[image_path] = row_split
            if split_unit == "session":
                session_splits[session_id] = row_split
            expected_raw = (row.get("expected_person_id") or "").strip()
            is_known = _parse_bool(row.get("is_known") or "", row_number)
            if is_known and not expected_raw:
                raise ValueError(f"清单第 {row_number} 行为已知人员但缺少 expected_person_id")
            if not is_known and expected_raw:
                raise ValueError(f"清单第 {row_number} 行为未知人员但填写了 expected_person_id")
            if expected_raw and int(expected_raw) <= 0:
                raise ValueError(f"清单第 {row_number} 行 expected_person_id 必须为正整数")
            if split and row_split != split:
                continue
            items.append(
                ManifestItem(
                    sample_id=sample_id,
                    image_path=image_path.resolve(),
                    split=row_split,
                    is_known=is_known,
                    expected_person_id=int(expected_raw) if expected_raw else None,
                    scene=(row.get("scene") or "unspecified").strip(),
                    session_id=session_id,
                    split_unit=split_unit,
                )
            )
    if not items:
        raise ValueError("数据清单没有符合筛选条件的样本")
    return items


def validate_gallery_isolation(items, person_ids, registered_samples) -> None:
    """Reject absent known identities and enrollment/test path or session overlap."""
    ids = set(person_ids)
    enrolled_paths = {sample.image_path.resolve() for sample in registered_samples}
    enrolled_sessions = {sample.session_id for sample in registered_samples}
    enrolled_hashes = {sha256_file(path) for path in enrolled_paths if path.is_file()}
    for item in items:
        if item.is_known and item.expected_person_id not in ids:
            raise ValueError(f"已知人员不在兼容标准库中: {item.sample_id}")
        if item.image_path.resolve() in enrolled_paths or (
            item.split_unit == "session" and item.session_id in enrolled_sessions
        ):
            raise ValueError(f"评估样本与注册数据重叠: {item.sample_id}")
        if item.image_path.is_file() and sha256_file(item.image_path) in enrolled_hashes:
            raise ValueError(f"评估样本与注册图像内容重复: {item.sample_id}")


def _valid_query(prediction: RawPrediction) -> bool:
    return (
        prediction.processing_status in {"matched", "unknown"}
        and prediction.candidate_person_id is not None
        and prediction.score is not None
        and math.isfinite(prediction.score)
    )


def classify(prediction: RawPrediction, threshold: float) -> str:
    if not _valid_query(prediction):
        return "processing_failure"
    accepted = prediction.score >= threshold
    if prediction.is_known:
        if not accepted:
            return "known_rejected"
        if prediction.candidate_person_id == prediction.expected_person_id:
            return "known_correct"
        return "known_wrong_identity"
    return "unknown_false_accept" if accepted else "unknown_correct_reject"


def calculate_metrics(predictions: Iterable[RawPrediction], threshold: float) -> dict[str, object]:
    rows = list(predictions)
    counts: dict[str, int] = {}
    for row in rows:
        outcome = classify(row, threshold)
        counts[outcome] = counts.get(outcome, 0) + 1
    known = [row for row in rows if row.is_known]
    unknown = [row for row in rows if not row.is_known]
    valid_unknown = [row for row in unknown if _valid_query(row)]
    processing_status_counts: dict[str, int] = {}
    for row in rows:
        processing_status_counts[row.processing_status] = (
            processing_status_counts.get(row.processing_status, 0) + 1
        )

    def ratio(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    known_correct = counts.get("known_correct", 0)
    known_rejected = counts.get("known_rejected", 0)
    known_wrong = counts.get("known_wrong_identity", 0)
    unknown_false = counts.get("unknown_false_accept", 0)
    unknown_rejected = counts.get("unknown_correct_reject", 0)
    return {
        "threshold": threshold,
        "attempts": len(rows),
        "known_attempts": len(known),
        "unknown_attempts": len(unknown),
        "valid_unknown_queries": len(valid_unknown),
        "counts": counts,
        "processing_status_counts": processing_status_counts,
        "known_end_to_end_recognition_rate": ratio(known_correct, len(known)),
        "known_rejection_rate": ratio(known_rejected, len(known)),
        "known_wrong_identity_rate": ratio(known_wrong, len(known)),
        "unknown_false_accept_rate_all_attempts": ratio(unknown_false, len(unknown)),
        "unknown_false_accept_rate_valid_queries": ratio(unknown_false, len(valid_unknown)),
        "unknown_correct_reject_rate_all_attempts": ratio(unknown_rejected, len(unknown)),
    }


def choose_threshold(
    predictions: Iterable[RawPrediction], target_unknown_far: float
) -> tuple[float, dict[str, object]]:
    rows = list(predictions)
    if not 0 <= target_unknown_far <= 1:
        raise ValueError("目标未知误接收率必须在 [0, 1] 范围内")
    unknown_scores = [row.score for row in rows if not row.is_known and _valid_query(row)]
    if not unknown_scores:
        raise ValueError("验证集中没有可用的未知人员查询，无法标定阈值")
    if not any(row.is_known and _valid_query(row) for row in rows):
        raise ValueError("验证集中没有可用的已知人员查询，无法优化识别率")
    all_scores = [row.score for row in rows if _valid_query(row)]
    candidates = {-1.0, 1.0}
    for score in all_scores:
        assert score is not None
        candidates.add(float(score))
        if score < 1.0:
            candidates.add(min(1.0, float(score) + 1e-7))
    feasible: list[tuple[float, float, dict[str, object]]] = []
    for threshold in sorted(candidates):
        metrics = calculate_metrics(rows, threshold)
        far = metrics["unknown_false_accept_rate_valid_queries"]
        if far is not None and float(far) <= target_unknown_far:
            known_rate = metrics["known_end_to_end_recognition_rate"]
            feasible.append((float(known_rate or 0.0), threshold, metrics))
    if not feasible:
        raise ValueError("当前验证分数无法满足目标未知误接收率")
    # Maximize known end-to-end recognition, then prefer the lowest feasible threshold.
    best_rate = max(item[0] for item in feasible)
    best = min((item for item in feasible if item[0] == best_rate), key=lambda x: x[1])
    return best[1], best[2]


def write_predictions(path: Path, predictions: Iterable[RawPrediction], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(predictions)
    fieldnames = list(RawPrediction.__dataclass_fields__) + ["decision"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["decision"] = classify(row, threshold)
            writer.writerow(payload)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
