from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock
from typing import Iterable

import numpy as np

from .recognizer import l2_normalize


@dataclass(frozen=True)
class FeatureRecord:
    person_id: int
    person_name: str
    feature: np.ndarray


@dataclass(frozen=True)
class Candidate:
    person_id: int
    person_name: str
    score: float


@dataclass(frozen=True)
class GallerySnapshot:
    person_ids: tuple[int, ...]
    person_names: tuple[str, ...]
    templates: np.ndarray
    sample_counts: tuple[int, ...]

    @classmethod
    def empty(cls) -> "GallerySnapshot":
        matrix = np.empty((0, 0), dtype=np.float32)
        matrix.setflags(write=False)
        return cls((), (), matrix, ())

    @classmethod
    def from_records(cls, records: Iterable[FeatureRecord]) -> "GallerySnapshot":
        grouped: dict[int, tuple[str, list[np.ndarray]]] = {}
        for record in records:
            feature = l2_normalize(record.feature)
            if record.person_id not in grouped:
                grouped[record.person_id] = (record.person_name, [])
            grouped[record.person_id][1].append(feature)
        if not grouped:
            return cls.empty()

        person_ids: list[int] = []
        names: list[str] = []
        templates: list[np.ndarray] = []
        counts: list[int] = []
        expected_dim: int | None = None
        for person_id in sorted(grouped):
            name, features = grouped[person_id]
            dims = {item.size for item in features}
            if len(dims) != 1:
                raise ValueError(f"人员 {person_id} 的特征维度不一致")
            dim = dims.pop()
            if expected_dim is None:
                expected_dim = dim
            elif dim != expected_dim:
                raise ValueError("标准库中存在不同维度的特征")
            template = l2_normalize(np.mean(np.stack(features), axis=0))
            person_ids.append(person_id)
            names.append(name)
            templates.append(template)
            counts.append(len(features))
        matrix = np.ascontiguousarray(np.stack(templates), dtype=np.float32)
        matrix.setflags(write=False)
        return cls(tuple(person_ids), tuple(names), matrix, tuple(counts))


class FaceMatcher:
    """Thread-safe immutable gallery snapshots with matrix cosine matching."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("相似度阈值必须在 [-1, 1] 内")
        self._snapshot = GallerySnapshot.empty()
        self._lock = RLock()

    @property
    def snapshot(self) -> GallerySnapshot:
        with self._lock:
            return self._snapshot

    def publish(self, snapshot: GallerySnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def top_match(self, feature: np.ndarray, mode: str = "matrix") -> Candidate | None:
        ranked = self.rank(feature, k=1, mode=mode)
        return ranked[0] if ranked else None

    def rank(self, feature: np.ndarray, k: int = 3, mode: str = "matrix") -> tuple[Candidate, ...]:
        """Return deterministic identity-level candidates without accepting any identity."""
        if type(k) is not int or k < 1:
            raise ValueError("候选数量必须为正整数")
        query = l2_normalize(feature)
        snapshot = self.snapshot
        if not snapshot.person_ids:
            return ()
        if query.size != snapshot.templates.shape[1]:
            raise ValueError("查询特征维度与标准库不一致")
        if mode == "matrix":
            scores = snapshot.templates @ query
        elif mode == "loop":
            scores = np.asarray([float(np.dot(row, query)) for row in snapshot.templates])
        else:
            raise ValueError(f"未知匹配模式: {mode}")
        if scores.size > max(64, 4 * k):
            # Partition scores in O(N), retaining the original identity order at ties.
            cutoff = np.partition(scores, scores.size - k)[-k]
            above = np.flatnonzero(scores > cutoff)
            tied = np.flatnonzero(scores == cutoff)[: k - above.size]
            selected = np.concatenate((above, tied))
            indices = selected[np.lexsort((selected, -scores[selected]))]
        else:
            indices = np.argsort(-scores, kind="stable")[:k]
        return tuple(
            Candidate(
                snapshot.person_ids[int(index)],
                snapshot.person_names[int(index)],
                float(np.clip(scores[int(index)], -1.0, 1.0)),
            )
            for index in indices
        )
