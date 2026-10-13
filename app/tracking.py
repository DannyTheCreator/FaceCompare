"""Small-scene IoU association and consecutive identity confirmation.

This is an original lightweight baseline, not an implementation of ByteTrack.
Track IDs are camera-session labels, not verified biometric identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .config import TrackingConfig
from .types import SceneResult


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left[:4], dtype=float), np.asarray(right[:4], dtype=float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or min(a[2], a[3], b[2], b[3]) <= 0:
        return 0.0
    overlap = np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    return intersection / (float(a[2] * a[3] + b[2] * b[3]) - intersection)


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    last_seen: float
    candidate_id: int | None = None
    hits: int = 0


class SceneTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self._context: tuple[int, int, int] | None = None
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_scene: SceneResult | None = None

    def reset(self) -> None:
        self._context = None
        self._tracks.clear()
        self._next_id = 1
        self._last_scene = None

    def update(self, scene: SceneResult) -> SceneResult:
        context = (scene.stream_id, scene.gallery_version, scene.mode_generation)
        if self._context != context:
            self.reset()
            self._context = context
        now = scene.captured_monotonic
        if not math.isfinite(now):
            raise ValueError("帧时间戳必须有限")
        if self._last_scene is not None:
            if now == self._last_scene.captured_monotonic:
                return self._last_scene  # A repeated snapshot is not new temporal evidence.
            if now < self._last_scene.captured_monotonic:
                raise ValueError("跟踪输入必须按采集时间递增")
        self._tracks = {
            key: track
            for key, track in self._tracks.items()
            if now - track.last_seen <= self.config.max_gap_ms / 1000
        }
        # Greedy global IoU ordering, with at most one detection per track.
        pairs = sorted(
            (
                (box_iou(track.box, face.face), track_id, index)
                for track_id, track in self._tracks.items()
                for index, face in enumerate(scene.faces)
                if face.face is not None
            ),
            key=lambda pair: (-pair[0], pair[1], pair[2]),
        )
        assignments: dict[int, int] = {}
        used: set[int] = set()
        for overlap, track_id, index in pairs:
            if (
                overlap >= self.config.iou_threshold
                and track_id not in used
                and index not in assignments
            ):
                assignments[index] = track_id
                used.add(track_id)
        for track_id, track in self._tracks.items():
            if track_id not in used:
                track.candidate_id, track.hits = None, 0
        output = []
        for index, face in enumerate(scene.faces):
            if face.face is None:
                continue
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id, face.face[:4].copy(), now)
            track = self._tracks[track_id]
            track.box, track.last_seen = face.face[:4].copy(), now
            if face.status == "matched" and face.person_id is not None:
                track.hits = (
                    min(self.config.confirmation_frames, track.hits + 1)
                    if track.candidate_id == face.person_id
                    else 1
                )
                track.candidate_id = face.person_id
            else:
                track.candidate_id, track.hits = None, 0
            confirmed = face.status == "matched" and track.hits >= self.config.confirmation_frames
            status = "pending" if face.status == "matched" and not confirmed else face.status
            output.append(
                replace(
                    face,
                    track_id=track_id,
                    confirmation_hits=track.hits,
                    confirmed=confirmed,
                    status=status,
                    person_id=face.person_id if confirmed else None,
                    person_name=face.person_name if confirmed else None,
                    message=face.message
                    if status != "pending"
                    else f"等待连续确认 {track.hits}/{self.config.confirmation_frames}",
                )
            )
        self._last_scene = replace(scene, faces=tuple(output))
        return self._last_scene
