from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .image_io import write_image
from .matcher import FeatureRecord
from .recognizer import l2_normalize
from .types import PreparedSample, RecognitionResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class PersonRecord:
    person_id: int
    external_id: str | None
    name: str
    sample_count: int
    created_at: str


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    person_id: int
    image_path: Path
    session_id: str
    quality: dict
    created_at: str


class GalleryRepository:
    def __init__(self, database: Path, samples_dir: Path, *, read_only: bool = False):
        self.database = database
        self.samples_dir = samples_dir
        self.read_only = read_only
        if read_only:
            if not database.is_file():
                raise FileNotFoundError("实验标准库不存在，请先独立建库")
            return
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database.resolve().as_uri() + "?mode=ro" if self.read_only else self.database,
            timeout=10,
            uri=self.read_only,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if not self.read_only:
            connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT UNIQUE,
                    name TEXT NOT NULL CHECK(length(trim(name)) > 0),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    person_id INTEGER NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
                    image_path TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    feature BLOB NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    quality_json TEXT NOT NULL,
                    model_hash TEXT NOT NULL,
                    preprocess_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    frame_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    person_id INTEGER,
                    candidate_person_id INTEGER,
                    score REAL,
                    threshold REAL NOT NULL,
                    timings_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_file_deletions (
                    image_path TEXT PRIMARY KEY
                );
                INSERT OR IGNORE INTO metadata(key, value) VALUES ('gallery_version', '0');
                """
            )
            connection.commit()

    @staticmethod
    def _increment_version(connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE metadata SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
            "WHERE key = 'gallery_version'"
        )

    def gallery_version(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'gallery_version'"
            ).fetchone()
        return int(row["value"]) if row else 0

    def create_person(self, name: str, external_id: str | None = None) -> int:
        cleaned_name = name.strip()
        cleaned_external = external_id.strip() if external_id and external_id.strip() else None
        if not cleaned_name:
            raise ValueError("姓名不能为空")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO persons(external_id, name, created_at) VALUES (?, ?, ?)",
                (cleaned_external, cleaned_name, _utc_now()),
            )
            self._increment_version(connection)
            connection.commit()
            return int(cursor.lastrowid)

    def list_persons(self) -> list[PersonRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.person_id, p.external_id, p.name, p.created_at,
                       COUNT(s.sample_id) AS sample_count
                FROM persons p LEFT JOIN samples s ON s.person_id = p.person_id
                GROUP BY p.person_id ORDER BY p.person_id
                """
            ).fetchall()
        return [
            PersonRecord(
                int(row["person_id"]),
                row["external_id"],
                str(row["name"]),
                int(row["sample_count"]),
                str(row["created_at"]),
            )
            for row in rows
        ]

    def list_samples(self, person_id: int | None = None) -> list[SampleRecord]:
        query = (
            "SELECT sample_id, person_id, image_path, session_id, quality_json, created_at "
            "FROM samples"
        )
        params: tuple[int, ...] = ()
        if person_id is not None:
            query += " WHERE person_id = ?"
            params = (int(person_id),)
        query += " ORDER BY created_at"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            SampleRecord(
                sample_id=str(row["sample_id"]),
                person_id=int(row["person_id"]),
                image_path=(self.database.parent / str(row["image_path"])).resolve(),
                session_id=str(row["session_id"]),
                quality=json.loads(str(row["quality_json"])),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def add_samples(
        self,
        person_id: int,
        samples: Sequence[PreparedSample],
        session_id: str,
        model_hash: str,
        preprocess_version: str,
    ) -> list[str]:
        if not samples:
            raise ValueError("没有可保存的合格样本")
        if self.read_only:
            raise ValueError("实验标准库为只读")
        if not session_id.strip() or not model_hash.strip() or not preprocess_version.strip():
            raise ValueError("会话和特征版本不能为空")
        person_dir = self.samples_dir / str(int(person_id))
        person_dir.mkdir(parents=True, exist_ok=True)
        prepared_files: list[tuple[str, Path, Path, PreparedSample]] = []
        try:
            for sample in samples:
                sample_id = uuid.uuid4().hex
                final_path = person_dir / f"{sample_id}.jpg"
                temporary_path = person_dir / f".{sample_id}.tmp.jpg"
                prepared_files.append((sample_id, temporary_path, final_path, sample))
                l2_normalize(sample.feature)  # Validate before writing files or database rows.
                if not sample.quality.accepted:
                    raise ValueError("不能保存未通过质量检查的样本")
                write_image(temporary_path, sample.image)

            with self._connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM persons WHERE person_id = ?", (int(person_id),)
                ).fetchone()
                if not exists:
                    raise ValueError(f"人员不存在: {person_id}")
                dimensions = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT feature_dim FROM samples WHERE model_hash = ? AND preprocess_version = ?",
                        (model_hash, preprocess_version),
                    )
                }
                dimensions.update(int(np.asarray(sample.feature).size) for sample in samples)
                if len(dimensions) != 1:
                    raise ValueError("样本特征维度与当前模型标准库不一致")
                try:
                    for sample_id, temporary_path, final_path, sample in prepared_files:
                        os.replace(temporary_path, final_path)
                        relative_path = os.path.relpath(final_path, self.database.parent)
                        feature = l2_normalize(sample.feature)
                        connection.execute(
                            """
                            INSERT INTO samples(
                                sample_id, person_id, image_path, session_id, feature,
                                feature_dim, quality_json, model_hash, preprocess_version,
                                created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                sample_id,
                                int(person_id),
                                relative_path,
                                session_id,
                                feature.tobytes(),
                                int(feature.size),
                                json.dumps(sample.quality.as_dict(), ensure_ascii=False),
                                model_hash,
                                preprocess_version,
                                _utc_now(),
                            ),
                        )
                    self._increment_version(connection)
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
        except Exception:
            for _, temporary_path, final_path, _ in prepared_files:
                temporary_path.unlink(missing_ok=True)
                final_path.unlink(missing_ok=True)
            raise
        return [item[0] for item in prepared_files]

    def load_features(self, model_hash: str, preprocess_version: str) -> list[FeatureRecord]:
        return self.load_feature_snapshot(model_hash, preprocess_version)[0]

    def load_feature_snapshot(
        self, model_hash: str, preprocess_version: str
    ) -> tuple[list[FeatureRecord], int]:
        with self._connect() as connection:
            connection.execute("BEGIN")
            version = int(
                connection.execute(
                    "SELECT value FROM metadata WHERE key='gallery_version'"
                ).fetchone()[0]
            )
            rows = connection.execute(
                """
                SELECT s.person_id, p.name, s.feature, s.feature_dim
                FROM samples s JOIN persons p ON p.person_id = s.person_id
                WHERE s.model_hash = ? AND s.preprocess_version = ?
                ORDER BY s.person_id, s.created_at
                """,
                (model_hash, preprocess_version),
            ).fetchall()
        records: list[FeatureRecord] = []
        for row in rows:
            feature = np.frombuffer(row["feature"], dtype=np.float32).copy()
            if feature.size != int(row["feature_dim"]):
                raise ValueError(f"样本特征损坏，person_id={row['person_id']}")
            records.append(FeatureRecord(int(row["person_id"]), str(row["name"]), feature))
        return records, version

    def delete_sample(self, sample_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT image_path FROM samples WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"样本不存在: {sample_id}")
            self._sample_path(str(row["image_path"]))
            connection.execute(
                "INSERT OR IGNORE INTO pending_file_deletions VALUES (?)", (row["image_path"],)
            )
            connection.execute("DELETE FROM samples WHERE sample_id = ?", (sample_id,))
            self._increment_version(connection)
            connection.commit()
        self.cleanup_pending()

    def delete_person(self, person_id: int) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT image_path FROM samples WHERE person_id = ?", (int(person_id),)
            ).fetchall()
            for row in rows:
                self._sample_path(str(row["image_path"]))
                connection.execute(
                    "INSERT OR IGNORE INTO pending_file_deletions VALUES (?)", (row["image_path"],)
                )
            cursor = connection.execute(
                "DELETE FROM persons WHERE person_id = ?", (int(person_id),)
            )
            if cursor.rowcount == 0:
                raise ValueError(f"人员不存在: {person_id}")
            self._increment_version(connection)
            connection.commit()
        self.cleanup_pending()
        person_dir = self.samples_dir / str(int(person_id))
        try:
            person_dir.rmdir()
        except OSError:
            pass

    def _sample_path(self, relative: str) -> Path:
        path = (self.database.parent / relative).resolve()
        if (
            not path.is_relative_to(self.samples_dir.resolve())
            or path == self.samples_dir.resolve()
        ):
            raise ValueError("图片路径超出配置的样本目录")
        return path

    def cleanup_pending(self) -> int:
        """Retry only committed deletion records. Never remove arbitrary orphan files."""
        if self.read_only:
            raise ValueError("实验标准库为只读")
        with self._connect() as connection:
            rows = connection.execute("SELECT image_path FROM pending_file_deletions").fetchall()
            for row in rows:
                try:
                    self._sample_path(str(row["image_path"])).unlink(missing_ok=True)
                except (OSError, ValueError):
                    continue
                connection.execute(
                    "DELETE FROM pending_file_deletions WHERE image_path=?", (row["image_path"],)
                )
            connection.commit()
            return int(
                connection.execute("SELECT COUNT(*) FROM pending_file_deletions").fetchone()[0]
            )

    def integrity_report(self, model_hash: str, preprocess_version: str) -> dict:
        """Counts only: suitable for UI, without exposing names or embeddings."""
        report = {
            "missing_images": 0,
            "unsafe_paths": 0,
            "invalid_features": 0,
            "incompatible_samples": 0,
            "orphan_images": 0,
            "pending_deletions": 0,
        }
        referenced = set()
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM samples").fetchall()
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            pending = (
                list(connection.execute("SELECT image_path FROM pending_file_deletions"))
                if "pending_file_deletions" in tables
                else []
            )
        report["pending_deletions"] = len(pending)
        for row in rows:
            try:
                path = self._sample_path(row["image_path"])
                referenced.add(path)
                report["missing_images"] += int(not path.is_file())
            except ValueError:
                report["unsafe_paths"] += 1
            report["incompatible_samples"] += int(
                row["model_hash"] != model_hash or row["preprocess_version"] != preprocess_version
            )
            try:
                vector = np.frombuffer(row["feature"], dtype=np.float32)
                if vector.size != row["feature_dim"]:
                    raise ValueError("dimension")
                l2_normalize(vector)
            except ValueError:
                report["invalid_features"] += 1
        for row in pending:
            try:
                referenced.add(self._sample_path(row["image_path"]))
            except ValueError:
                report["unsafe_paths"] += 1
        report["orphan_images"] = sum(
            path.resolve() not in referenced for path in self.samples_dir.rglob("*.jpg")
        )
        return report

    def log_recognition(self, result: RecognitionResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recognition_logs(
                    created_at, frame_id, status, person_id, candidate_person_id,
                    score, threshold, timings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    int(result.frame_id),
                    result.status,
                    result.person_id,
                    result.candidate_person_id,
                    result.score,
                    result.threshold,
                    json.dumps(result.timings_ms, ensure_ascii=False),
                ),
            )
            connection.commit()
