from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from voice_service.domain import AudioMetrics, JobStatus, NarrationJob


class NarrationJobRepository:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS narration_jobs (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    language TEXT NOT NULL,
                    voice_profile TEXT NOT NULL,
                    generation_profile TEXT NOT NULL,
                    resolved_seed INTEGER NOT NULL,
                    requested_seed INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    current_chunk INTEGER NOT NULL DEFAULT 0,
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    model_id TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    voice_identity TEXT NOT NULL,
                    generation_identity TEXT NOT NULL,
                    chunking_identity TEXT,
                    request_fingerprint TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE,
                    output_uri TEXT,
                    result_duration_seconds REAL,
                    result_sample_rate INTEGER,
                    result_channels INTEGER,
                    result_subtype TEXT,
                    result_byte_size INTEGER,
                    result_sha256 TEXT,
                    result_frame_count INTEGER,
                    result_riff_data_byte_size INTEGER,
                    sample_peak_dbfs REAL,
                    clipping_sample_count INTEGER,
                    error_category TEXT,
                    error_message TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS ix_narration_jobs_status_created
                    ON narration_jobs(status, created_at);
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(narration_jobs)")
            }
            if "chunking_identity" not in columns:
                connection.execute("ALTER TABLE narration_jobs ADD COLUMN chunking_identity TEXT")
            if "result_frame_count" not in columns:
                connection.execute(
                    "ALTER TABLE narration_jobs ADD COLUMN result_frame_count INTEGER"
                )
            if "result_riff_data_byte_size" not in columns:
                connection.execute(
                    "ALTER TABLE narration_jobs ADD COLUMN result_riff_data_byte_size INTEGER"
                )

    def create(self, job: NarrationJob) -> None:
        values = self._values(job)
        columns = ", ".join(values)
        placeholders = ", ".join(f":{column}" for column in values)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"INSERT INTO narration_jobs ({columns}) VALUES ({placeholders})",
                values,
            )

    def get(self, job_id: str) -> NarrationJob | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM narration_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def get_by_idempotency_key(self, key: str) -> NarrationJob | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM narration_jobs WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def mark_running(self, job_id: str, *, started_at: str, total_chunks: int) -> None:
        self._update(
            job_id,
            status=JobStatus.RUNNING.value,
            started_at=started_at,
            completed_at=None,
            current_chunk=0,
            total_chunks=total_chunks,
            error_category=None,
            error_message=None,
        )

    def update_progress(self, job_id: str, *, current_chunk: int) -> None:
        self._update(job_id, current_chunk=current_chunk)

    def mark_completed(
        self,
        job_id: str,
        *,
        completed_at: str,
        output_uri: str,
        metrics: AudioMetrics,
        metadata: dict[str, object],
    ) -> None:
        self._update(
            job_id,
            status=JobStatus.COMPLETED.value,
            completed_at=completed_at,
            output_uri=output_uri,
            result_duration_seconds=metrics.duration_seconds,
            result_sample_rate=metrics.sample_rate,
            result_channels=metrics.channels,
            result_subtype=metrics.subtype,
            result_byte_size=metrics.byte_size,
            result_sha256=metrics.sha256,
            result_frame_count=metrics.frame_count,
            result_riff_data_byte_size=metrics.riff_data_byte_size,
            sample_peak_dbfs=metrics.sample_peak_dbfs,
            clipping_sample_count=metrics.clipping_sample_count,
            error_category=None,
            error_message=None,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )

    def mark_failed(
        self,
        job_id: str,
        *,
        completed_at: str,
        category: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        values: dict[str, object] = {
            "status": JobStatus.FAILED.value,
            "completed_at": completed_at,
            "error_category": category,
            "error_message": message,
        }
        if metadata is not None:
            values["metadata_json"] = json.dumps(metadata, sort_keys=True)
        self._update(
            job_id,
            **values,
        )

    def requeue(self, job_id: str) -> None:
        self._update(
            job_id,
            status=JobStatus.QUEUED.value,
            started_at=None,
            completed_at=None,
            current_chunk=0,
            error_category=None,
            error_message=None,
        )

    def recoverable(self) -> Sequence[NarrationJob]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM narration_jobs
                   WHERE status IN ('queued','running')
                   ORDER BY created_at, id"""
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def prune_terminal_before(self, cutoff: str) -> list[str]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT id FROM narration_jobs
                   WHERE status IN ('completed','failed')
                     AND completed_at IS NOT NULL AND completed_at < ?""",
                (cutoff,),
            ).fetchall()
            ids = [str(row["id"]) for row in rows]
            connection.executemany(
                "DELETE FROM narration_jobs WHERE id = ?", [(job_id,) for job_id in ids]
            )
        return ids

    def delete(self, job_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM narration_jobs WHERE id = ?", (job_id,))

    def count_active(self) -> int:
        with self._lock, self._connect() as connection:
            value = connection.execute(
                "SELECT COUNT(*) FROM narration_jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
        return int(value)

    def _update(self, job_id: str, **values: object) -> None:
        assignments = ", ".join(f"{column} = :{column}" for column in values)
        parameters = {**values, "id": job_id}
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE narration_jobs SET {assignments} WHERE id = :id", parameters
            )
            if cursor.rowcount != 1:
                raise LookupError("narration job not found")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _values(job: NarrationJob) -> dict[str, object]:
        return {
            "id": job.id,
            "text": job.text,
            "language": job.language,
            "voice_profile": job.voice_profile,
            "generation_profile": job.generation_profile,
            "resolved_seed": job.resolved_seed,
            "requested_seed": job.requested_seed,
            "status": job.status.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "current_chunk": job.current_chunk,
            "total_chunks": job.total_chunks,
            "model_id": job.model_id,
            "model_revision": job.model_revision,
            "voice_identity": job.voice_identity,
            "generation_identity": job.generation_identity,
            "chunking_identity": job.chunking_identity,
            "request_fingerprint": job.request_fingerprint,
            "idempotency_key": job.idempotency_key,
            "output_uri": job.output_uri,
            "result_duration_seconds": job.result_duration_seconds,
            "result_sample_rate": job.result_sample_rate,
            "result_channels": job.result_channels,
            "result_subtype": job.result_subtype,
            "result_byte_size": job.result_byte_size,
            "result_sha256": job.result_sha256,
            "result_frame_count": job.result_frame_count,
            "result_riff_data_byte_size": job.result_riff_data_byte_size,
            "sample_peak_dbfs": job.sample_peak_dbfs,
            "clipping_sample_count": job.clipping_sample_count,
            "error_category": job.error_category,
            "error_message": job.error_message,
            "metadata_json": json.dumps(job.metadata, sort_keys=True),
        }

    @staticmethod
    def _from_row(row: sqlite3.Row) -> NarrationJob:
        return NarrationJob(
            id=row["id"],
            text=row["text"],
            language=row["language"],
            voice_profile=row["voice_profile"],
            generation_profile=row["generation_profile"],
            resolved_seed=row["resolved_seed"],
            requested_seed=row["requested_seed"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            current_chunk=row["current_chunk"],
            total_chunks=row["total_chunks"],
            model_id=row["model_id"],
            model_revision=row["model_revision"],
            voice_identity=row["voice_identity"],
            generation_identity=row["generation_identity"],
            chunking_identity=row["chunking_identity"] or "",
            request_fingerprint=row["request_fingerprint"],
            idempotency_key=row["idempotency_key"],
            output_uri=row["output_uri"],
            result_duration_seconds=row["result_duration_seconds"],
            result_sample_rate=row["result_sample_rate"],
            result_channels=row["result_channels"],
            result_subtype=row["result_subtype"],
            result_byte_size=row["result_byte_size"],
            result_sha256=row["result_sha256"],
            result_frame_count=row["result_frame_count"],
            result_riff_data_byte_size=row["result_riff_data_byte_size"],
            sample_peak_dbfs=row["sample_peak_dbfs"],
            clipping_sample_count=row["clipping_sample_count"],
            error_category=row["error_category"],
            error_message=row["error_message"],
            metadata=json.loads(row["metadata_json"]),
        )
