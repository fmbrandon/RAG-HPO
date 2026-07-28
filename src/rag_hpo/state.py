from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rag_hpo.models import AnnotationResult
from rag_hpo.privacy import ensure_private_directory, restrict_owner

STATE_SCHEMA_VERSION = "1.0"


class StateMismatchError(ValueError):
    pass


class PipelineState:
    def __init__(
        self,
        path: Path,
        *,
        input_sha256: str,
        artifact_sha256: str,
        pipeline_version: str,
        resume: bool,
    ) -> None:
        ensure_private_directory(path.parent)
        existed = path.exists()
        if existed and not resume:
            path.unlink()
        self.path = path
        self.connection = sqlite3.connect(path)
        restrict_owner(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rows (
                row_index INTEGER PRIMARY KEY,
                patient_id TEXT NOT NULL,
                note_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                results_json TEXT,
                error_code TEXT,
                error_message TEXT
            )
            """
        )
        expected = {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "input_sha256": input_sha256,
            "artifact_sha256": artifact_sha256,
            "pipeline_version": pipeline_version,
        }
        stored = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if stored:
            mismatches = {
                key: (stored.get(key), value)
                for key, value in expected.items()
                if stored.get(key) != value
            }
            if mismatches:
                self.connection.close()
                raise StateMismatchError(f"checkpoint metadata does not match: {mismatches}")
        else:
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                expected.items(),
            )
            self.connection.commit()

    def completed(self, row_index: int, note_sha256: str) -> list[AnnotationResult] | None:
        row = self.connection.execute(
            "SELECT note_sha256, status, results_json FROM rows WHERE row_index = ?",
            (row_index,),
        ).fetchone()
        if row is None:
            return None
        stored_hash, status, payload = row
        if stored_hash != note_sha256:
            raise StateMismatchError(f"note hash changed at row {row_index}")
        if status != "complete" or not payload:
            return None
        values = json.loads(payload)
        return [AnnotationResult.model_validate(value) for value in values]

    def save_complete(
        self,
        row_index: int,
        patient_id: str,
        note_sha256: str,
        results: list[AnnotationResult],
    ) -> None:
        payload = json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
        )
        self.connection.execute(
            """
            INSERT INTO rows(
                row_index, patient_id, note_sha256, status, results_json,
                error_code, error_message
            ) VALUES (?, ?, ?, 'complete', ?, NULL, NULL)
            ON CONFLICT(row_index) DO UPDATE SET
                patient_id=excluded.patient_id,
                note_sha256=excluded.note_sha256,
                status='complete',
                results_json=excluded.results_json,
                error_code=NULL,
                error_message=NULL
            """,
            (row_index, patient_id, note_sha256, payload),
        )
        self.connection.commit()

    def save_error(
        self,
        row_index: int,
        patient_id: str,
        note_sha256: str,
        error_code: str,
        error_message: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO rows(
                row_index, patient_id, note_sha256, status, results_json,
                error_code, error_message
            ) VALUES (?, ?, ?, 'error', NULL, ?, ?)
            ON CONFLICT(row_index) DO UPDATE SET
                patient_id=excluded.patient_id,
                note_sha256=excluded.note_sha256,
                status='error',
                results_json=NULL,
                error_code=excluded.error_code,
                error_message=excluded.error_message
            """,
            (row_index, patient_id, note_sha256, error_code, error_message),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> PipelineState:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
