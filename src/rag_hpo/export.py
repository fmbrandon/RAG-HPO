from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from rag_hpo.models import AnnotationResult
from rag_hpo.privacy import ensure_private_directory, restrict_owner

FIELDS = [
    "patient_id",
    "phrase",
    "category",
    "hpo_id",
    "hpo_term",
    "vector_score",
    "mapping_status",
    "error_code",
    "error_message",
    "evidence_start",
    "evidence_end",
    "assertion_status",
    "confidence",
    "category_confidence",
    "review_status",
    "source_methods",
    "candidate_hpo_ids",
    "retrieval_candidate_hpo_ids",
    "modifier_hpo_ids",
    "evidence_segments",
    "mapping_verdict",
    "phenotype_confidence",
    "mapping_set_confidence",
    "overall_confidence",
    "confidence_basis",
    "evidence_text",
]


def _replace(path: Path, content: str) -> None:
    ensure_private_directory(path.parent)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        restrict_owner(temporary_path)
        try:
            os.replace(temporary_path, path)
        except PermissionError as exc:
            msg = (
                f"Permission denied when writing to '{path}'. If the output file "
                "is open in Excel or another program, please close it and try again."
            )
            raise PermissionError(msg) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def export_results(results: list[AnnotationResult], output_dir: Path) -> tuple[Path, Path]:
    ensure_private_directory(output_dir)
    csv_path = output_dir / "rag_hpo_results.csv"
    json_path = output_dir / "rag_hpo_results.json"

    rows = [result.model_dump(mode="json") for result in results]
    csv_rows = [
        {
            **row,
            "source_methods": "|".join(row["source_methods"] or []),
            "candidate_hpo_ids": "|".join(row["candidate_hpo_ids"] or []),
            "retrieval_candidate_hpo_ids": "|".join(row["retrieval_candidate_hpo_ids"] or []),
            "modifier_hpo_ids": "|".join(row["modifier_hpo_ids"] or []),
            "evidence_segments": "|".join(
                f"{start}:{end}" for start, end in (row["evidence_segments"] or [])
            ),
        }
        for row in rows
    ]
    buffer = __import__("io").StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(csv_rows)
    _replace(csv_path, buffer.getvalue())
    _replace(json_path, json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return csv_path, json_path
