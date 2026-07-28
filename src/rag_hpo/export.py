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
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def export_results(results: list[AnnotationResult], output_dir: Path) -> tuple[Path, Path]:
    ensure_private_directory(output_dir)
    csv_path = output_dir / "rag_hpo_results.csv"
    json_path = output_dir / "rag_hpo_results.json"

    rows = [result.model_dump(mode="json") for result in results]
    buffer = __import__("io").StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    _replace(csv_path, buffer.getvalue())
    _replace(json_path, json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return csv_path, json_path
