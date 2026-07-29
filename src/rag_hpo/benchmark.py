from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file

HPO_ID = re.compile(r"HP:\d{7}")


@dataclass(frozen=True)
class CaseScore:
    patient_id: str
    predicted_ids: list[str]
    reference_ids: list[str]
    true_positive_ids: list[str]
    false_positive_ids: list[str]
    false_negative_ids: list[str]
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return precision, recall, f1


def load_hpo_aliases(obo_path: Path) -> dict[str, str]:
    """Return current-ID mappings for current and alternate HPO identifiers."""
    aliases: dict[str, str] = {}
    current_id: str | None = None
    alternate_ids: list[str] = []
    replaced_by: str | None = None

    def finish_term() -> None:
        if current_id is None:
            return
        target = replaced_by or current_id
        aliases[current_id] = target
        aliases.update({alternate: target for alternate in alternate_ids})

    with obo_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                finish_term()
                current_id = None
                alternate_ids = []
                replaced_by = None
            elif line.startswith("id: HP:"):
                current_id = line.removeprefix("id: ").strip()
            elif line.startswith("alt_id: HP:"):
                alternate_ids.append(line.removeprefix("alt_id: ").strip())
            elif line.startswith("replaced_by: HP:"):
                replaced_by = line.removeprefix("replaced_by: ").strip()
            elif line.startswith("[") and line.endswith("]"):
                finish_term()
                current_id = None
                alternate_ids = []
                replaced_by = None
    finish_term()
    if not aliases:
        raise ValueError(f"no HPO identifiers were found in {obo_path}")
    return aliases


def _normalize_hpo_id(value: object, aliases: dict[str, str]) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    return aliases.get(candidate, candidate)


def load_reference_sets(
    path: Path,
    aliases: dict[str, str],
    *,
    patient_column: str = "Patient ID",
    hpo_column: str = "hpo_term",
) -> dict[str, set[str]]:
    references: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or patient_column not in reader.fieldnames:
            raise ValueError(f"reference CSV requires a {patient_column!r} column")
        if hpo_column not in reader.fieldnames:
            raise ValueError(f"reference CSV requires a {hpo_column!r} column")
        for row in reader:
            patient_id = str(row.get(patient_column) or "").strip()
            hpo_id = _normalize_hpo_id(row.get(hpo_column), aliases)
            if patient_id and hpo_id:
                references.setdefault(patient_id, set()).add(hpo_id)
    return references


def load_reference_groups(
    path: Path,
    aliases: dict[str, str],
    *,
    patient_column: str = "Patient ID",
    hpo_column: str = "hpo_term",
) -> dict[str, list[set[str]]]:
    """Load one reference finding per row, with comma-delimited IDs as alternatives."""
    references: dict[str, list[set[str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or patient_column not in reader.fieldnames:
            raise ValueError(f"reference CSV requires a {patient_column!r} column")
        if hpo_column not in reader.fieldnames:
            raise ValueError(f"reference CSV requires a {hpo_column!r} column")
        for row in reader:
            patient_id = str(row.get(patient_column) or "").strip()
            identifiers = {
                aliases.get(identifier, identifier)
                for identifier in HPO_ID.findall(str(row.get(hpo_column) or ""))
            }
            if patient_id and identifiers:
                references.setdefault(patient_id, []).append(identifiers)
    return references


def load_prediction_sets(
    path: Path,
    aliases: dict[str, str],
    *,
    accepted_only: bool = False,
) -> dict[str, set[str]]:
    if path.suffix.lower() == ".json":
        raw_rows: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, list):
            raise ValueError("prediction JSON must contain a list of result objects")
        rows = raw_rows
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

    predictions: dict[str, set[str]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise ValueError("each prediction must be an object")
        patient_id = str(raw_row.get("patient_id") or "").strip()
        hpo_id = _normalize_hpo_id(raw_row.get("hpo_id"), aliases)
        if patient_id:
            predictions.setdefault(patient_id, set())
            accepted = not accepted_only or raw_row.get("review_status") in (
                None,
                "",
                "accepted",
            )
            if hpo_id and accepted and raw_row.get("mapping_status") in (None, "", "mapped"):
                predictions[patient_id].add(hpo_id)
    return predictions


def score_sets(
    predictions: dict[str, set[str]],
    references: dict[str, set[str]],
    *,
    patient_ids: list[str] | None = None,
) -> list[CaseScore]:
    selected_ids = patient_ids or sorted(
        set(predictions) | set(references),
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    scores: list[CaseScore] = []
    for patient_id in selected_ids:
        predicted = predictions.get(patient_id, set())
        reference = references.get(patient_id, set())
        true_positive = predicted & reference
        false_positive = predicted - reference
        false_negative = reference - predicted
        tp, fp, fn = len(true_positive), len(false_positive), len(false_negative)
        precision, recall, f1 = _metrics(tp, fp, fn)
        scores.append(
            CaseScore(
                patient_id=patient_id,
                predicted_ids=sorted(predicted),
                reference_ids=sorted(reference),
                true_positive_ids=sorted(true_positive),
                false_positive_ids=sorted(false_positive),
                false_negative_ids=sorted(false_negative),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return scores


def _format_reference_group(group: set[str]) -> str:
    return "|".join(sorted(group))


def _maximum_reference_matching(
    predictions: set[str],
    groups: list[set[str]],
) -> dict[int, str]:
    matched_group: dict[int, str] = {}

    def augment(prediction: str, visited: set[int]) -> bool:
        for index, group in enumerate(groups):
            if index in visited or prediction not in group:
                continue
            visited.add(index)
            existing = matched_group.get(index)
            if existing is None or augment(existing, visited):
                matched_group[index] = prediction
                return True
        return False

    for prediction in sorted(predictions):
        augment(prediction, set())
    return matched_group


def score_reference_groups(
    predictions: dict[str, set[str]],
    references: dict[str, list[set[str]]],
    *,
    patient_ids: list[str] | None = None,
) -> list[CaseScore]:
    """Score predictions against one-to-one alternative-ID reference findings."""
    selected_ids = patient_ids or sorted(
        references,
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    scores: list[CaseScore] = []
    for patient_id in selected_ids:
        predicted = predictions.get(patient_id, set())
        groups = references.get(patient_id, [])
        matched_group = _maximum_reference_matching(predicted, groups)
        matched_predictions = set(matched_group.values())
        unmatched_groups = [
            _format_reference_group(group)
            for index, group in enumerate(groups)
            if index not in matched_group
        ]
        tp = len(matched_group)
        fp = len(predicted) - tp
        fn = len(groups) - tp
        precision, recall, f1 = _metrics(tp, fp, fn)
        scores.append(
            CaseScore(
                patient_id=patient_id,
                predicted_ids=sorted(predicted),
                reference_ids=[_format_reference_group(group) for group in groups],
                true_positive_ids=sorted(matched_predictions),
                false_positive_ids=sorted(predicted - matched_predictions),
                false_negative_ids=sorted(unmatched_groups),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return scores


def summarize(scores: list[CaseScore]) -> dict[str, Any]:
    tp = sum(score.tp for score in scores)
    fp = sum(score.fp for score in scores)
    fn = sum(score.fn for score in scores)
    precision, recall, f1 = _metrics(tp, fp, fn)
    count = len(scores)
    return {
        "case_count": count,
        "micro": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "macro": {
            "precision": sum(score.precision for score in scores) / count if count else 0.0,
            "recall": sum(score.recall for score in scores) / count if count else 0.0,
            "f1": sum(score.f1 for score in scores) / count if count else 0.0,
        },
    }


def write_benchmark_report(
    scores: list[CaseScore],
    output_dir: Path,
    *,
    predictions_path: Path,
    references_path: Path,
    ontology_path: Path,
    metadata: dict[str, str],
    input_path: Path | None = None,
    prompt_path: Path | None = None,
    vector_manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_per_case.csv"
    json_path = output_dir / "benchmark_report.json"
    columns = list(CaseScore.__annotations__)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for score in scores:
            row = asdict(score)
            for key in (
                "predicted_ids",
                "reference_ids",
                "true_positive_ids",
                "false_positive_ids",
                "false_negative_ids",
            ):
                row[key] = "|".join(row[key])
            writer.writerow(row)

    provenance = {
        "predictions_path": predictions_path.name,
        "predictions_sha256": sha256_file(predictions_path),
        "references_path": references_path.name,
        "references_sha256": sha256_file(references_path),
        "ontology_path": ontology_path.name,
        "ontology_sha256": sha256_file(ontology_path),
        **dict(sorted(metadata.items())),
    }
    for prefix, path in (
        ("input", input_path),
        ("prompts", prompt_path),
        ("vector_manifest", vector_manifest_path),
    ):
        if path is not None:
            provenance[f"{prefix}_path"] = path.name
            provenance[f"{prefix}_sha256"] = sha256_file(path)

    report = {
        "schema_version": "1.0",
        "provenance": provenance,
        "summary": summarize(scores),
        "cases": [asdict(score) for score in scores],
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path
