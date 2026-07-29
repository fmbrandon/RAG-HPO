#!/usr/bin/env python3
"""Compare two recognizers and their union/intersection on one reference corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    CaseScore,
    load_hpo_aliases,
    load_prediction_sets,
    load_reference_groups,
    score_reference_groups,
    summarize,
)
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.statistics import paired_inference


def _merge_prediction_files(
    paths: list[Path],
    aliases: dict[str, str],
) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for path in paths:
        for patient_id, values in load_prediction_sets(path, aliases).items():
            merged.setdefault(patient_id, set()).update(values)
    return merged


def _pairs(scores: list[CaseScore], field: str) -> set[tuple[str, str]]:
    return {(score.patient_id, value) for score in scores for value in getattr(score, field)}


def _overlap(first: set[tuple[str, str]], second: set[tuple[str, str]]) -> dict[str, int]:
    return {
        "shared": len(first & second),
        "first_only": len(first - second),
        "second_only": len(second - first),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, action="append", required=True)
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--second", type=Path, action="append", required=True)
    parser.add_argument("--second-label", required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aliases = load_hpo_aliases(args.ontology)
    first = _merge_prediction_files(args.first, aliases)
    second = _merge_prediction_files(args.second, aliases)
    references = load_reference_groups(args.references, aliases)
    patient_ids = sorted(
        set(first) & set(second) & set(references),
        key=lambda value: (
            not value.isdigit(),
            int(value) if value.isdigit() else value,
        ),
    )
    if not patient_ids:
        parser.error("the prediction files have no shared reference cases")

    union = {
        patient_id: first.get(patient_id, set()) | second.get(patient_id, set())
        for patient_id in patient_ids
    }
    intersection = {
        patient_id: first.get(patient_id, set()) & second.get(patient_id, set())
        for patient_id in patient_ids
    }
    first_scores = score_reference_groups(first, references, patient_ids=patient_ids)
    second_scores = score_reference_groups(second, references, patient_ids=patient_ids)
    union_scores = score_reference_groups(union, references, patient_ids=patient_ids)
    intersection_scores = score_reference_groups(
        intersection,
        references,
        patient_ids=patient_ids,
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "case_count": len(patient_ids),
        "reference_policy": "comma-delimited IDs are alternatives",
        "systems": {
            args.first_label: summarize(first_scores),
            args.second_label: summarize(second_scores),
            "unverified_union": summarize(union_scores),
            "exact_id_intersection": summarize(intersection_scores),
        },
        "per_case_f1": paired_inference(
            [
                second_score.f1 - first_score.f1
                for first_score, second_score in zip(first_scores, second_scores, strict=True)
            ],
            seed=20260728,
            iterations=20_000,
        ),
        "per_case_f1_direction": f"{args.second_label} minus {args.first_label}",
        "prediction_overlap": {
            "true_positives": _overlap(
                _pairs(first_scores, "true_positive_ids"),
                _pairs(second_scores, "true_positive_ids"),
            ),
            "false_positives": _overlap(
                _pairs(first_scores, "false_positive_ids"),
                _pairs(second_scores, "false_positive_ids"),
            ),
        },
        "provenance": {
            "first": [{"path": path.name, "sha256": sha256_file(path)} for path in args.first],
            "second": [{"path": path.name, "sha256": sha256_file(path)} for path in args.second],
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
        },
        "interpretation_guard": (
            "The union is an upper-bound candidate experiment, not a release policy. "
            "It must be followed by assertion and ambiguity verification."
        ),
    }
    ensure_private_directory(args.output.parent)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
