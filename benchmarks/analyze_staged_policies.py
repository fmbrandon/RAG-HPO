#!/usr/bin/env python3
"""Compare general staged-output policies on a discovery cohort."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    load_hpo_aliases,
    load_reference_groups,
    score_reference_groups,
    summarize,
)

HPO_ID = re.compile(r"^HP:\d{7}$")
Row = dict[str, Any]


def _selection(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = raw.get("selection", raw)
    for key in ("selected_case_ids", "confirmation_case_ids", "case_ids"):
        selected = value.get(key)
        if isinstance(selected, list):
            return [str(item) for item in selected]
    raise ValueError(f"{path} does not contain a recognized case-ID list")


def _mapped(row: Row) -> bool:
    return (
        row.get("mapping_status") == "mapped"
        and isinstance(row.get("hpo_id"), str)
        and bool(HPO_ID.fullmatch(row["hpo_id"]))
    )


def _has_model(row: Row) -> bool:
    return any(str(value).startswith("model-pass") for value in row.get("source_methods") or [])


def _has_recognizer(row: Row) -> bool:
    return bool({"native", "fasthpocr"} & set(row.get("source_methods") or []))


def _has_both_recognizers(row: Row) -> bool:
    return {"native", "fasthpocr"} <= set(row.get("source_methods") or [])


def _has_pass(row: Row, pass_name: str) -> bool:
    return pass_name in set(row.get("source_methods") or [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("predictions must be a JSON array of objects")
    rows: list[Row] = raw
    aliases = load_hpo_aliases(args.ontology)
    references = load_reference_groups(args.references, aliases)
    selected = _selection(args.selection)

    policies: dict[str, Callable[[Row], bool]] = {
        "accepted": lambda row: row.get("review_status") == "accepted",
        "accepted_high": lambda row: (
            row.get("review_status") == "accepted" and row.get("confidence") == "high"
        ),
        "accepted_model": lambda row: row.get("review_status") == "accepted" and _has_model(row),
        "accepted_model_pass1": lambda row: (
            row.get("review_status") == "accepted" and _has_pass(row, "model-pass-1")
        ),
        "accepted_model_pass2": lambda row: (
            row.get("review_status") == "accepted" and _has_pass(row, "model-pass-2")
        ),
        "accepted_model_recognizer_agreement": lambda row: (
            row.get("review_status") == "accepted" and _has_model(row) and _has_recognizer(row)
        ),
        "accepted_pass1_plus_model_recognizer_agreement": lambda row: (
            row.get("review_status") == "accepted"
            and (_has_pass(row, "model-pass-1") or (_has_model(row) and _has_recognizer(row)))
        ),
        "accepted_pass1_plus_dual_recognizer": lambda row: (
            row.get("review_status") == "accepted"
            and (_has_pass(row, "model-pass-1") or _has_both_recognizers(row))
        ),
        "accepted_plus_high_review": lambda row: (
            row.get("review_status") == "accepted"
            or (row.get("review_status") == "review" and row.get("confidence") == "high")
        ),
        "all_mapped": lambda _row: True,
        "all_mapped_model": _has_model,
        "all_mapped_model_pass1": lambda row: _has_pass(
            row,
            "model-pass-1",
        ),
    }
    report: dict[str, object] = {
        "schema_version": "1.0",
        "cohort_case_count": len(selected),
        "provenance": {
            "predictions_sha256": sha256_file(args.predictions),
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "selection_sha256": sha256_file(args.selection),
        },
        "policies": {},
    }
    for name, include in policies.items():
        predictions: dict[str, set[str]] = {patient_id: set() for patient_id in selected}
        for row in rows:
            patient_id = str(row.get("patient_id") or "")
            if patient_id not in predictions or not _mapped(row) or not include(row):
                continue
            hpo_id = str(row["hpo_id"])
            predictions[patient_id].add(aliases.get(hpo_id, hpo_id))
        scores = score_reference_groups(
            predictions,
            references,
            patient_ids=selected,
        )
        report["policies"][name] = summarize(scores)  # type: ignore[index]

    candidate_predictions: dict[str, set[str]] = {patient_id: set() for patient_id in selected}
    for row in rows:
        patient_id = str(row.get("patient_id") or "")
        if patient_id not in candidate_predictions:
            continue
        for raw_hpo_id in row.get("candidate_hpo_ids") or []:
            hpo_id = str(raw_hpo_id)
            if HPO_ID.fullmatch(hpo_id):
                candidate_predictions[patient_id].add(aliases.get(hpo_id, hpo_id))
    candidate_scores = score_reference_groups(
        candidate_predictions,
        references,
        patient_ids=selected,
    )
    report["candidate_oracle"] = summarize(candidate_scores)

    top1_predictions: dict[str, set[str]] = {patient_id: set() for patient_id in selected}
    for row in rows:
        patient_id = str(row.get("patient_id") or "")
        candidate_ids = row.get("candidate_hpo_ids") or []
        if patient_id in top1_predictions and _has_pass(row, "model-pass-1") and candidate_ids:
            hpo_id = str(candidate_ids[0])
            if HPO_ID.fullmatch(hpo_id):
                top1_predictions[patient_id].add(aliases.get(hpo_id, hpo_id))
    top1_scores = score_reference_groups(
        top1_predictions,
        references,
        patient_ids=selected,
    )
    report["candidate_top1_model_pass1"] = summarize(top1_scores)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
