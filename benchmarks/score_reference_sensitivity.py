#!/usr/bin/env python3
"""Score compound manual-reference cells under explicit alternative policies."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    _metrics,
    load_hpo_aliases,
    load_prediction_sets,
    load_reference_sets,
    score_sets,
    summarize,
)

HPO_ID = re.compile(r"HP:\d{7}")


def _match_groups(predictions: set[str], groups: list[set[str]]) -> int:
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
    return len(matched_group)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    precision, recall, f1 = _metrics(tp, fp, fn)
    return {
        "case_count": len(rows),
        "micro": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    patient_ids = [
        str(value)
        for value in selection["selection"].get(
            "selected_case_ids",
            selection["selection"].get("confirmation_case_ids", []),
        )
    ]
    aliases = load_hpo_aliases(args.ontology)
    predictions = load_prediction_sets(args.predictions, aliases)
    strict_references = load_reference_sets(args.references, aliases)
    strict_scores = score_sets(
        predictions,
        strict_references,
        patient_ids=patient_ids,
    )

    groups_by_patient: dict[str, list[set[str]]] = defaultdict(list)
    split_references: dict[str, set[str]] = defaultdict(set)
    with args.references.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            patient_id = str(row.get("Patient ID") or "").strip()
            raw_ids = HPO_ID.findall(str(row.get("hpo_term") or ""))
            group = {aliases.get(value, value) for value in raw_ids}
            if patient_id and group:
                groups_by_patient[patient_id].append(group)
                split_references[patient_id].update(group)

    alternative_rows = []
    for patient_id in patient_ids:
        predicted = predictions.get(patient_id, set())
        groups = groups_by_patient.get(patient_id, [])
        tp = _match_groups(predicted, groups)
        fp = len(predicted) - tp
        fn = len(groups) - tp
        precision, recall, f1 = _metrics(tp, fp, fn)
        alternative_rows.append(
            {
                "patient_id": patient_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    split_scores = score_sets(
        predictions,
        split_references,
        patient_ids=patient_ids,
    )
    report = {
        "schema_version": "1.0",
        "strict_unsplit_cells": summarize(strict_scores),
        "compound_ids_are_alternatives": _aggregate(alternative_rows),
        "compound_ids_are_all_required": summarize(split_scores),
        "policy_note": (
            "The lab owner confirmed that comma-delimited IDs are alternatives for one "
            "finding. The strict and all-required results are retained only as historical "
            "sensitivity bounds."
        ),
        "provenance": {
            "predictions_sha256": sha256_file(args.predictions),
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
