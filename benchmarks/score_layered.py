#!/usr/bin/env python3
"""Produce exact and one-/two-edge HPO scores from one prediction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    load_prediction_calculation_errors,
    load_prediction_groups,
    load_reference_groups,
)
from rag_hpo.diagnostics import parse_obo
from rag_hpo.hierarchy_scoring import score_layered, summarize_layered


def _selection_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selection = raw.get("selection", raw)
    for key in (
        "selected_case_ids",
        "confirmation_case_ids",
        "repeat_case_ids",
        "case_ids",
    ):
        values = selection.get(key)
        if isinstance(values, list):
            return [str(value) for value in values]
    raise ValueError("selection manifest does not contain a recognized case-ID list")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score alternative-aware exact, one-edge, and two-edge HPO agreement."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-only", action="store_true")
    args = parser.parse_args()

    ontology = parse_obo(args.ontology)
    predictions = load_prediction_groups(
        args.predictions,
        ontology.aliases,
        accepted_only=args.accepted_only,
    )
    references = load_reference_groups(args.references, ontology.aliases)
    patient_ids = _selection_ids(args.selection_manifest)
    calculation_errors = load_prediction_calculation_errors(args.predictions)
    unscorable = {
        patient_id: calculation_errors.get(patient_id, [])
        for patient_id in patient_ids
        if patient_id in calculation_errors
    }
    for patient_id in patient_ids:
        if patient_id not in references:
            unscorable.setdefault(patient_id, []).append("missing_reference")
    scorable_ids = [
        patient_id for patient_id in patient_ids if patient_id not in unscorable
    ]
    layers = {
        str(distance): summarize_layered(
            score_layered(
                predictions,
                references,
                ontology,
                max_distance=distance,
                patient_ids=scorable_ids,
            )
        )
        for distance in (0, 1, 2)
    }
    report = {
        "schema_version": "1.0",
        "interpretation": {
            "primary": "distance 0 is strict alternative-aware exact scoring",
            "sensitivity": (
                "distances 1 and 2 count one-to-one parent, child, or sibling "
                "matches and do not replace the primary score"
            ),
        },
        "selection": {
            "selected_case_count": len(patient_ids),
            "scored_case_count": len(scorable_ids),
            "unscorable_case_count": len(unscorable),
            "unscorable_cases": [
                {
                    "patient_id": patient_id,
                    "error_codes": sorted(set(error_codes)),
                }
                for patient_id, error_codes in unscorable.items()
            ],
        },
        "provenance": {
            "prediction_policy": "accepted-only" if args.accepted_only else "all-mapped",
            "predictions_sha256": sha256_file(args.predictions),
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "ontology_version": ontology.data_version,
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
        },
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "layers": layers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
