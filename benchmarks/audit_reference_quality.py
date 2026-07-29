#!/usr/bin/env python3
"""Audit manual-reference identifiers without modifying the source CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.diagnostics import parse_obo

HPO_ID = re.compile(r"HP:\d{7}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ontology = parse_obo(args.ontology)

    with args.references.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    compound_cells = []
    invalid_cells = []
    status_counts: Counter[str] = Counter()
    duplicate_pairs: Counter[tuple[str, str]] = Counter()
    affected_patients: set[str] = set()
    obsolete_ids: set[str] = set()
    replaced_obsolete_ids: set[str] = set()
    missing_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        patient_id = str(row.get("Patient ID") or "").strip()
        raw_value = str(row.get("hpo_term") or "").strip()
        ids = HPO_ID.findall(raw_value)
        if len(ids) > 1:
            compound_cells.append(
                {
                    "row": row_number,
                    "patient_id": patient_id,
                    "raw_value": raw_value,
                    "candidate_ids": ids,
                }
            )
            affected_patients.add(patient_id)
        if not ids:
            invalid_cells.append(
                {
                    "row": row_number,
                    "patient_id": patient_id,
                    "raw_value": raw_value,
                }
            )
            continue
        for hpo_id in ids:
            duplicate_pairs[(patient_id, hpo_id)] += 1
            raw_term = ontology.terms.get(hpo_id)
            normalized = ontology.normalize(hpo_id)
            term = ontology.terms.get(normalized)
            if raw_term is not None and raw_term.obsolete:
                if raw_term.replaced_by:
                    status_counts["obsolete_with_replacement"] += 1
                    replaced_obsolete_ids.add(hpo_id)
                else:
                    status_counts["obsolete_without_replacement"] += 1
                    obsolete_ids.add(hpo_id)
            elif term is None:
                status_counts["missing"] += 1
                missing_ids.add(hpo_id)
            elif normalized != hpo_id:
                status_counts["alternate_id"] += 1
            else:
                status_counts["current"] += 1

    duplicate_values = [
        {"patient_id": patient_id, "hpo_id": hpo_id, "count": count}
        for (patient_id, hpo_id), count in sorted(duplicate_pairs.items())
        if count > 1
    ]
    report = {
        "schema_version": "1.0",
        "provenance": {
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "ontology_data_version": ontology.data_version,
        },
        "reference_row_count": len(rows),
        "parsed_identifier_count": sum(status_counts.values()),
        "identifier_status_counts": dict(sorted(status_counts.items())),
        "compound_id_cells": {
            "count": len(compound_cells),
            "affected_patient_ids": sorted(affected_patients, key=int),
            "rows": compound_cells,
            "scoring_risk": (
                "The exact scorer treats each unsplit cell as one impossible identifier. "
                "The lab owner confirmed that the IDs are alternatives for one finding."
            ),
        },
        "cells_without_hpo_id": invalid_cells,
        "duplicate_patient_id_pairs": duplicate_values,
        "obsolete_ids_with_replacement": sorted(replaced_obsolete_ids),
        "obsolete_ids_without_replacement": sorted(obsolete_ids),
        "missing_ids": sorted(missing_ids),
        "recommended_policy": (
            "Keep the source unchanged and score each compound cell as one alternative-ID "
            "group using deterministic one-to-one matching."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
