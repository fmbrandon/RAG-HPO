#!/usr/bin/env python3
"""Lock the untouched paired CSC confirmation cohort before live inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.privacy import ensure_private_directory, restrict_owner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--private-input-csv", type=Path, required=True)
    parser.add_argument("--private-repeat-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--historical-model", default="LLaMa 4-Scout")
    parser.add_argument("--repeat-size", type=int, default=20)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        input_rows = {
            str(row["Case"]).strip(): {
                "Case": str(row["Case"]).strip(),
                "clinical_note": str(row["clinical_note"]),
            }
            for row in reader
            if row.get("Case") and str(row.get("clinical_note") or "").strip()
        }
    reference_counts: Counter[str] = Counter()
    with args.references.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            patient_id = str(row.get("Patient ID") or "").strip()
            if patient_id and row.get("hpo_term"):
                reference_counts[patient_id] += 1
    with args.historical_metrics.open(encoding="utf-8-sig", newline="") as handle:
        historical_ids = {
            str(row["patient_id"]).strip()
            for row in csv.DictReader(handle)
            if row["sheet"] == "CSC Comparison Results" and row["model"] == args.historical_model
        }
    discovery = json.loads(args.discovery_manifest.read_text(encoding="utf-8"))
    discovery_ids = {str(value) for value in discovery["selection"]["selected_case_ids"]}
    eligible = sorted(
        set(input_rows) & set(reference_counts) & historical_ids,
        key=int,
    )
    missing_discovery = discovery_ids - set(eligible)
    if missing_discovery:
        parser.error(
            "discovery cases are no longer eligible: "
            + ", ".join(sorted(missing_discovery, key=int))
        )
    confirmation_ids = [value for value in eligible if value not in discovery_ids]
    if not confirmation_ids:
        parser.error("no untouched eligible cases remain")
    if not 1 <= args.repeat_size <= len(confirmation_ids):
        parser.error("--repeat-size must be between 1 and the confirmation cohort size")

    ranked = sorted(
        confirmation_ids,
        key=lambda patient_id: (
            reference_counts[patient_id],
            len(input_rows[patient_id]["clinical_note"]),
            int(patient_id),
        ),
    )
    repeat_ids = sorted(
        {
            ranked[min(len(ranked) - 1, int((index + 0.5) * len(ranked) / args.repeat_size))]
            for index in range(args.repeat_size)
        },
        key=int,
    )
    if len(repeat_ids) != args.repeat_size:
        parser.error("stratified repeat selection produced duplicate cases")

    ensure_private_directory(args.private_input_csv.parent)
    with args.private_input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Case", "clinical_note"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(input_rows[patient_id] for patient_id in confirmation_ids)
    restrict_owner(args.private_input_csv)
    ensure_private_directory(args.private_repeat_csv.parent)
    with args.private_repeat_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Case", "clinical_note"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(input_rows[patient_id] for patient_id in repeat_ids)
    restrict_owner(args.private_repeat_csv)

    manifest = {
        "schema_version": "1.0",
        "purpose": "untouched paired CSC accuracy confirmation",
        "historical_model": args.historical_model,
        "selection": {
            "eligible_case_count": len(eligible),
            "discovery_case_count": len(discovery_ids),
            "confirmation_case_count": len(confirmation_ids),
            "confirmation_case_ids": confirmation_ids,
            "repeat_case_count": len(repeat_ids),
            "repeat_case_ids": repeat_ids,
            "repeat_method": (
                "even positions after deterministic ordering by manual-reference "
                "count, note length, and numeric case ID"
            ),
        },
        "source_hashes": {
            "input_sha256": sha256_file(args.input),
            "references_sha256": sha256_file(args.references),
            "historical_metrics_sha256": sha256_file(args.historical_metrics),
            "discovery_manifest_sha256": sha256_file(args.discovery_manifest),
            "private_confirmation_input_sha256": sha256_file(args.private_input_csv),
            "private_repeat_input_sha256": sha256_file(args.private_repeat_csv),
        },
        "privacy": {
            "tracked_manifest_contains_note_text": False,
            "confirmation_input_location": "private audit artifacts outside Git",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
