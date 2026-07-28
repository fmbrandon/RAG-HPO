#!/usr/bin/env python3
"""Create a transparent, deterministic CSC sample for a live paired benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--required-case", default="68")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--historical-model", default="LLaMa 4-Scout")
    args = parser.parse_args()
    if args.sample_size < 2:
        parser.error("--sample-size must be at least 2")

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        input_reader = csv.DictReader(handle)
        input_rows = list(input_reader)
        input_fields = input_reader.fieldnames
    if input_fields != ["Case", "clinical_note"]:
        parser.error("input must contain exactly Case and clinical_note columns")
    inputs = {row["Case"]: row for row in input_rows}

    with args.references.open(encoding="utf-8-sig", newline="") as handle:
        references = {
            row["Patient ID"]
            for row in csv.DictReader(handle)
            if row.get("Patient ID") and row.get("hpo_term")
        }
    with args.historical_metrics.open(encoding="utf-8-sig", newline="") as handle:
        historical = {
            row["patient_id"]
            for row in csv.DictReader(handle)
            if row["sheet"] == "CSC Comparison Results" and row["model"] == args.historical_model
        }

    eligible = sorted(set(inputs) & references & historical, key=int)
    if args.required_case not in eligible:
        parser.error(f"required Case {args.required_case} is not eligible")
    if args.sample_size > len(eligible):
        parser.error("sample size exceeds the eligible case count")

    population = [case_id for case_id in eligible if case_id != args.required_case]
    selected = random.Random(args.seed).sample(  # noqa: S311
        population, args.sample_size - 1
    )
    selected.append(args.required_case)
    selected.sort(key=int)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=input_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(inputs[case_id] for case_id in selected)

    manifest = {
        "schema_version": "1.0",
        "purpose": "paired current-versus-historical CSC benchmark",
        "selection": {
            "eligible_case_count": len(eligible),
            "method": (
                f"Case {args.required_case} forced for repaired-input validation; "
                f"remaining {args.sample_size - 1} sampled without replacement"
            ),
            "required_case": args.required_case,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "selected_case_ids": selected,
        },
        "historical_model": args.historical_model,
        "source_hashes": {
            "input_sha256": sha256(args.input),
            "references_sha256": sha256(args.references),
            "historical_metrics_sha256": sha256(args.historical_metrics),
            "sample_input_sha256": sha256(args.output_csv),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
