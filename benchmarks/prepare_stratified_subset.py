#!/usr/bin/env python3
"""Create a deterministic, performance-blind benchmark subset.

Cases are stratified by note length and manual-reference count.  The selector
never reads predictions or scores, so it cannot preferentially choose cases on
which a model already performs well.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _excluded_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    selection = raw.get("selection", raw)
    for key in (
        "selected_case_ids",
        "confirmation_case_ids",
        "case_ids",
    ):
        values = selection.get(key)
        if isinstance(values, list):
            return {str(value).strip() for value in values}
    raise ValueError(f"{path} does not contain a recognized case-ID list")


def _rank_bins(
    case_ids: list[str],
    values: dict[str, int],
    bin_count: int,
) -> dict[str, int]:
    ranked = sorted(case_ids, key=lambda value: (values[value], _natural_key(value)))
    total = len(ranked)
    return {
        patient_id: min(bin_count - 1, index * bin_count // total)
        for index, patient_id in enumerate(ranked)
    }


def _allocate(
    strata: dict[tuple[int, int], list[str]],
    sample_size: int,
) -> dict[tuple[int, int], int]:
    nonempty = sorted(strata)
    total = sum(len(strata[key]) for key in nonempty)
    allocation = {key: 0 for key in nonempty}

    # When possible, retain at least one case from every observed combination.
    if sample_size >= len(nonempty):
        for key in nonempty:
            allocation[key] = 1

    remaining = sample_size - sum(allocation.values())
    while remaining:
        candidates = [key for key in nonempty if allocation[key] < len(strata[key])]
        if not candidates:
            raise ValueError("unable to allocate requested sample")
        key = max(
            candidates,
            key=lambda value: (
                sample_size * len(strata[value]) / total - allocation[value],
                len(strata[value]) - allocation[value],
                tuple(-part for part in value),
            ),
        )
        allocation[key] += 1
        remaining -= 1
    return allocation


def _summary(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patient-column", required=True)
    parser.add_argument("--reference-patient-column", default="Patient ID")
    parser.add_argument("--note-column", default="clinical_note")
    parser.add_argument("--reference-id-column", default="hpo_term")
    parser.add_argument("--exclude-manifest", type=Path)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--strata", type=int, default=3)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    if args.sample_size < 2:
        parser.error("--sample-size must be at least 2")
    if args.strata < 2:
        parser.error("--strata must be at least 2")

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if args.patient_column not in fieldnames or args.note_column not in fieldnames:
            parser.error("input is missing the patient or note column")
        rows = {
            str(row[args.patient_column]).strip(): row
            for row in reader
            if str(row.get(args.patient_column) or "").strip()
            and str(row.get(args.note_column) or "").strip()
        }

    reference_counts: Counter[str] = Counter()
    with args.references.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        reference_fields = reader.fieldnames or []
        if (
            args.reference_patient_column not in reference_fields
            or args.reference_id_column not in reference_fields
        ):
            parser.error("reference file is missing the patient or HPO ID column")
        for row in reader:
            patient_id = str(row.get(args.reference_patient_column) or "").strip()
            hpo_id = str(row.get(args.reference_id_column) or "").strip()
            if patient_id and hpo_id:
                reference_counts[patient_id] += 1

    try:
        excluded = _excluded_ids(args.exclude_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    eligible = sorted(
        (set(rows) & set(reference_counts)) - excluded,
        key=_natural_key,
    )
    if args.sample_size > len(eligible):
        parser.error(f"sample size {args.sample_size} exceeds {len(eligible)} eligible cases")

    note_lengths = {
        patient_id: len(str(rows[patient_id][args.note_column])) for patient_id in eligible
    }
    reference_values = {patient_id: reference_counts[patient_id] for patient_id in eligible}
    length_bins = _rank_bins(eligible, note_lengths, args.strata)
    reference_bins = _rank_bins(eligible, reference_values, args.strata)
    grouped: dict[tuple[int, int], list[str]] = defaultdict(list)
    for patient_id in eligible:
        grouped[(length_bins[patient_id], reference_bins[patient_id])].append(patient_id)
    allocation = _allocate(dict(grouped), args.sample_size)

    selected: list[str] = []
    for stratum, count in sorted(allocation.items()):
        candidates = sorted(grouped[stratum], key=_natural_key)
        derived_seed = args.seed + 1009 * stratum[0] + 9176 * stratum[1]
        selected.extend(random.Random(derived_seed).sample(candidates, count))  # noqa: S311
    selected.sort(key=_natural_key)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output_csv.parent.chmod(0o700)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[patient_id] for patient_id in selected)
    args.output_csv.chmod(0o600)

    selected_note_lengths = [note_lengths[value] for value in selected]
    selected_reference_counts = [reference_values[value] for value in selected]
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "purpose": "performance-blind stratified subset evaluation",
        "corpus": args.corpus,
        "selection": {
            "method": (
                f"deterministic {args.strata}x{args.strata} rank strata over "
                "note character count and "
                "manual-reference count; seeded sampling within each stratum"
            ),
            "seed": args.seed,
            "sample_size": args.sample_size,
            "eligible_case_count": len(eligible),
            "excluded_case_count": len(set(rows) & excluded),
            "selected_case_ids": selected,
            "strata": [
                {
                    "note_length_bin": key[0],
                    "reference_count_bin": key[1],
                    "eligible": len(grouped[key]),
                    "selected": allocation[key],
                }
                for key in sorted(grouped)
            ],
        },
        "coverage": {
            "eligible_note_characters": _summary(list(note_lengths.values())),
            "selected_note_characters": _summary(selected_note_lengths),
            "eligible_reference_ids": _summary(list(reference_values.values())),
            "selected_reference_ids": _summary(selected_reference_counts),
            "selected_reference_id_total": sum(selected_reference_counts),
        },
        "source_hashes": {
            "input_sha256": sha256_file(args.input),
            "references_sha256": sha256_file(args.references),
            "exclude_manifest_sha256": (
                sha256_file(args.exclude_manifest) if args.exclude_manifest is not None else None
            ),
            "private_subset_input_sha256": sha256_file(args.output_csv),
        },
        "privacy": {
            "tracked_manifest_contains_note_text": False,
            "subset_input_location": "private audit artifacts outside Git",
        },
        "inference_limit": (
            "Thirty cases support an efficient comparative estimate and include "
            "hundreds of reference phenotypes, but corpus-wide claims must include "
            "bootstrap intervals and must not be described as a full-corpus result."
        ),
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
