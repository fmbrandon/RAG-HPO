#!/usr/bin/env python3
"""Repair failed final categorization without repeating extraction or mapping."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

from rag_hpo.artifacts import sha256_file
from rag_hpo.config import DEFAULT_BASE_URL, DEFAULT_MODEL, ProviderConfig
from rag_hpo.export import export_results
from rag_hpo.models import AnnotationInput, AnnotationResult
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.provider import OpenAICompatibleProvider
from rag_hpo.staged_pipeline import StagedAnnotationPipeline

CALCULATION_ERROR = "incomplete_final_category"


def _load_inputs(path: Path) -> dict[str, AnnotationInput]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, AnnotationInput] = {}
    for raw in rows:
        row = AnnotationInput.model_validate(
            {
                "patient_id": raw.get("patient_id"),
                "clinical_note": raw.get("clinical_note"),
            }
        )
        output[row.patient_id] = row
    return output


def _load_results(path: Path) -> list[AnnotationResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("prediction JSON must contain a list")
    return [AnnotationResult.model_validate(row) for row in raw]


def _affected_cases(results: list[AnnotationResult]) -> list[str]:
    return sorted(
        {result.patient_id for result in results if result.error_code == CALCULATION_ERROR},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse completed extraction and mapping rows and rerun only failed "
            "final-category decisions."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAG_HPO_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("RAG_HPO_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--confirm-external-transmission", action="store_true")
    parser.add_argument("--keep-raw-responses", action="store_true")
    args = parser.parse_args()

    hostname = urlparse(args.base_url).hostname
    remote = hostname not in {"localhost", "127.0.0.1", "::1"}
    if remote and not args.confirm_external_transmission:
        parser.error("remote repair requires --confirm-external-transmission")
    if remote and not os.environ.get("RAG_HPO_API_KEY"):
        parser.error("RAG_HPO_API_KEY is not set")

    results = _load_results(args.predictions)
    inputs = _load_inputs(args.input)
    discovered = _affected_cases(results)
    selected = discovered if args.case_ids is None else [str(value) for value in args.case_ids]
    unknown = sorted(set(selected) - set(discovered))
    if unknown:
        parser.error(f"selected cases have no {CALCULATION_ERROR!r} rows: {unknown}")
    missing_inputs = sorted(set(selected) - set(inputs))
    if missing_inputs:
        parser.error(f"input CSV is missing selected cases: {missing_inputs}")

    by_case: dict[str, list[AnnotationResult]] = defaultdict(list)
    for result in results:
        by_case[result.patient_id].append(result)

    ensure_private_directory(args.output_dir)
    config = ProviderConfig.from_env(base_url=args.base_url, model=args.model)
    with OpenAICompatibleProvider(config) as provider:
        repairer = StagedAnnotationPipeline.for_final_category_repair(
            provider=provider,
            output_dir=args.output_dir,
            keep_raw_responses=args.keep_raw_responses,
        )
        for row_index, patient_id in enumerate(
            tqdm(selected, desc="Final category repair", unit="case", dynamic_ncols=True)
        ):
            repaired = repairer.repair_final_categories(
                inputs[patient_id],
                row_index,
                by_case[patient_id],
            )
            remaining = [result for result in repaired if result.error_code == CALCULATION_ERROR]
            if remaining:
                raise RuntimeError(f"case {patient_id} still has {len(remaining)} failed decisions")
            by_case[patient_id] = repaired
        usage = dict(provider.usage)

    merged: list[AnnotationResult] = []
    emitted: set[str] = set()
    for result in results:
        if result.patient_id in selected:
            if result.patient_id not in emitted:
                merged.extend(by_case[result.patient_id])
                emitted.add(result.patient_id)
        else:
            merged.append(result)

    csv_path, json_path = export_results(merged, args.output_dir)
    manifest_path = args.output_dir / "final_category_repair_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "repair": CALCULATION_ERROR,
                "selected_case_ids": selected,
                "case_count": len(selected),
                "input_sha256": sha256_file(args.input),
                "source_predictions_sha256": sha256_file(args.predictions),
                "output_predictions_sha256": sha256_file(json_path),
                "provider": config.redacted(),
                "provider_usage": usage,
                "raw_responses_retained": args.keep_raw_responses,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    restrict_owner(manifest_path)
    print(
        json.dumps(
            {
                "affected_cases": len(selected),
                "csv": str(csv_path),
                "json": str(json_path),
                "manifest": str(manifest_path),
                "remaining_calculation_errors": sum(
                    result.error_code == CALCULATION_ERROR for result in merged
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
