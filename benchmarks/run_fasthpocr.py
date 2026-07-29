#!/usr/bin/env python3
"""Run a traceable FastHPOCR benchmark without retaining complete notes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    load_hpo_aliases,
    load_prediction_sets,
    load_reference_groups,
    score_reference_groups,
    summarize,
    write_benchmark_report,
)
from rag_hpo.fasthpocr import FastHPORecognizer, build_fasthpocr_index
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.registry import (
    REGISTRY_NAME,
    build_registry,
    load_registry_bundle,
    write_registry_bundle,
)

PREDICTION_COLUMNS = (
    "patient_id",
    "phrase",
    "category",
    "hpo_id",
    "hpo_term",
    "vector_score",
    "mapping_status",
    "error_code",
    "error_message",
    "span_start",
    "span_end",
    "candidate_hpo_ids",
    "recognizer",
)


def _load_inputs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "clinical_note" not in reader.fieldnames:
            raise ValueError("input CSV requires a clinical_note column")
        for index, row in enumerate(reader, start=1):
            patient_id = str(
                row.get("patient_id") or row.get("Case") or row.get("Patient ID") or index
            ).strip()
            note = str(row.get("clinical_note") or "")
            rows.append((patient_id, note))
    return rows


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    restrict_owner(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", choices=("CSC", "GSC"), required=True)
    parser.add_argument("--addons", type=Path)
    parser.add_argument("--longest-match", action="store_true")
    args = parser.parse_args()

    ensure_private_directory(args.output_dir)
    registry_dir = args.index_dir / "registry"
    if (registry_dir / REGISTRY_NAME).is_file():
        registry, _, _ = load_registry_bundle(registry_dir)
    else:
        registry = build_registry(
            args.ontology,
            addons_path=args.addons,
            limit=None,
        )
        write_registry_bundle(
            registry_dir,
            registry=registry,
            hpo_source=str(args.ontology.resolve()),
            hpo_sha256=sha256_file(args.ontology),
            addons_sha256=sha256_file(args.addons) if args.addons else None,
        )
    build_started = time.perf_counter()
    index_path, index_manifest, reused = build_fasthpocr_index(
        ontology_path=args.ontology,
        index_dir=args.index_dir,
        addons_path=args.addons,
        registry_dir=registry_dir,
    )
    build_seconds = time.perf_counter() - build_started
    recognizer = FastHPORecognizer(
        index_path,
        longest_match=args.longest_match,
        registry=registry,
    )
    inputs = _load_inputs(args.input)
    aliases = load_hpo_aliases(args.ontology)

    prediction_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    annotation_started = time.perf_counter()
    mention_count = 0
    for patient_id, note in inputs:
        try:
            annotations = recognizer.annotate(note)
            mention_count += len(annotations)
            for annotation in annotations:
                prediction_rows.append(
                    {
                        "patient_id": patient_id,
                        "phrase": annotation.phrase,
                        "category": "Abnormal",
                        "hpo_id": aliases.get(annotation.hpo_id, annotation.hpo_id),
                        "hpo_term": annotation.hpo_term,
                        "vector_score": "",
                        "mapping_status": (
                            "ambiguous" if annotation.resolution == "ambiguous" else "mapped"
                        ),
                        "error_code": (
                            "ambiguous_hpo" if annotation.resolution == "ambiguous" else ""
                        ),
                        "error_message": (
                            "Multiple canonical HPO IDs share this normalized phrase."
                            if annotation.resolution == "ambiguous"
                            else ""
                        ),
                        "span_start": annotation.start_offset,
                        "span_end": annotation.end_offset,
                        "candidate_hpo_ids": ",".join(annotation.candidate_hpo_ids),
                        "recognizer": "FastHPOCR",
                    }
                )
        except Exception as exc:
            message = str(exc).replace("\n", " ")[:300]
            errors.append({"patient_id": patient_id, "error": message})
            prediction_rows.append(
                {
                    "patient_id": patient_id,
                    "phrase": "",
                    "category": "",
                    "hpo_id": "",
                    "hpo_term": "",
                    "vector_score": "",
                    "mapping_status": "error",
                    "error_code": "fasthpocr_error",
                    "error_message": message,
                    "span_start": "",
                    "span_end": "",
                    "candidate_hpo_ids": "",
                    "recognizer": "FastHPOCR",
                }
            )
    annotation_seconds = time.perf_counter() - annotation_started

    predictions_path = args.output_dir / "fasthpocr_predictions.csv"
    _write_predictions(predictions_path, prediction_rows)
    predictions = load_prediction_sets(predictions_path, aliases)
    reference_groups = load_reference_groups(args.references, aliases)
    input_patient_ids = [patient_id for patient_id, _ in inputs]
    patient_ids = [patient_id for patient_id in input_patient_ids if patient_id in reference_groups]
    unscored_patient_ids = [
        patient_id for patient_id in input_patient_ids if patient_id not in reference_groups
    ]
    scores = score_reference_groups(
        predictions,
        reference_groups,
        patient_ids=patient_ids,
    )
    per_case_path, report_path = write_benchmark_report(
        scores,
        args.output_dir,
        predictions_path=predictions_path,
        references_path=args.references,
        ontology_path=args.ontology,
        input_path=args.input,
        metadata={
            "corpus": args.corpus,
            "recognizer": "FastHPOCR",
            "recognizer_version": str(index_manifest["fast_hpo_cr_version"]),
            "longest_match": str(args.longest_match).lower(),
            "reference_policy": "comma-delimited IDs are alternatives",
            "index_manifest_sha256": sha256_file(args.index_dir / "fasthpocr_index_manifest.json"),
        },
    )
    run_record = {
        "schema_version": "1.0",
        "corpus": args.corpus,
        "input_case_count": len(inputs),
        "scored_case_count": len(patient_ids),
        "unscored_input_case_ids": unscored_patient_ids,
        "mention_count": mention_count,
        "unique_prediction_count": sum(len(value) for value in predictions.values()),
        "error_count": len(errors),
        "errors": errors,
        "longest_match": args.longest_match,
        "index_reused": reused,
        "timing_seconds": {
            "index_build_or_validation": build_seconds,
            "annotation": annotation_seconds,
            "annotation_per_case": annotation_seconds / len(inputs) if inputs else 0.0,
        },
        "summary": summarize(scores),
        "provenance": {
            "input_sha256": sha256_file(args.input),
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "addons_sha256": sha256_file(args.addons) if args.addons else None,
            "index_sha256": index_manifest["index_sha256"],
            "predictions_sha256": sha256_file(predictions_path),
        },
    }
    run_path = args.output_dir / "fasthpocr_run.json"
    run_path.write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (per_case_path, report_path, run_path):
        restrict_owner(path)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary": run_record["summary"],
                "timing_seconds": run_record["timing_seconds"],
                "error_count": len(errors),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
