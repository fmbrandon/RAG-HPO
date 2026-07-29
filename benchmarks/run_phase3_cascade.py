#!/usr/bin/env python3
"""Evaluate deterministic Phase 3 cascade policies without retaining full notes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    load_hpo_aliases,
    load_reference_groups,
    score_reference_groups,
    summarize,
)
from rag_hpo.cascade import PrecisionCascade
from rag_hpo.fasthpocr import FastHPORecognizer
from rag_hpo.lexical import NativeLexicalRecognizer
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.registry import load_registry_bundle

POLICIES = (
    "exact_unique_raw",
    "exact_unique_asserted",
    "native_unique_asserted",
    "cascade_offline",
)


def _inputs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "clinical_note" not in reader.fieldnames:
            raise ValueError("input CSV requires a clinical_note column")
        for index, row in enumerate(reader, start=1):
            patient_id = str(
                row.get("patient_id") or row.get("Case") or row.get("Patient ID") or index
            ).strip()
            rows.append((patient_id, str(row.get("clinical_note") or "")))
    return rows


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    return (1 + beta_squared) * precision * recall / denominator if denominator else 0.0


def _policy_summary(
    predictions: dict[str, set[str]],
    references: dict[str, list[set[str]]],
    patient_ids: list[str],
) -> dict[str, Any]:
    summary = summarize(
        score_reference_groups(
            predictions,
            references,
            patient_ids=patient_ids,
        )
    )
    micro = summary["micro"]
    micro["f0_5"] = _f_beta(float(micro["precision"]), float(micro["recall"]))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--fasthpocr-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ensure_private_directory(args.output_dir)
    registry, registry_manifest, lexical_manifest = load_registry_bundle(args.registry_dir)
    recognizer = NativeLexicalRecognizer(registry)
    cascade = PrecisionCascade(registry, recognizer=recognizer)
    fast = (
        FastHPORecognizer(
            args.fasthpocr_index,
            longest_match=True,
            registry=registry,
        )
        if args.fasthpocr_index
        else None
    )
    inputs = _inputs(args.input)
    aliases = load_hpo_aliases(args.ontology)
    references = load_reference_groups(args.references, aliases)
    eligible = [patient_id for patient_id, _ in inputs if patient_id in references]
    unscored = [patient_id for patient_id, _ in inputs if patient_id not in references]

    predictions = {policy: {patient_id: set() for patient_id in eligible} for policy in POLICIES}
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    mention_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for patient_id, note in inputs:
        native = recognizer.recognize(note)
        fast_annotations = fast.annotate(note) if fast else []
        result = cascade.annotate(note, fast_annotations=fast_annotations)
        for mention in native:
            if patient_id not in predictions["exact_unique_raw"]:
                continue
            if mention.evidence == "exact" and len(mention.candidate_hpo_ids) == 1:
                predictions["exact_unique_raw"][patient_id].add(mention.candidate_hpo_ids[0])
        for decision in result.decisions:
            status_counts[decision.status] += 1
            reason_counts[decision.reason] += 1
            assertion_counts[decision.assertion.status] += 1
            evidence_counts[decision.evidence] += 1
            mention_rows.append(
                {
                    "patient_id": patient_id,
                    "phrase": decision.phrase,
                    "span_start": decision.start_offset,
                    "span_end": decision.end_offset,
                    "candidate_hpo_ids": ",".join(decision.candidate_hpo_ids),
                    "selected_hpo_id": decision.selected_hpo_id or "",
                    "status": decision.status,
                    "reason": decision.reason,
                    "methods": ",".join(decision.methods),
                    "evidence": decision.evidence,
                    "assertion": decision.assertion.status,
                    "assertion_cue": decision.assertion.cue or "",
                }
            )
            if patient_id not in predictions["cascade_offline"]:
                continue
            if (
                decision.evidence == "exact"
                and len(decision.candidate_hpo_ids) == 1
                and decision.assertion.accepted
            ):
                predictions["exact_unique_asserted"][patient_id].add(decision.candidate_hpo_ids[0])
            if (
                len(decision.candidate_hpo_ids) == 1
                and decision.assertion.accepted
                and "native" in decision.methods
            ):
                predictions["native_unique_asserted"][patient_id].add(decision.candidate_hpo_ids[0])
            if decision.selected_hpo_id and decision.status in (
                "accepted",
                "accepted_verified",
            ):
                predictions["cascade_offline"][patient_id].add(decision.selected_hpo_id)
    elapsed = time.perf_counter() - started

    mention_path = args.output_dir / "phase3_mentions.csv"
    with mention_path.open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "patient_id",
            "phrase",
            "span_start",
            "span_end",
            "candidate_hpo_ids",
            "selected_hpo_id",
            "status",
            "reason",
            "methods",
            "evidence",
            "assertion",
            "assertion_cue",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(mention_rows)
    restrict_owner(mention_path)

    prediction_hashes: dict[str, str] = {}
    for policy, values in predictions.items():
        path = args.output_dir / f"{policy}_predictions.json"
        rows = [
            {"patient_id": patient_id, "hpo_id": hp_id, "mapping_status": "mapped"}
            for patient_id in eligible
            for hp_id in sorted(values[patient_id])
        ]
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        restrict_owner(path)
        prediction_hashes[policy] = sha256_file(path)

    report = {
        "schema_version": "1.0",
        "mode": "offline_no_model",
        "input_case_count": len(inputs),
        "scored_case_count": len(eligible),
        "unscored_input_case_ids": unscored,
        "elapsed_seconds": elapsed,
        "seconds_per_case": elapsed / len(inputs) if inputs else 0.0,
        "mention_count": len(mention_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "assertion_counts": dict(sorted(assertion_counts.items())),
        "evidence_counts": dict(sorted(evidence_counts.items())),
        "policies": {
            policy: _policy_summary(values, references, eligible)
            for policy, values in predictions.items()
        },
        "provenance": {
            "input_sha256": sha256_file(args.input),
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "registry_sha256": registry_manifest.registry_sha256,
            "lexical_sha256": lexical_manifest.lexical_sha256,
            "fasthpocr_index_sha256": (
                sha256_file(args.fasthpocr_index) if args.fasthpocr_index else None
            ),
            "prediction_sha256": prediction_hashes,
            "mentions_sha256": sha256_file(mention_path),
        },
    }
    report_path = args.output_dir / "phase3_cascade_run.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
