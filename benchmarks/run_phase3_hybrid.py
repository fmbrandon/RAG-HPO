#!/usr/bin/env python3
"""Evaluate the recall-preserving Phase 3 verifier on a fixed cohort."""

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
    load_prediction_sets,
    load_reference_groups,
    score_reference_groups,
    summarize,
)
from rag_hpo.config import ProviderConfig
from rag_hpo.hybrid import HybridPredictionVerifier, ModelFinding
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.provider import OpenAICompatibleProvider, ProviderError
from rag_hpo.registry import load_registry_bundle

RETAINED_STATUSES = {"accepted", "retained_ambiguous"}


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    return (1 + beta_squared) * precision * recall / denominator if denominator else 0.0


def _summary(
    predictions: dict[str, set[str]],
    references: dict[str, list[set[str]]],
    patient_ids: list[str],
) -> dict[str, Any]:
    result = summarize(
        score_reference_groups(
            predictions,
            references,
            patient_ids=patient_ids,
        )
    )
    micro = result["micro"]
    micro["f0_5"] = _f_beta(float(micro["precision"]), float(micro["recall"]))
    return result


def _load_inputs(path: Path) -> dict[str, str]:
    inputs: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "clinical_note" not in reader.fieldnames:
            raise ValueError("input CSV requires a clinical_note column")
        for index, row in enumerate(reader, start=1):
            patient_id = str(
                row.get("patient_id") or row.get("Case") or row.get("Patient ID") or index
            ).strip()
            note = str(row.get("clinical_note") or "")
            if patient_id and note.strip():
                inputs[patient_id] = note
    return inputs


def _load_selection(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selection = raw.get("selection", raw)
    for key in ("selected_case_ids", "confirmation_case_ids", "case_ids"):
        values = selection.get(key)
        if isinstance(values, list):
            return [str(value) for value in values]
    raise ValueError(f"{path} does not contain a recognized case-ID list")


def _load_findings(
    paths: list[Path],
    aliases: dict[str, str],
) -> dict[str, list[ModelFinding]]:
    findings: dict[str, dict[str, ModelFinding]] = {}
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path} contains a non-object row")
            patient_id = str(row.get("patient_id") or "").strip()
            raw_hpo_id = str(row.get("hpo_id") or "").strip()
            hpo_id = aliases.get(raw_hpo_id, raw_hpo_id)
            if (
                not patient_id
                or not hpo_id
                or row.get("mapping_status") not in (None, "", "mapped")
            ):
                continue
            raw_score = row.get("vector_score")
            score = float(raw_score) if raw_score not in (None, "") else None
            findings.setdefault(patient_id, {}).setdefault(
                hpo_id,
                ModelFinding(
                    phrase=str(row.get("phrase") or "").strip(),
                    hpo_id=hpo_id,
                    vector_score=score,
                ),
            )
    return {
        patient_id: list(patient_findings.values())
        for patient_id, patient_findings in findings.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--model-predictions", type=Path, action="append", required=True)
    parser.add_argument("--fasthpocr-predictions", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live-verify", action="store_true")
    args = parser.parse_args()

    ensure_private_directory(args.output_dir)
    aliases = load_hpo_aliases(args.ontology)
    references = load_reference_groups(args.references, aliases)
    inputs = _load_inputs(args.input)
    selected_ids = _load_selection(args.selection)
    missing = [
        patient_id
        for patient_id in selected_ids
        if patient_id not in inputs or patient_id not in references
    ]
    if missing:
        raise ValueError(f"selected cases lack input or references: {missing}")
    model_findings = _load_findings(args.model_predictions, aliases)
    missing_predictions = [
        patient_id for patient_id in selected_ids if patient_id not in model_findings
    ]
    if missing_predictions:
        raise ValueError(f"selected cases lack model predictions: {missing_predictions}")
    fast_predictions = load_prediction_sets(args.fasthpocr_predictions, aliases)
    registry, registry_manifest, lexical_manifest = load_registry_bundle(args.registry_dir)

    provider = OpenAICompatibleProvider(ProviderConfig.from_env()) if args.live_verify else None
    verifier = HybridPredictionVerifier(registry, provider=provider)
    baseline = {
        patient_id: {finding.hpo_id for finding in model_findings[patient_id]}
        for patient_id in selected_ids
    }
    hybrid: dict[str, set[str]] = {}
    decision_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    provider_errors: Counter[str] = Counter()
    verification_calls = 0
    reviewed_count = 0
    approximate_input_tokens = 0
    started = time.perf_counter()
    try:
        for patient_id in selected_ids:
            findings = model_findings[patient_id]
            high_confidence = baseline[patient_id] & fast_predictions.get(patient_id, set())
            try:
                result = verifier.verify(
                    inputs[patient_id],
                    findings,
                    high_confidence_ids=high_confidence,
                )
            except ProviderError as exc:
                provider_errors[exc.code] += 1
                fallback = HybridPredictionVerifier(registry, provider=None)
                result = fallback.verify(
                    inputs[patient_id],
                    findings,
                    high_confidence_ids=high_confidence,
                )
            verification_calls += result.verification_call_count
            reviewed_count += result.reviewed_count
            approximate_input_tokens += result.approximate_input_tokens
            hybrid[patient_id] = {
                decision.hpo_id
                for decision in result.decisions
                if decision.status in RETAINED_STATUSES
            }
            for decision in result.decisions:
                status_counts[decision.status] += 1
                reason_counts[decision.reason] += 1
                decision_rows.append(
                    {
                        "patient_id": patient_id,
                        "mention_id": decision.mention_id,
                        "phrase": decision.phrase,
                        "hpo_id": decision.hpo_id,
                        "status": decision.status,
                        "reason": decision.reason,
                        "confidence": decision.confidence,
                        "assertion": (decision.assertion.status if decision.assertion else ""),
                    }
                )
    finally:
        if provider is not None:
            provider.close()
    elapsed = time.perf_counter() - started

    decisions_path = args.output_dir / "phase3_hybrid_decisions.csv"
    with decisions_path.open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "patient_id",
            "mention_id",
            "phrase",
            "hpo_id",
            "status",
            "reason",
            "confidence",
            "assertion",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(decision_rows)
    restrict_owner(decisions_path)

    predictions_path = args.output_dir / "phase3_hybrid_predictions.json"
    prediction_rows = [
        {
            "patient_id": patient_id,
            "hpo_id": hpo_id,
            "mapping_status": "mapped",
        }
        for patient_id in selected_ids
        for hpo_id in sorted(hybrid[patient_id])
    ]
    predictions_path.write_text(
        json.dumps(prediction_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(predictions_path)

    baseline_summary = _summary(baseline, references, selected_ids)
    hybrid_summary = _summary(hybrid, references, selected_ids)
    baseline_recall = float(baseline_summary["micro"]["recall"])
    observed_recall = float(hybrid_summary["micro"]["recall"])
    recall_gate = observed_recall >= 0.58 and baseline_recall - observed_recall <= 0.02
    report = {
        "schema_version": "1.0",
        "mode": "live_verification" if args.live_verify else "offline_fallback",
        "case_count": len(selected_ids),
        "recall_gate": {
            "minimum": 0.58,
            "maximum_drop_from_baseline": 0.02,
            "baseline": baseline_recall,
            "observed": observed_recall,
            "drop_from_baseline": baseline_recall - observed_recall,
            "passed": recall_gate,
        },
        "baseline": baseline_summary,
        "hybrid": hybrid_summary,
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "provider_error_counts": dict(sorted(provider_errors.items())),
        "verification": {
            "call_count": verification_calls,
            "reviewed_count": reviewed_count,
            "approximate_input_tokens": approximate_input_tokens,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "seconds_per_case": elapsed / len(selected_ids),
        },
        "provenance": {
            "input_sha256": sha256_file(args.input),
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "selection_sha256": sha256_file(args.selection),
            "model_prediction_sha256": {
                str(path): sha256_file(path) for path in args.model_predictions
            },
            "fasthpocr_prediction_sha256": sha256_file(args.fasthpocr_predictions),
            "registry_sha256": registry_manifest.registry_sha256,
            "lexical_sha256": lexical_manifest.lexical_sha256,
            "decisions_sha256": sha256_file(decisions_path),
            "predictions_sha256": sha256_file(predictions_path),
        },
    }
    report_path = args.output_dir / "phase3_hybrid_run.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if recall_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
