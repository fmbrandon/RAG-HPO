#!/usr/bin/env python3
"""Replay and score the frozen Phase 3 model-verification decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    CaseScore,
    load_hpo_aliases,
    load_prediction_sets,
    load_reference_groups,
    score_reference_groups,
    summarize,
)
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.statistics import paired_inference


def _f_beta(score: CaseScore, beta: float = 0.5) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * score.precision + score.recall
    return (1 + beta_squared) * score.precision * score.recall / denominator if denominator else 0.0


def _summary(scores: list[CaseScore]) -> dict[str, Any]:
    result = summarize(scores)
    micro = result["micro"]
    precision = float(micro["precision"])
    recall = float(micro["recall"])
    denominator = 0.25 * precision + recall
    micro["f0_5"] = 1.25 * precision * recall / denominator if denominator else 0.0
    return result


def _selection(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selection = raw.get("selection", raw)
    for key in ("selected_case_ids", "confirmation_case_ids", "case_ids"):
        value = selection.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    raise ValueError(f"{path} does not contain a recognized case-ID list")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--iterations", type=int, default=100_000)
    args = parser.parse_args()

    aliases = load_hpo_aliases(args.ontology)
    references = load_reference_groups(args.references, aliases)
    baseline = load_prediction_sets(args.baseline_predictions, aliases)
    patient_ids = _selection(args.selection)
    rejected: dict[str, set[str]] = {}
    with args.decisions.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("status") == "rejected"
                and row.get("reason") == "model_review_unsupported_high"
            ):
                patient_id = str(row.get("patient_id") or "")
                raw_hpo_id = str(row.get("hpo_id") or "")
                hpo_id = aliases.get(raw_hpo_id, raw_hpo_id)
                if patient_id and hpo_id:
                    rejected.setdefault(patient_id, set()).add(hpo_id)

    hybrid = {
        patient_id: set(baseline.get(patient_id, set())) - rejected.get(patient_id, set())
        for patient_id in patient_ids
    }
    baseline_scores = score_reference_groups(baseline, references, patient_ids=patient_ids)
    hybrid_scores = score_reference_groups(hybrid, references, patient_ids=patient_ids)
    baseline_summary = _summary(baseline_scores)
    hybrid_summary = _summary(hybrid_scores)

    metrics: dict[str, Callable[[CaseScore], float]] = {
        "precision": lambda score: score.precision,
        "recall": lambda score: score.recall,
        "f1": lambda score: score.f1,
        "f0_5": _f_beta,
    }
    paired = {
        name: paired_inference(
            [
                metric(hybrid_score) - metric(baseline_score)
                for baseline_score, hybrid_score in zip(baseline_scores, hybrid_scores, strict=True)
            ],
            seed=args.seed,
            iterations=args.iterations,
        )
        for name, metric in metrics.items()
    }

    ensure_private_directory(args.output_dir)
    predictions_path = args.output_dir / "phase3_revised_predictions.json"
    rows = [
        {
            "patient_id": patient_id,
            "hpo_id": hpo_id,
            "mapping_status": "mapped",
        }
        for patient_id in patient_ids
        for hpo_id in sorted(hybrid[patient_id])
    ]
    predictions_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(predictions_path)

    baseline_micro = baseline_summary["micro"]
    hybrid_micro = hybrid_summary["micro"]
    baseline_recall = float(baseline_micro["recall"])
    hybrid_recall = float(hybrid_micro["recall"])
    recall_drop = baseline_recall - hybrid_recall
    report = {
        "schema_version": "1.0",
        "policy": (
            "Retain local assertion flags; remove only model-reviewed "
            "unsupported/high-confidence predictions."
        ),
        "case_count": len(patient_ids),
        "baseline": baseline_summary,
        "hybrid": hybrid_summary,
        "rejection_yield": {
            "removed_prediction_count": sum(len(value) for value in rejected.values()),
            "removed_true_positives": int(baseline_micro["tp"]) - int(hybrid_micro["tp"]),
            "removed_false_positives": int(baseline_micro["fp"]) - int(hybrid_micro["fp"]),
        },
        "gates": {
            "recall": {
                "minimum": 0.58,
                "maximum_drop_from_baseline": 0.02,
                "observed": hybrid_recall,
                "drop_from_baseline": recall_drop,
                "passed": hybrid_recall >= 0.58 and recall_drop <= 0.02,
            },
            "default_precision": {
                "minimum": 0.80,
                "observed": hybrid_micro["precision"],
                "passed": float(hybrid_micro["precision"]) >= 0.80,
            },
            "paired_f0_5": {
                "requires_positive_ci95": True,
                "passed": float(paired["f0_5"]["mean_difference_ci95_lower"]) > 0,
            },
        },
        "paired_case_inference": paired,
        "provenance": {
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions),
            "decisions_sha256": sha256_file(args.decisions),
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "selection_sha256": sha256_file(args.selection),
            "revised_predictions_sha256": sha256_file(predictions_path),
        },
    }
    report_path = args.output_dir / "phase3_revised_comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
