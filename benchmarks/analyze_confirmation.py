#!/usr/bin/env python3
"""Analyze the locked confirmation cohort and repeated live runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import _metrics
from rag_hpo.diagnostics import holm_adjust
from rag_hpo.statistics import paired_inference


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    precision, recall, f1 = _metrics(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": statistics.fmean(float(row["precision"]) for row in rows),
        "macro_recall": statistics.fmean(float(row["recall"]) for row in rows),
        "macro_f1": statistics.fmean(float(row["f1"]) for row in rows),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_intervals(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)  # noqa: S311
    samples: dict[str, list[float]] = {
        "micro_precision": [],
        "micro_recall": [],
        "micro_f1": [],
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
    }
    for _ in range(iterations):
        selected = rng.choices(rows, k=len(rows))
        aggregate = _aggregate(selected)
        for key in samples:
            aggregate_key = key.replace("micro_", "") if key.startswith("micro_") else key
            samples[key].append(float(aggregate[aggregate_key]))
    return {
        key: {
            "lower": _quantile(values, 0.025),
            "upper": _quantile(values, 0.975),
        }
        for key, values in samples.items()
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-report", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--discovery-report", type=Path)
    parser.add_argument("--repeat-report", type=Path, action="append", default=[])
    parser.add_argument("--historical-model", default="LLaMa 4-Scout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--iterations", type=int, default=100_000)
    args = parser.parse_args()

    current_report = json.loads(args.current_report.read_text(encoding="utf-8"))
    current = {str(row["patient_id"]): row for row in current_report["cases"]}
    cohort = json.loads(args.cohort_manifest.read_text(encoding="utf-8"))
    selected = [str(value) for value in cohort["selection"]["confirmation_case_ids"]]
    if set(current) != set(selected):
        parser.error("current report does not exactly match the locked confirmation cohort")

    with args.historical_metrics.open(encoding="utf-8-sig", newline="") as handle:
        historical_all = {
            str(row["patient_id"]): {
                **row,
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
            for row in csv.DictReader(handle)
            if row["sheet"] == "CSC Comparison Results" and row["model"] == args.historical_model
        }
    historical = {
        patient_id: historical_all[patient_id]
        for patient_id in selected
        if patient_id in historical_all
    }
    if set(historical) != set(selected):
        parser.error("historical metrics do not exactly match the locked cohort")

    current_rows = [current[patient_id] for patient_id in selected]
    historical_rows = [historical[patient_id] for patient_id in selected]
    paired_metrics = []
    raw_p_values = []
    for metric in ("precision", "recall", "f1"):
        differences = [
            float(current[patient_id][metric]) - float(historical[patient_id][metric])
            for patient_id in selected
        ]
        inference = paired_inference(
            differences,
            seed=args.seed,
            iterations=args.iterations,
        )
        paired_metrics.append({"metric": metric, **inference})
        raw_p_values.append(float(inference["two_sided_sign_flip_p"]))
    for row, adjusted in zip(
        paired_metrics,
        holm_adjust(raw_p_values),
        strict=True,
    ):
        row["holm_adjusted_two_sided_p"] = adjusted

    repeat_summary: dict[str, Any] | None = None
    if args.repeat_report:
        if len(args.repeat_report) < 3:
            parser.error("variability analysis requires at least three repeat reports")
        repeat_reports = [
            json.loads(path.read_text(encoding="utf-8")) for path in args.repeat_report
        ]
        repeat_ids = {str(value) for value in cohort["selection"]["repeat_case_ids"]}
        indexed = [
            {str(row["patient_id"]): row for row in report["cases"]} for report in repeat_reports
        ]
        if any(set(report) != repeat_ids for report in indexed):
            parser.error("repeat reports do not exactly match the locked repeat cohort")
        f1_ranges = []
        f1_standard_deviations = []
        jaccards = []
        for patient_id in sorted(repeat_ids, key=int):
            f1_values = [float(report[patient_id]["f1"]) for report in indexed]
            f1_ranges.append(max(f1_values) - min(f1_values))
            f1_standard_deviations.append(statistics.stdev(f1_values))
            prediction_sets = [set(report[patient_id]["predicted_ids"]) for report in indexed]
            for left_index in range(len(prediction_sets)):
                for right_index in range(left_index + 1, len(prediction_sets)):
                    jaccards.append(
                        _jaccard(
                            prediction_sets[left_index],
                            prediction_sets[right_index],
                        )
                    )
        repeat_summary = {
            "repeat_count": len(indexed),
            "case_count": len(repeat_ids),
            "mean_within_case_f1_range": statistics.fmean(f1_ranges),
            "maximum_within_case_f1_range": max(f1_ranges),
            "mean_within_case_f1_standard_deviation": statistics.fmean(f1_standard_deviations),
            "mean_pairwise_prediction_jaccard": statistics.fmean(jaccards),
            "report_hashes": [sha256_file(path) for path in args.repeat_report],
        }

    combined: dict[str, Any] | None = None
    if args.discovery_report is not None:
        discovery = json.loads(args.discovery_report.read_text(encoding="utf-8"))
        discovery_rows = list(discovery["cases"])
        discovery_ids = [str(row["patient_id"]) for row in discovery_rows]
        historical_combined_rows = [
            historical_all[patient_id] for patient_id in discovery_ids + selected
        ]
        combined = {
            "case_count": len(discovery_rows) + len(current_rows),
            "current_descriptive": _aggregate(discovery_rows + current_rows),
            "historical_descriptive": _aggregate(historical_combined_rows),
            "note": (
                "Descriptive only: the discovery cohort was used to form hypotheses "
                "and is not part of confirmatory inference."
            ),
        }

    current_aggregate = _aggregate(current_rows)
    historical_aggregate = _aggregate(historical_rows)
    report = {
        "schema_version": "1.0",
        "confirmation_case_count": len(selected),
        "current": current_aggregate,
        "historical": historical_aggregate,
        "current_bootstrap_ci95": _bootstrap_intervals(
            current_rows,
            seed=args.seed,
            iterations=args.iterations,
        ),
        "paired_case_inference": paired_metrics,
        "repeat_variability": repeat_summary,
        "combined_discovery_and_confirmation": combined,
        "metric_shift_arithmetic": {
            "predicted_positive_difference": (
                int(current_aggregate["tp"])
                + int(current_aggregate["fp"])
                - int(historical_aggregate["tp"])
                - int(historical_aggregate["fp"])
            ),
            "true_positive_difference": int(current_aggregate["tp"])
            - int(historical_aggregate["tp"]),
            "false_positive_difference": int(current_aggregate["fp"])
            - int(historical_aggregate["fp"]),
            "false_negative_difference": int(current_aggregate["fn"])
            - int(historical_aggregate["fn"]),
        },
        "provenance": {
            "current_report_sha256": sha256_file(args.current_report),
            "historical_metrics_sha256": sha256_file(args.historical_metrics),
            "cohort_manifest_sha256": sha256_file(args.cohort_manifest),
            "discovery_report_sha256": (
                sha256_file(args.discovery_report) if args.discovery_report is not None else None
            ),
        },
        "limitations": [
            "Historical per-case counts do not reveal the historical predicted IDs.",
            "The deprecated historical model cannot be rerun under identical conditions.",
            "Provider temperature zero does not guarantee bitwise deterministic inference.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
