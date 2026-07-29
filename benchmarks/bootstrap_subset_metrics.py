#!/usr/bin/env python3
"""Add deterministic case-bootstrap intervals to a locked subset report."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
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


def _selected_ids(manifest: dict[str, Any]) -> list[str]:
    selection = manifest.get("selection", manifest)
    for key in ("selected_case_ids", "confirmation_case_ids", "case_ids"):
        values = selection.get(key)
        if isinstance(values, list):
            return [str(value) for value in values]
    raise ValueError("selection manifest does not contain a recognized case-ID list")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    if args.iterations < 1_000:
        parser.error("--iterations must be at least 1000")

    report = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    rows = report.get("cases")
    if not isinstance(rows, list) or not rows:
        parser.error("benchmark report does not contain case rows")
    expected = _selected_ids(manifest)
    observed = [str(row["patient_id"]) for row in rows]
    if set(observed) != set(expected) or len(observed) != len(expected):
        parser.error("benchmark cases do not exactly match the locked subset")

    rng = random.Random(args.seed)  # noqa: S311
    distributions: dict[str, list[float]] = {
        "micro_precision": [],
        "micro_recall": [],
        "micro_f1": [],
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
    }
    for _ in range(args.iterations):
        sampled = rng.choices(rows, k=len(rows))
        aggregate = _aggregate(sampled)
        distributions["micro_precision"].append(float(aggregate["precision"]))
        distributions["micro_recall"].append(float(aggregate["recall"]))
        distributions["micro_f1"].append(float(aggregate["f1"]))
        distributions["macro_precision"].append(float(aggregate["macro_precision"]))
        distributions["macro_recall"].append(float(aggregate["macro_recall"]))
        distributions["macro_f1"].append(float(aggregate["macro_f1"]))

    result = {
        "schema_version": "1.0",
        "method": ("case-level nonparametric bootstrap with replacement; percentile 95% intervals"),
        "seed": args.seed,
        "iterations": args.iterations,
        "case_count": len(rows),
        "point_estimate": _aggregate(rows),
        "ci95": {
            key: {
                "lower": _quantile(values, 0.025),
                "upper": _quantile(values, 0.975),
            }
            for key, values in distributions.items()
        },
        "provenance": {
            "benchmark_report_sha256": sha256_file(args.benchmark_report),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
        },
        "interpretation": (
            "Intervals quantify sampling uncertainty within the locked subset. "
            "They do not account for corpus shift, gold-standard error, or model-run "
            "variability."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
