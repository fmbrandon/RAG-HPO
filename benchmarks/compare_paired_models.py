#!/usr/bin/env python3
"""Compare current per-case CSC F1 with a historical model using paired tests."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from rag_hpo.benchmark import _metrics
from rag_hpo.statistics import paired_inference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-report", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--historical-model", default="LLaMa 4-Scout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--iterations", type=int, default=100_000)
    args = parser.parse_args()

    current_report = json.loads(args.current_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    selected = manifest["selection"]["selected_case_ids"]
    current = {row["patient_id"]: row for row in current_report["cases"]}

    with args.historical_metrics.open(encoding="utf-8-sig", newline="") as handle:
        historical = {
            row["patient_id"]: row
            for row in csv.DictReader(handle)
            if row["sheet"] == "CSC Comparison Results"
            and row["model"] == args.historical_model
            and row["patient_id"] in selected
        }
    missing = [
        case_id for case_id in selected if case_id not in current or case_id not in historical
    ]
    if missing:
        parser.error(f"missing paired cases: {', '.join(missing)}")

    pairs = []
    for case_id in selected:
        new_row = current[case_id]
        old_row = historical[case_id]
        pairs.append(
            {
                "patient_id": case_id,
                "current_f1": float(new_row["f1"]),
                "historical_f1": float(old_row["f1"]),
                "f1_difference": float(new_row["f1"]) - float(old_row["f1"]),
                "current_tp": int(new_row["tp"]),
                "current_fp": int(new_row["fp"]),
                "current_fn": int(new_row["fn"]),
                "historical_tp": int(old_row["tp"]),
                "historical_fp": int(old_row["fp"]),
                "historical_fn": int(old_row["fn"]),
            }
        )

    def aggregate(prefix: str) -> dict[str, float | int]:
        tp = sum(int(row[f"{prefix}_tp"]) for row in pairs)
        fp = sum(int(row[f"{prefix}_fp"]) for row in pairs)
        fn = sum(int(row[f"{prefix}_fn"]) for row in pairs)
        precision, recall, f1 = _metrics(tp, fp, fn)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": statistics.fmean(float(row[f"{prefix}_f1"]) for row in pairs),
        }

    inference = paired_inference(
        [float(row["f1_difference"]) for row in pairs],
        seed=args.seed,
        iterations=args.iterations,
    )
    better = (
        float(inference["mean_difference_ci95_lower"]) > 0
        and float(inference["one_sided_sign_flip_p"]) < 0.05
    )
    report = {
        "schema_version": "1.0",
        "case_count": len(pairs),
        "current_model": current_report["provenance"]["model"],
        "historical_model": args.historical_model,
        "selection": manifest["selection"],
        "current": aggregate("current"),
        "historical": aggregate("historical"),
        "paired_inference": inference,
        "conclusion": {
            "criterion": (
                "better only when the paired mean-F1 95% bootstrap CI is entirely "
                "above zero and the one-sided paired sign-flip p-value is below 0.05"
            ),
            "current_scores_better": better,
        },
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_keys = ("current", "historical", "paired_inference", "conclusion")
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
