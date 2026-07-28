#!/usr/bin/env python3
"""Freeze discovery-only score thresholds before inspecting confirmation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import _metrics


def _evaluate(
    rows: list[dict[str, str]],
    *,
    column: str,
    threshold: float,
    baseline_fn: int,
) -> dict[str, float | int]:
    retained = [row for row in rows if float(row[column] or 0.0) >= threshold]
    tp = sum(row["exact_status"] == "tp" for row in retained)
    fp = sum(row["exact_status"] == "fp" for row in retained)
    removed_tp = sum(
        row["exact_status"] == "tp" and float(row[column] or 0.0) < threshold for row in rows
    )
    fn = baseline_fn + removed_tp
    precision, recall, f1 = _metrics(tp, fp, fn)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--diagnostic-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.diagnostic_summary.read_text(encoding="utf-8"))
    baseline_fn = int(summary["strict_exact_set"]["micro"]["fn"])
    with args.ledger.open(encoding="utf-8", newline="") as handle:
        prediction_rows = [
            row
            for row in csv.DictReader(handle)
            if row["record_type"] == "prediction"
            and row["exact_status"] in {"tp", "fp"}
            and row["candidate_score"]
        ]
    grids = {}
    frozen = {}
    for column, maximum in (("candidate_score", 100), ("candidate_margin", 25)):
        values = [
            _evaluate(
                prediction_rows,
                column=column,
                threshold=index / 100,
                baseline_fn=baseline_fn,
            )
            for index in range(maximum + 1)
        ]
        grids[column] = values
        frozen[column] = max(
            values,
            key=lambda row: (
                float(row["f1"]),
                float(row["recall"]),
                -float(row["threshold"]),
            ),
        )
    output = {
        "schema_version": "1.0",
        "purpose": "discovery-only threshold calibration frozen before confirmation scoring",
        "prediction_count": len(prediction_rows),
        "baseline": summary["strict_exact_set"]["micro"],
        "frozen_thresholds": frozen,
        "grids": grids,
        "provenance": {
            "ledger_sha256": sha256_file(args.ledger),
            "diagnostic_summary_sha256": sha256_file(args.diagnostic_summary),
        },
        "interpretation_limit": (
            "These cutoffs are exploratory until applied unchanged to the locked "
            "confirmation cohort. They can remove predictions but cannot recover "
            "phenotypes that were never extracted or retrieved."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["frozen_thresholds"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
