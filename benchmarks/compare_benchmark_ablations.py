#!/usr/bin/env python3
"""Compare same-case benchmark reports with multiplicity correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.diagnostics import holm_adjust
from rag_hpo.statistics import paired_inference


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("value must be LABEL=PATH")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=_labeled_path, required=True)
    parser.add_argument("--alternative", type=_labeled_path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--iterations", type=int, default=100_000)
    args = parser.parse_args()

    baseline_label, baseline_path = args.baseline
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_cases = {str(row["patient_id"]): row for row in baseline["cases"]}
    comparisons = []
    p_values = []
    for label, path in args.alternative:
        report = json.loads(path.read_text(encoding="utf-8"))
        cases = {str(row["patient_id"]): row for row in report["cases"]}
        if set(cases) != set(baseline_cases):
            parser.error(f"{label} does not contain the same cases as the baseline")
        differences = [
            float(cases[patient_id]["f1"]) - float(baseline_cases[patient_id]["f1"])
            for patient_id in sorted(cases, key=int)
        ]
        inference = paired_inference(
            differences,
            seed=args.seed,
            iterations=args.iterations,
        )
        p_values.append(float(inference["one_sided_sign_flip_p"]))
        comparisons.append(
            {
                "label": label,
                "report_sha256": sha256_file(path),
                "summary": report["summary"],
                "paired_f1_vs_baseline": inference,
            }
        )
    for comparison, adjusted in zip(
        comparisons,
        holm_adjust(p_values),
        strict=True,
    ):
        comparison["holm_adjusted_one_sided_p"] = adjusted

    output = {
        "schema_version": "1.0",
        "baseline": {
            "label": baseline_label,
            "report_sha256": sha256_file(baseline_path),
            "summary": baseline["summary"],
        },
        "comparisons": comparisons,
        "criterion": (
            "Exploratory improvement requires a positive paired mean-F1 interval and "
            "Holm-adjusted one-sided p below 0.05. Confirmation remains separate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
