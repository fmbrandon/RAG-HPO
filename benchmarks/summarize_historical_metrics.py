#!/usr/bin/env python3
"""Create one corpus-aware summary matrix from historical row-level counts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import _metrics
from rag_hpo.privacy import ensure_private_directory, restrict_owner

SHEET_TO_CORPUS = {
    "Premium LLM Comparison": "Premium",
    "CSC Comparison Results": "CSC",
    "GSC Comparison Results": "GSC",
}
SUMMARY_COLUMNS = (
    "corpus",
    "model",
    "case_count",
    "tp",
    "fp",
    "fn",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sheet", "model", "patient_id", "tp", "fp", "fn", "precision", "recall", "f1"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("historical metric CSV does not match the expected schema")
    return rows


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        corpus = SHEET_TO_CORPUS.get(row["sheet"])
        if corpus is None:
            raise ValueError(f"unknown historical sheet: {row['sheet']}")
        grouped[(corpus, row["model"])].append(row)

    output: list[dict[str, Any]] = []
    for (corpus, model), values in sorted(grouped.items()):
        tp = sum(int(value["tp"]) for value in values)
        fp = sum(int(value["fp"]) for value in values)
        fn = sum(int(value["fn"]) for value in values)
        precision, recall, f1 = _metrics(tp, fp, fn)
        count = len(values)
        output.append(
            {
                "corpus": corpus,
                "model": model,
                "case_count": count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
                "macro_precision": sum(float(value["precision"]) for value in values) / count,
                "macro_recall": sum(float(value["recall"]) for value in values) / count,
                "macro_f1": sum(float(value["f1"]) for value in values) / count,
            }
        )
    return output


def paired_comparison(
    rows: list[dict[str, str]],
    *,
    corpus: str,
    first_model: str,
    second_model: str,
) -> dict[str, Any]:
    sheet = next(key for key, value in SHEET_TO_CORPUS.items() if value == corpus)
    by_model = {
        model: {
            row["patient_id"]: row
            for row in rows
            if row["sheet"] == sheet and row["model"] == model
        }
        for model in (first_model, second_model)
    }
    shared = sorted(
        set(by_model[first_model]) & set(by_model[second_model]),
        key=lambda value: (
            not value.isdigit(),
            int(value) if value.isdigit() else value,
        ),
    )
    summaries = summarize_rows(
        [
            row
            for model in (first_model, second_model)
            for patient_id, row in by_model[model].items()
            if patient_id in shared
        ]
    )
    wins = {first_model: 0, second_model: 0, "tie": 0}
    for patient_id in shared:
        first_f1 = float(by_model[first_model][patient_id]["f1"])
        second_f1 = float(by_model[second_model][patient_id]["f1"])
        if first_f1 > second_f1:
            wins[first_model] += 1
        elif second_f1 > first_f1:
            wins[second_model] += 1
        else:
            wins["tie"] += 1
    return {
        "corpus": corpus,
        "shared_case_count": len(shared),
        "models": summaries,
        "per_case_f1_wins": wins,
        "missing_from_first_model": sorted(
            set(by_model[second_model]) - set(by_model[first_model])
        ),
        "missing_from_second_model": sorted(
            set(by_model[first_model]) - set(by_model[second_model])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.historical_metrics)
    summary = summarize_rows(rows)
    ensure_private_directory(args.output_dir)
    csv_path = args.output_dir / "historical_model_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    restrict_owner(csv_path)

    report = {
        "schema_version": "1.0",
        "source_sha256": sha256_file(args.historical_metrics),
        "summary": summary,
        "paired_fast_hpo_cr_vs_llama4_scout": [
            paired_comparison(
                rows,
                corpus=corpus,
                first_model="FastHPOCR",
                second_model="LLaMa 4-Scout",
            )
            for corpus in ("CSC", "GSC")
        ],
        "limitations": [
            "Historical rows retain counts but not predicted IDs or text spans.",
            "Historical rows predate the confirmed alternative-ID scoring policy.",
            "CSC and GSC patient identifiers are corpus-local.",
        ],
    }
    report_path = args.output_dir / "historical_model_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(report_path)
    print(json.dumps(report["paired_fast_hpo_cr_vs_llama4_scout"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
