from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from rag_hpo.benchmark import score_reference_groups, summarize

_DISPOSITION_PRIORITY = {"accepted": 0, "review": 1, "rejected": 2, "absent": 3}


def load_dispositions(path: Path, aliases: dict[str, str]) -> dict[tuple[str, str], str]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("prediction JSON must contain a list")
        rows = raw
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    output: dict[tuple[str, str], str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        patient_id = str(row.get("patient_id") or "").strip()
        raw_hpo_id = str(row.get("hpo_id") or "").strip()
        if not patient_id or not raw_hpo_id:
            continue
        hpo_id = aliases.get(raw_hpo_id, raw_hpo_id)
        status = str(row.get("review_status") or "accepted").strip()
        if status not in _DISPOSITION_PRIORITY:
            status = "review"
        key = (patient_id, hpo_id)
        prior = output.get(key)
        if prior is None or _DISPOSITION_PRIORITY[status] < _DISPOSITION_PRIORITY[prior]:
            output[key] = status
    return output


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def evaluate_consistency(
    prediction_sets: list[dict[str, set[str]]],
    references: dict[str, list[set[str]]],
    *,
    patient_ids: list[str],
    dispositions: list[dict[tuple[str, str], str]] | None = None,
    metric_tolerance: float = 0.02,
    jaccard_threshold: float = 0.85,
    recurrence_threshold: float = 0.90,
) -> dict[str, Any]:
    if len(prediction_sets) < 2:
        raise ValueError("at least two prediction runs are required")
    run_scores = [
        score_reference_groups(values, references, patient_ids=patient_ids)
        for values in prediction_sets
    ]
    summaries = [summarize(values) for values in run_scores]
    pairwise: list[dict[str, Any]] = []
    for left_index, right_index in combinations(range(len(prediction_sets)), 2):
        case_values = [
            _jaccard(
                prediction_sets[left_index].get(patient_id, set()),
                prediction_sets[right_index].get(patient_id, set()),
            )
            for patient_id in patient_ids
        ]
        pairwise.append(
            {
                "left_run": left_index + 1,
                "right_run": right_index + 1,
                "mean_case_jaccard": sum(case_values) / len(case_values) if case_values else 1.0,
                "minimum_case_jaccard": min(case_values) if case_values else 1.0,
            }
        )

    run_id_sets = [
        {
            (patient_id, hpo_id)
            for patient_id in patient_ids
            for hpo_id in values.get(patient_id, set())
        }
        for values in prediction_sets
    ]
    all_ids = set.union(*run_id_sets) if run_id_sets else set()
    stable_ids = set.intersection(*run_id_sets) if run_id_sets else set()
    recurrence = len(stable_ids) / len(all_ids) if all_ids else 1.0
    metric_ranges = {
        metric: max(summary["micro"][metric] for summary in summaries)
        - min(summary["micro"][metric] for summary in summaries)
        for metric in ("precision", "recall", "f1")
    }
    per_case_ranges: list[float] = []
    for case_index, _patient_id in enumerate(patient_ids):
        values: list[float] = []
        for scores in run_scores:
            value = scores[case_index].f1
            if value is None:
                raise ValueError("consistency scoring received an unscorable case")
            values.append(value)
        per_case_ranges.append(max(values) - min(values))

    disposition_agreement: float | None = None
    if dispositions is not None:
        selected_patients = set(patient_ids)
        keys = {key for value in dispositions for key in value if key[0] in selected_patients}
        agreements = [
            len({value.get(key, "absent") for value in dispositions}) == 1 for key in keys
        ]
        disposition_agreement = sum(agreements) / len(agreements) if agreements else 1.0

    mean_jaccard = (
        sum(value["mean_case_jaccard"] for value in pairwise) / len(pairwise) if pairwise else 1.0
    )
    gates = {
        "micro_metric_range_within_tolerance": all(
            value <= metric_tolerance for value in metric_ranges.values()
        ),
        "mean_pairwise_case_jaccard": mean_jaccard >= jaccard_threshold,
        "all_run_id_recurrence": recurrence >= recurrence_threshold,
    }
    return {
        "schema_version": "1.0",
        "run_count": len(prediction_sets),
        "case_count": len(patient_ids),
        "predeclared_thresholds": {
            "maximum_micro_metric_range": metric_tolerance,
            "minimum_mean_pairwise_case_jaccard": jaccard_threshold,
            "minimum_all_run_id_recurrence": recurrence_threshold,
        },
        "runs": summaries,
        "pairwise": pairwise,
        "mean_pairwise_case_jaccard": mean_jaccard,
        "all_run_id_recurrence": recurrence,
        "stable_accepted_id_count": len(stable_ids),
        "unique_accepted_id_count": len(all_ids),
        "micro_metric_ranges": metric_ranges,
        "mean_per_case_f1_range": (
            sum(per_case_ranges) / len(per_case_ranges) if per_case_ranges else 0.0
        ),
        "maximum_per_case_f1_range": max(per_case_ranges) if per_case_ranges else 0.0,
        "disposition_agreement": disposition_agreement,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
