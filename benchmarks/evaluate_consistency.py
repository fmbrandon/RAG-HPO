#!/usr/bin/env python3
"""Evaluate repeated subset runs without performing new inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    load_hpo_aliases,
    load_prediction_sets,
    load_reference_groups,
)
from rag_hpo.consistency import evaluate_consistency, load_dispositions


def _selection_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selection = raw.get("selection", raw)
    for key in (
        "repeat_case_ids",
        "selected_case_ids",
        "confirmation_case_ids",
        "case_ids",
    ):
        values = selection.get(key)
        if isinstance(values, list):
            return [str(value) for value in values]
    raise ValueError("selection manifest does not contain a recognized case-ID list")


def _safe_run_metadata(prediction_path: Path) -> dict[str, Any]:
    path = prediction_path.parent / "rag_hpo_run_manifest.json"
    if not path.exists():
        return {"manifest": "not-found"}
    raw = json.loads(path.read_text(encoding="utf-8"))
    provider = raw.get("provider") if isinstance(raw.get("provider"), dict) else {}
    return {
        "manifest_sha256": sha256_file(path),
        "artifact_manifest_sha256": raw.get("artifact_manifest_sha256"),
        "prompt_bundle_sha256": raw.get("prompt_bundle_sha256"),
        "config_sha256": raw.get("config_sha256"),
        "provider": {
            key: provider.get(key)
            for key in ("base_url", "model", "response_mode")
            if provider.get(key) is not None
        },
        "provider_usage": raw.get("provider_usage"),
        "run_attempt_count": raw.get("run_attempt_count"),
        "runtime": raw.get("runtime"),
        "inference_parameters": raw.get("inference_parameters"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure repeatability across two or more accepted-only subset runs."
    )
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-tolerance", type=float, default=0.02)
    parser.add_argument("--jaccard-threshold", type=float, default=0.85)
    parser.add_argument("--recurrence-threshold", type=float, default=0.90)
    args = parser.parse_args()
    if len(args.predictions) < 2:
        parser.error("repeat --predictions at least twice")

    aliases = load_hpo_aliases(args.ontology)
    references = load_reference_groups(args.references, aliases)
    patient_ids = _selection_ids(args.selection_manifest)
    prediction_sets = [
        load_prediction_sets(path, aliases, accepted_only=True) for path in args.predictions
    ]
    dispositions = [load_dispositions(path, aliases) for path in args.predictions]
    report = evaluate_consistency(
        prediction_sets,
        references,
        patient_ids=patient_ids,
        dispositions=dispositions,
        metric_tolerance=args.metric_tolerance,
        jaccard_threshold=args.jaccard_threshold,
        recurrence_threshold=args.recurrence_threshold,
    )
    report["provenance"] = {
        "prediction_runs": [
            {
                "name": path.name,
                "sha256": sha256_file(path),
                **_safe_run_metadata(path),
            }
            for path in args.predictions
        ],
        "references_sha256": sha256_file(args.references),
        "ontology_sha256": sha256_file(args.ontology),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
