#!/usr/bin/env python3
"""Score normalized RAG-HPO predictions against one explicit reference corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    load_hpo_aliases,
    load_prediction_groups,
    load_prediction_sets,
    load_reference_groups,
    score_prediction_groups,
    score_reference_groups,
    summarize,
    write_benchmark_report,
)


def _selection_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selection = raw.get("selection", raw)
    for key in ("selected_case_ids", "confirmation_case_ids", "case_ids"):
        values = selection.get(key)
        if isinstance(values, list):
            return [str(value) for value in values]
    raise ValueError("selection manifest does not contain a recognized case-ID list")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute exact-set HPO metrics without mixing CSC and GSC corpora."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Score exactly the case IDs recorded in a locked selection manifest.",
    )
    parser.add_argument("--model", default="not-recorded")
    parser.add_argument("--prompt-version", default="not-recorded")
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Score accepted staged findings and exclude review/rejected rows.",
    )
    parser.add_argument(
        "--prediction-unit",
        choices=("candidate-set", "selected-id"),
        default="candidate-set",
        help=(
            "Score each bounded candidate set as one phenotype finding, or use the "
            "legacy selected-ID document policy."
        ),
    )
    args = parser.parse_args()
    if args.case_ids and args.selection_manifest is not None:
        parser.error("--case-id and --selection-manifest are mutually exclusive")
    try:
        case_ids = (
            _selection_ids(args.selection_manifest)
            if args.selection_manifest is not None
            else args.case_ids
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    aliases = load_hpo_aliases(args.ontology)
    references = load_reference_groups(args.references, aliases)
    if args.prediction_unit == "candidate-set":
        prediction_groups = load_prediction_groups(
            args.predictions,
            aliases,
            accepted_only=args.accepted_only,
        )
        scores = score_prediction_groups(
            prediction_groups,
            references,
            patient_ids=case_ids,
        )
    else:
        predictions = load_prediction_sets(
            args.predictions,
            aliases,
            accepted_only=args.accepted_only,
        )
        scores = score_reference_groups(predictions, references, patient_ids=case_ids)
    if not scores:
        parser.error("no benchmark cases were selected")
    csv_path, json_path = write_benchmark_report(
        scores,
        args.output_dir,
        predictions_path=args.predictions,
        references_path=args.references,
        ontology_path=args.ontology,
        metadata={
            "model": args.model,
            "prompt_version": args.prompt_version,
            "prediction_policy": ("accepted-only" if args.accepted_only else "all-mapped"),
            "prediction_unit": args.prediction_unit,
            "selection_manifest": (
                args.selection_manifest.name
                if args.selection_manifest is not None
                else "not-supplied"
            ),
            "selection_manifest_sha256": (
                sha256_file(args.selection_manifest)
                if args.selection_manifest is not None
                else "not-supplied"
            ),
        },
        input_path=args.input,
        prompt_path=args.prompt_file,
        vector_manifest_path=args.vector_manifest,
    )
    print(
        json.dumps(
            {
                "summary": summarize(scores),
                "per_case_csv": str(csv_path),
                "report_json": str(json_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
