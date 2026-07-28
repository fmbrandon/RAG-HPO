#!/usr/bin/env python3
"""Score normalized RAG-HPO predictions against one explicit reference corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hpo.benchmark import (
    load_hpo_aliases,
    load_prediction_sets,
    load_reference_sets,
    score_sets,
    summarize,
    write_benchmark_report,
)


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
    parser.add_argument("--model", default="not-recorded")
    parser.add_argument("--prompt-version", default="not-recorded")
    args = parser.parse_args()

    aliases = load_hpo_aliases(args.ontology)
    predictions = load_prediction_sets(args.predictions, aliases)
    references = load_reference_sets(args.references, aliases)
    scores = score_sets(predictions, references, patient_ids=args.case_ids)
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
