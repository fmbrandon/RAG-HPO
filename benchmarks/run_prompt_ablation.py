#!/usr/bin/env python3
"""Run an authorized benchmark cohort with a frozen historical prompt pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.config import DEFAULT_BASE_URL, DEFAULT_MODEL, ProviderConfig, ResponseMode
from rag_hpo.pipeline import AnnotationPipeline, load_csv_inputs
from rag_hpo.provider import OpenAICompatibleProvider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    prompt_data = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    required = {"phenotype_extraction", "hpo_mapping"}
    if missing := required - prompt_data.keys():
        parser.error(f"prompt file is missing: {', '.join(sorted(missing))}")
    rows, errors = load_csv_inputs(args.input)
    if errors:
        parser.error("ablation input contains invalid rows")
    config = ProviderConfig.from_env(
        base_url=args.base_url,
        model=args.model,
        response_mode=ResponseMode.STRICT,
    )
    with OpenAICompatibleProvider(config) as provider:
        pipeline = AnnotationPipeline(
            provider=provider,
            vector_dir=args.vector_dir,
            output_dir=args.output_dir,
            resume=args.resume,
            keep_state=False,
            keep_raw_responses=False,
        )
        pipeline.prompts = prompt_data
        results = pipeline.run(rows)
    summary = {
        "schema_version": "1.0",
        "input_rows": len(rows),
        "result_rows": len(results),
        "error_rows": sum(row.mapping_status == "error" for row in results),
        "prompt_sha256": sha256_file(args.prompt_file),
        "input_sha256": sha256_file(args.input),
        "vector_manifest_sha256": sha256_file(args.vector_dir / "hpo_manifest.json"),
        "model": args.model,
        "response_mode": ResponseMode.STRICT.value,
        "raw_responses_retained": False,
    }
    summary_path = args.output_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["error_rows"] == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
