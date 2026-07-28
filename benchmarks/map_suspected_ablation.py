#!/usr/bin/env python3
"""Counterfactually map extracted Suspected rows without changing production policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rag_hpo.config import DEFAULT_BASE_URL, DEFAULT_MODEL, ProviderConfig, ResponseMode
from rag_hpo.models import HPOMapping
from rag_hpo.pipeline import AnnotationPipeline
from rag_hpo.provider import OpenAICompatibleProvider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = json.loads(args.predictions.read_text(encoding="utf-8"))
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        notes = {
            str(row.get("Case") or row.get("patient_id") or "").strip(): str(
                row.get("clinical_note") or ""
            )
            for row in csv.DictReader(handle)
        }
    suspected = [
        row
        for row in rows
        if row.get("category") == "Suspected" and row.get("mapping_status") == "not_mapped_category"
    ]
    config = ProviderConfig.from_env(
        base_url=args.base_url,
        model=args.model,
        response_mode=ResponseMode.STRICT,
    )
    mapped_rows = []
    with OpenAICompatibleProvider(config) as provider:
        pipeline = AnnotationPipeline(
            provider=provider,
            vector_dir=args.vector_dir,
            output_dir=args.output.parent / ".unused-pipeline-output",
            resume=False,
            keep_state=False,
            keep_raw_responses=False,
        )
        for row in suspected:
            phrase = str(row.get("phrase") or "")
            candidates = pipeline._candidates(phrase)
            payload = json.dumps(
                {
                    "phrase": phrase,
                    "original_context": notes.get(str(row.get("patient_id")), ""),
                    "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                },
                ensure_ascii=False,
            )
            mapping, _ = provider.request(
                system_message=pipeline.prompts["hpo_mapping"],
                user_message=payload,
                response_model=HPOMapping,
                temperature=0.0,
            )
            by_id = {candidate.hpo_id: candidate for candidate in candidates}
            selected = by_id.get(mapping.hpo_id or "")
            mapped_rows.append(
                {
                    **row,
                    "hpo_id": selected.hpo_id if selected else None,
                    "hpo_term": selected.term if selected else None,
                    "vector_score": selected.score if selected else None,
                    "mapping_status": "mapped" if selected else "no_candidate_fit",
                }
            )
    merged = rows + mapped_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "suspected_rows": len(suspected),
                "mapped_rows": sum(row["mapping_status"] == "mapped" for row in mapped_rows),
                "model": args.model,
                "raw_responses_retained": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
