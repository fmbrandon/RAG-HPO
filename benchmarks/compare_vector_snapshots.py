#!/usr/bin/env python3
"""Compare candidate coverage across HPO snapshots with identical vector rules."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

from rag_hpo.artifacts import load_artifacts, sha256_file
from rag_hpo.benchmark import load_hpo_aliases
from rag_hpo.diagnostics import distinct_candidates
from rag_hpo.embeddings import create_backend

HPO_ID = re.compile(r"HP:\d{7}")


def _parse_vector(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("vector must be LABEL=PATH")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", action="append", type=_parse_vector, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--current-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.current_report.read_text(encoding="utf-8"))
    selected_ids = {str(row["patient_id"]) for row in report["cases"]}
    false_negatives = {
        (str(row["patient_id"]), hpo_id)
        for row in report["cases"]
        for hpo_id in row["false_negative_ids"]
    }
    aliases = load_hpo_aliases(args.ontology)
    reference_rows = []
    with args.references.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            patient_id = str(row.get("Patient ID") or "").strip()
            raw_ids = HPO_ID.findall(str(row.get("hpo_term") or ""))
            if patient_id not in selected_ids or len(raw_ids) != 1:
                continue
            reference_rows.append(
                {
                    "patient_id": patient_id,
                    "hpo_id": aliases.get(raw_ids[0], raw_ids[0]),
                    "description": str(row.get("hpo_description") or "").strip(),
                }
            )
    backend = create_backend("sapbert")
    queries = backend.encode([row["description"] for row in reference_rows])
    snapshot_results = {}
    for label, vector_dir in args.vector:
        entries, matrix, manifest = load_artifacts(vector_dir)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(np.asarray(matrix, dtype=np.float32))
        scores, indices = index.search(
            np.asarray(queries, dtype=np.float32),
            min(512, len(entries)),
        )
        hits: Counter[str] = Counter()
        fn_hits: Counter[str] = Counter()
        fn_total = 0
        for row_index, reference in enumerate(reference_rows):
            candidates = distinct_candidates(
                entries,
                list(scores[row_index]),
                list(indices[row_index]),
                limit=32,
            )
            policy_sets = {
                "raw8": {candidate.hpo_id for candidate in candidates if candidate.raw_rank <= 8},
                "unique8": {
                    candidate.hpo_id for candidate in candidates if candidate.distinct_rank <= 8
                },
                "unique16": {
                    candidate.hpo_id for candidate in candidates if candidate.distinct_rank <= 16
                },
                "unique32": {candidate.hpo_id for candidate in candidates},
            }
            is_fn = (reference["patient_id"], reference["hpo_id"]) in false_negatives
            fn_total += int(is_fn)
            for policy, values in policy_sets.items():
                hit = reference["hpo_id"] in values
                hits[policy] += int(hit)
                if is_fn:
                    fn_hits[policy] += int(hit)
        total = len(reference_rows)
        snapshot_results[label] = {
            "ontology_sha256": manifest.hpo_sha256,
            "vector_manifest_sha256": sha256_file(vector_dir / "hpo_manifest.json"),
            "metadata_count": manifest.metadata_count,
            "all_reference_recall": {
                policy: count / total for policy, count in sorted(hits.items())
            },
            "false_negative_count": fn_total,
            "false_negative_recall": {
                policy: count / fn_total for policy, count in sorted(fn_hits.items())
            },
        }
    output = {
        "schema_version": "1.0",
        "reference_count": len(reference_rows),
        "snapshots": snapshot_results,
        "controlled_factors": {
            "embedding_model": backend.model_id,
            "embedding_revision": backend.revision,
            "metadata_rule": "current label, synonym, and HPO_addons construction",
            "query_text": "manual HPO description",
        },
        "interpretation_limit": (
            "This is a retrieval-coverage ablation. It does not include live extraction "
            "or final language-model mapping."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
