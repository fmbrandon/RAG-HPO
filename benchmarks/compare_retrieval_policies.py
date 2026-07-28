#!/usr/bin/env python3
"""Compare current distinct-candidate policies with reconstructed legacy retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag_hpo.artifacts import load_artifacts, sha256_file
from rag_hpo.benchmark import load_hpo_aliases
from rag_hpo.diagnostics import distinct_candidates, holm_adjust
from rag_hpo.embeddings import create_backend
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.statistics import paired_inference


def _legacy_candidates(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    indices: np.ndarray,
    phrase: str,
) -> list[str]:
    tokens = set(re.findall(r"\w+", phrase.casefold()))
    raw_consumed = 0
    seen: set[str] = set()
    output: list[str] = []
    for score, index in zip(scores, indices, strict=True):
        if index < 0 or raw_consumed >= 500:
            break
        row = rows[int(index)]
        repetitions = int(row["repeat_count"])
        available = min(repetitions, 500 - raw_consumed)
        raw_consumed += available
        hpo_id = str(row["hpo_id"])
        if hpo_id in seen:
            continue
        info_tokens = set(re.findall(r"\w+", str(row["phrase"]).casefold()))
        if tokens & info_tokens or float(score) >= 0.35 or len(output) < 15:
            seen.add(hpo_id)
            output.append(hpo_id)
            if len(output) == 20:
                break
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--current-report", type=Path, required=True)
    parser.add_argument("--current-vector-dir", type=Path, required=True)
    parser.add_argument("--legacy-artifact-dir", type=Path, required=True)
    parser.add_argument("--private-detail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--iterations", type=int, default=100_000)
    args = parser.parse_args()

    aliases = load_hpo_aliases(args.ontology)
    current_report = json.loads(args.current_report.read_text(encoding="utf-8"))
    false_negatives = {
        (str(case["patient_id"]), hpo_id)
        for case in current_report["cases"]
        for hpo_id in case["false_negative_ids"]
    }
    selected_ids = {str(case["patient_id"]) for case in current_report["cases"]}
    reference_rows: list[dict[str, str]] = []
    with args.references.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            patient_id = str(row.get("Patient ID") or "").strip()
            hpo_id = aliases.get(str(row.get("hpo_term") or "").strip())
            description = str(row.get("hpo_description") or "").strip()
            if patient_id in selected_ids and hpo_id and description:
                reference_rows.append(
                    {
                        "patient_id": patient_id,
                        "hpo_id": hpo_id,
                        "description": description,
                    }
                )

    backend = create_backend("sapbert")
    queries = backend.encode([row["description"] for row in reference_rows])

    entries, current_matrix, current_manifest = load_artifacts(args.current_vector_dir)
    current_index = faiss.IndexFlatIP(current_matrix.shape[1])
    current_index.add(np.asarray(current_matrix, dtype=np.float32))
    current_scores, current_indices = current_index.search(
        np.asarray(queries, dtype=np.float32), min(512, len(entries))
    )

    legacy_metadata_path = args.legacy_artifact_dir / "legacy_retrieval_meta.json"
    legacy_vector_path = args.legacy_artifact_dir / "legacy_retrieval_vectors.npz"
    legacy_metadata = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
    legacy_rows: list[dict[str, Any]] = legacy_metadata["collapsed_entries"]
    with np.load(legacy_vector_path) as archive:
        legacy_matrix = np.asarray(archive["emb"], dtype=np.float32)
    legacy_index = faiss.IndexFlatIP(legacy_matrix.shape[1])
    legacy_index.add(legacy_matrix)
    legacy_scores, legacy_indices = legacy_index.search(
        np.asarray(queries, dtype=np.float32), min(4096, len(legacy_rows))
    )

    policies = ("current_raw8", "current_unique8", "current_unique16", "current_unique32", "legacy")
    hits: Counter[str] = Counter()
    fn_hits: Counter[str] = Counter()
    fn_total = 0
    detail: list[dict[str, Any]] = []
    for row_index, reference in enumerate(reference_rows):
        current = distinct_candidates(
            entries,
            list(current_scores[row_index]),
            list(current_indices[row_index]),
            limit=64,
        )
        candidate_sets = {
            "current_raw8": {candidate.hpo_id for candidate in current if candidate.raw_rank <= 8},
            "current_unique8": {
                candidate.hpo_id for candidate in current if candidate.distinct_rank <= 8
            },
            "current_unique16": {
                candidate.hpo_id for candidate in current if candidate.distinct_rank <= 16
            },
            "current_unique32": {
                candidate.hpo_id for candidate in current if candidate.distinct_rank <= 32
            },
            "legacy": set(
                _legacy_candidates(
                    legacy_rows,
                    legacy_scores[row_index],
                    legacy_indices[row_index],
                    reference["description"],
                )
            ),
        }
        is_fn = (reference["patient_id"], reference["hpo_id"]) in false_negatives
        if is_fn:
            fn_total += 1
        result: dict[str, Any] = {
            **reference,
            "current_false_negative": is_fn,
        }
        for policy in policies:
            hit = reference["hpo_id"] in candidate_sets[policy]
            result[f"{policy}_hit"] = hit
            hits[policy] += int(hit)
            if is_fn:
                fn_hits[policy] += int(hit)
        detail.append(result)

    ensure_private_directory(args.private_detail.parent)
    with args.private_detail.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(detail)
    restrict_owner(args.private_detail)

    case_totals: Counter[str] = Counter()
    case_hits: dict[str, Counter[str]] = defaultdict(Counter)
    for row in detail:
        patient_id = str(row["patient_id"])
        case_totals[patient_id] += 1
        for policy in policies:
            case_hits[patient_id][policy] += int(bool(row[f"{policy}_hit"]))
    comparisons = []
    for policy in policies[1:]:
        differences = [
            case_hits[patient_id][policy] / case_totals[patient_id]
            - case_hits[patient_id]["current_raw8"] / case_totals[patient_id]
            for patient_id in sorted(case_totals, key=int)
        ]
        inference = paired_inference(
            differences,
            seed=args.seed,
            iterations=args.iterations,
        )
        comparisons.append(
            {
                "policy": policy,
                "baseline": "current_raw8",
                "paired_case_recall": inference,
            }
        )
    adjusted = holm_adjust(
        [
            float(comparison["paired_case_recall"]["one_sided_sign_flip_p"])
            for comparison in comparisons
        ]
    )
    for comparison, adjusted_p in zip(comparisons, adjusted, strict=True):
        comparison["holm_adjusted_one_sided_p"] = adjusted_p

    total = len(reference_rows)
    report = {
        "schema_version": "1.0",
        "reference_count": total,
        "current_false_negative_count": fn_total,
        "all_reference_oracle_recall": {policy: hits[policy] / total for policy in policies},
        "false_negative_oracle_recall": {policy: fn_hits[policy] / fn_total for policy in policies},
        "hits": {
            "all_references": dict(hits),
            "current_false_negatives": dict(fn_hits),
        },
        "paired_policy_comparisons": comparisons,
        "provenance": {
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "current_report_sha256": sha256_file(args.current_report),
            "current_vector_manifest_sha256": sha256_file(
                args.current_vector_dir / "hpo_manifest.json"
            ),
            "current_embedding_model": current_manifest.embedding_model,
            "current_embedding_revision": current_manifest.embedding_revision,
            "legacy_metadata_sha256": sha256_file(legacy_metadata_path),
            "legacy_vectors_sha256": sha256_file(legacy_vector_path),
        },
        "interpretation_limit": (
            "Oracle recall asks whether the manual ID is retrievable from its manual "
            "description. It isolates candidate coverage but does not measure extraction "
            "or the language model's final choice."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
