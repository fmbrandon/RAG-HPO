#!/usr/bin/env python3
"""Measure active-ID hybrid retrieval coverage against reference descriptions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from rag_hpo.artifacts import load_artifacts, sha256_file
from rag_hpo.benchmark import load_hpo_aliases
from rag_hpo.embeddings import create_backend
from rag_hpo.models import Candidate
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.registry import load_registry_bundle
from rag_hpo.retrieval import HybridCandidateRetriever

HPO_ID = re.compile(r"HP:\d{7}")
RRF_CONSTANT = 60


def _selection(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = raw.get("selection", raw)
    for key in ("selected_case_ids", "confirmation_case_ids", "case_ids"):
        if isinstance(value.get(key), list):
            return {str(item) for item in value[key]}
    raise ValueError(f"{path} does not contain a recognized case-ID list")


def _rerank(
    candidates: list[Candidate],
    *,
    dense_weight: float,
    exact_first: bool = False,
) -> list[str]:
    lexical_weight = 1.0 - dense_weight

    def score(candidate: Candidate) -> tuple[int, float, int, int, str]:
        dense_rank = candidate.dense_rank
        lexical_rank = candidate.lexical_rank
        lexical_score = candidate.lexical_score
        combined = 0.0
        if dense_rank is not None:
            combined += dense_weight / (RRF_CONSTANT + dense_rank)
        if lexical_rank is not None:
            combined += lexical_weight / (RRF_CONSTANT + lexical_rank)
        return (
            -int(bool(exact_first and lexical_score == 100.0)),
            -combined,
            dense_rank or 10**9,
            lexical_rank or 10**9,
            candidate.hpo_id,
        )

    return [candidate.hpo_id for candidate in sorted(candidates, key=score)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--private-detail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--analyze-weights",
        action="store_true",
        help="Report aggregate discovery-only RRF weight sensitivity.",
    )
    args = parser.parse_args()

    aliases = load_hpo_aliases(args.ontology)
    selected = _selection(args.selection)
    rows: list[dict[str, object]] = []
    with args.references.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            patient_id = str(raw.get("Patient ID") or "").strip()
            description = str(raw.get("hpo_description") or "").strip()
            acceptable = {
                aliases.get(value, value)
                for value in HPO_ID.findall(str(raw.get("hpo_term") or ""))
            }
            if (
                patient_id
                and description
                and acceptable
                and (selected is None or patient_id in selected)
            ):
                rows.append(
                    {
                        "patient_id": patient_id,
                        "description": description,
                        "acceptable_ids": acceptable,
                    }
                )
    if not rows:
        raise ValueError("no scorable reference descriptions were loaded")

    entries, matrix, manifest = load_artifacts(args.vector_dir)
    registry, registry_manifest, lexical_manifest = load_registry_bundle(args.vector_dir)
    backend = create_backend(manifest.embedding_backend, offline=args.offline)
    retriever = HybridCandidateRetriever(
        registry=registry,
        entries=entries,
        matrix=matrix,
        backend=backend,
    )
    predictions = retriever.retrieve_many(
        [str(row["description"]) for row in rows],
        distinct_limit=64,
    )
    hits: Counter[str] = Counter()
    detail: list[dict[str, object]] = []
    for row, candidates in zip(rows, predictions, strict=True):
        acceptable_ids = set(row["acceptable_ids"])
        current = _rerank(candidates, dense_weight=0.8)
        top16 = set(current[:16])
        top32 = set(current[:32])
        hit16 = bool(acceptable_ids & top16)
        hit32 = bool(acceptable_ids & top32)
        hits["top16"] += int(hit16)
        hits["top32"] += int(hit32)
        detail.append(
            {
                "patient_id": row["patient_id"],
                "description": row["description"],
                "acceptable_ids": "|".join(sorted(acceptable_ids)),
                "top16_hit": hit16,
                "top32_hit": hit32,
                "candidate_ids": "|".join(value.hpo_id for value in candidates),
            }
        )

    ensure_private_directory(args.private_detail.parent)
    with args.private_detail.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(detail[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(detail)
    restrict_owner(args.private_detail)
    total = len(rows)
    report = {
        "schema_version": "1.0",
        "reference_count": total,
        "top16": {
            "hits": hits["top16"],
            "recall": hits["top16"] / total,
            "minimum": 0.94,
            "passed": hits["top16"] / total >= 0.94,
        },
        "top32": {
            "hits": hits["top32"],
            "recall": hits["top32"] / total,
            "minimum": 0.95,
            "passed": hits["top32"] / total >= 0.95,
        },
        "provenance": {
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "vector_manifest_sha256": sha256_file(args.vector_dir / "hpo_manifest.json"),
            "registry_sha256": registry_manifest.registry_sha256,
            "lexical_sha256": lexical_manifest.lexical_sha256,
            "private_detail_sha256": sha256_file(args.private_detail),
        },
    }
    if args.analyze_weights:
        report["discovery_weight_sensitivity"] = {}
        for dense_weight in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0):
            weight_hits = Counter()
            for row, candidates in zip(rows, predictions, strict=True):
                acceptable_ids = set(row["acceptable_ids"])
                reranked = _rerank(candidates, dense_weight=dense_weight)
                weight_hits["top16"] += int(bool(acceptable_ids & set(reranked[:16])))
                weight_hits["top32"] += int(bool(acceptable_ids & set(reranked[:32])))
            report["discovery_weight_sensitivity"][str(dense_weight)] = {
                "top16": weight_hits["top16"] / total,
                "top32": weight_hits["top32"] / total,
            }
        for dense_weight in (0.8, 0.9, 0.95):
            weight_hits = Counter()
            for row, candidates in zip(rows, predictions, strict=True):
                acceptable_ids = set(row["acceptable_ids"])
                reranked = _rerank(
                    candidates,
                    dense_weight=dense_weight,
                    exact_first=True,
                )
                weight_hits["top16"] += int(bool(acceptable_ids & set(reranked[:16])))
                weight_hits["top32"] += int(bool(acceptable_ids & set(reranked[:32])))
            report["discovery_weight_sensitivity"][f"{dense_weight}-exact-first"] = {
                "top16": weight_hits["top16"] / total,
                "top32": weight_hits["top32"] / total,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["top16"]["passed"] and report["top32"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
