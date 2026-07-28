#!/usr/bin/env python3
"""Reconstruct the original lineage-expanded vector corpus without redundant vectors."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np

from rag_hpo.artifacts import sha256_file
from rag_hpo.diagnostics import parse_obo
from rag_hpo.embeddings import create_backend
from rag_hpo.privacy import ensure_private_directory, restrict_owner

ROOT_ID = "HP:0000001"
PARENTHESES = re.compile(r"\s*\([^)]*\)\s*")


def _legacy_clean(value: str) -> str:
    value = PARENTHESES.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return re.sub(r"[^\w\s]+$", "", value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-expanded-count", type=int)
    args = parser.parse_args()
    ontology = parse_obo(args.ontology)

    @cache
    def path_count(hpo_id: str) -> int:
        if hpo_id == ROOT_ID:
            return 1
        term = ontology.terms.get(hpo_id)
        if term is None or not term.parents:
            return 1
        return sum(path_count(parent) for parent in term.parents)

    collapsed: dict[tuple[str, str], dict[str, object]] = {}
    source_counts: defaultdict[str, int] = defaultdict(int)
    expanded_count = 0
    for hpo_id, term in sorted(ontology.terms.items()):
        if not term.label:
            continue
        repeats = path_count(hpo_id)
        for source, phrase in (
            ("hpo-label", term.label),
            *(("hpo-synonym", synonym) for synonym in term.synonyms),
        ):
            cleaned = _legacy_clean(phrase.title())
            if not cleaned:
                continue
            key = (hpo_id, cleaned)
            existing = collapsed.setdefault(
                key,
                {
                    "hpo_id": hpo_id,
                    "phrase": cleaned,
                    "term": term.label,
                    "source": source,
                    "repeat_count": 0,
                },
            )
            existing["repeat_count"] = int(existing["repeat_count"]) + repeats
            expanded_count += repeats
            source_counts[source] += repeats

    if args.expected_expanded_count is not None and expanded_count != args.expected_expanded_count:
        parser.error(
            f"expanded count {expanded_count} does not match {args.expected_expanded_count}"
        )

    rows = list(collapsed.values())
    backend = create_backend("sapbert")
    vectors = backend.encode([str(row["phrase"]) for row in rows])
    ensure_private_directory(args.artifact_dir)
    metadata_path = args.artifact_dir / "legacy_retrieval_meta.json"
    vector_path = args.artifact_dir / "legacy_retrieval_vectors.npz"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "ontology_data_version": ontology.data_version,
                "ontology_sha256": sha256_file(args.ontology),
                "embedding_model": backend.model_id,
                "embedding_revision": backend.revision,
                "expanded_entry_count": expanded_count,
                "collapsed_entries": rows,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    np.savez_compressed(vector_path, emb=np.asarray(vectors, dtype=np.float32))
    restrict_owner(metadata_path)
    restrict_owner(vector_path)

    profile = {
        "schema_version": "1.0",
        "ontology_data_version": ontology.data_version,
        "ontology_sha256": sha256_file(args.ontology),
        "term_count": len(ontology.terms),
        "expanded_entry_count": expanded_count,
        "collapsed_entry_count": len(rows),
        "unique_phrase_count": len({str(row["phrase"]) for row in rows}),
        "expanded_source_counts": dict(sorted(source_counts.items())),
        "embedding_model": backend.model_id,
        "embedding_revision": backend.revision,
        "private_artifact_hashes": {
            "metadata_sha256": sha256_file(metadata_path),
            "vectors_sha256": sha256_file(vector_path),
        },
        "reconstruction_note": (
            "Each label or synonym is repeated once per root-to-term lineage path, "
            "matching the original notebook's 245,916-row construction. Identical "
            "vectors are collapsed with repeat_count for efficient retrieval simulation."
        ),
    }
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
