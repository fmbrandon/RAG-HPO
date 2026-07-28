#!/usr/bin/env python3
"""Audit vector metadata for obsolete IDs and exact lexical ambiguity."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rag_hpo.artifacts import load_artifacts, sha256_file
from rag_hpo.diagnostics import parse_obo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ontology = parse_obo(args.ontology)
    entries, _, manifest = load_artifacts(args.vector_dir)

    sources: Counter[str] = Counter()
    id_status: Counter[str] = Counter()
    status_ids: dict[str, set[str]] = defaultdict(set)
    phrases: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        sources[entry.source] += 1
        term = ontology.terms.get(entry.hp_id)
        if term is None:
            status = "missing"
        elif term.obsolete and term.replaced_by:
            status = "obsolete_with_replacement"
        elif term.obsolete:
            status = "obsolete_without_replacement"
        else:
            status = "current"
        id_status[status] += 1
        status_ids[status].add(entry.hp_id)
        phrases[" ".join(entry.phrase.casefold().split())].add(entry.hp_id)
    ambiguous = {phrase: ids for phrase, ids in phrases.items() if len(ids) > 1}
    report = {
        "schema_version": "1.0",
        "provenance": {
            "vector_manifest_sha256": sha256_file(args.vector_dir / "hpo_manifest.json"),
            "ontology_sha256": sha256_file(args.ontology),
            "ontology_data_version": ontology.data_version,
            "embedding_model": manifest.embedding_model,
            "embedding_revision": manifest.embedding_revision,
        },
        "metadata_entry_count": len(entries),
        "source_counts": dict(sorted(sources.items())),
        "entry_identifier_status_counts": dict(sorted(id_status.items())),
        "unique_identifier_status_counts": {
            status: len(ids) for status, ids in sorted(status_ids.items())
        },
        "exact_cross_id_phrase_ambiguity": {
            "phrase_count": len(ambiguous),
            "entry_assignments": sum(len(ids) for ids in ambiguous.values()),
        },
        "risk": (
            "Obsolete labels and add-on phrases remain retrievable and can consume raw "
            "top-k positions before candidates are deduplicated by HPO ID."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
