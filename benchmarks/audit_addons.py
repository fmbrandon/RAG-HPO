#!/usr/bin/env python3
"""Audit curated HPO add-on phrases against a pinned ontology."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.diagnostics import parse_obo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addons", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ontology = parse_obo(args.ontology)
    with args.addons.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    status_counts: Counter[str] = Counter()
    phrases: dict[str, set[str]] = defaultdict(set)
    silently_skipped = []
    for row_number, row in enumerate(rows, start=2):
        hpo_id = str(row.get("HP_ID") or "").strip()
        phrase = " ".join(str(row.get("info") or "").casefold().split())
        if phrase and hpo_id:
            phrases[phrase].add(hpo_id)
        raw_term = ontology.terms.get(hpo_id)
        if raw_term is None:
            status_counts["missing"] += 1
            silently_skipped.append({"row": row_number, "hpo_id": hpo_id})
        elif raw_term.obsolete and raw_term.replaced_by:
            status_counts["obsolete_with_replacement"] += 1
        elif raw_term.obsolete:
            status_counts["obsolete_without_replacement"] += 1
        else:
            status_counts["current"] += 1
    conflicts = [
        {"phrase": phrase, "hpo_ids": sorted(ids)}
        for phrase, ids in sorted(phrases.items())
        if len(ids) > 1
    ]
    report = {
        "schema_version": "1.0",
        "provenance": {
            "addons_sha256": sha256_file(args.addons),
            "ontology_sha256": sha256_file(args.ontology),
            "ontology_data_version": ontology.data_version,
        },
        "row_count": len(rows),
        "unique_normalized_phrase_count": len(phrases),
        "identifier_status_counts": dict(sorted(status_counts.items())),
        "conflicting_phrase_assignments": conflicts,
        "silently_skipped_by_current_builder": silently_skipped,
        "risk": (
            "The current builder accepts only IDs present exactly in the ontology and "
            "does not automatically retarget add-on rows through replaced_by mappings."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
