#!/usr/bin/env python3
"""Compare benchmark identifiers and lexical ambiguity across HPO snapshots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.diagnostics import OntologySnapshot, parse_obo
from rag_hpo.privacy import ensure_private_directory, restrict_owner


def _parse_snapshot(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("snapshot must be LABEL=PATH")
    return label, Path(path)


def _benchmark_ids(references: Path, predictions: Path | None) -> dict[str, set[str]]:
    ids: dict[str, set[str]] = defaultdict(set)
    with references.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hpo_id = str(row.get("hpo_term") or "").strip()
            if hpo_id:
                ids[hpo_id].add("reference")
    if predictions is not None:
        if predictions.suffix.lower() == ".json":
            rows: list[dict[str, Any]] = json.loads(predictions.read_text(encoding="utf-8"))
        else:
            with predictions.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        for row in rows:
            hpo_id = str(row.get("hpo_id") or "").strip()
            if hpo_id:
                ids[hpo_id].add("prediction")
    return ids


def _status(snapshot: OntologySnapshot, hpo_id: str) -> tuple[str, str]:
    normalized = snapshot.normalize(hpo_id)
    raw_term = snapshot.terms.get(hpo_id)
    if raw_term is not None and raw_term.obsolete:
        status = (
            "obsolete_with_replacement" if raw_term.replaced_by else "obsolete_without_replacement"
        )
        return status, normalized
    term = snapshot.terms.get(normalized)
    if term is None:
        return "missing", normalized
    if hpo_id != normalized:
        return "alternate_id", normalized
    return "current", normalized


def _lexical_ambiguity(snapshot: OntologySnapshot) -> dict[str, Any]:
    phrases: dict[str, set[str]] = defaultdict(set)
    for hpo_id, term in snapshot.terms.items():
        if term.obsolete:
            continue
        for phrase in (term.label, *term.synonyms):
            normalized = " ".join(phrase.casefold().split())
            if normalized:
                phrases[normalized].add(hpo_id)
    ambiguous = {phrase: ids for phrase, ids in phrases.items() if len(ids) > 1}
    return {
        "unique_normalized_phrases": len(phrases),
        "ambiguous_phrase_count": len(ambiguous),
        "ambiguous_id_assignments": sum(len(ids) for ids in ambiguous.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="append", type=_parse_snapshot, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--private-detail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.snapshot) < 2:
        parser.error("at least two snapshots are required")
    labels = [label for label, _ in args.snapshot]
    if len(set(labels)) != len(labels):
        parser.error("snapshot labels must be unique")

    snapshots = {label: (path, parse_obo(path)) for label, path in args.snapshot}
    id_sources = _benchmark_ids(args.references, args.predictions)
    benchmark_ids = sorted(id_sources)
    baseline_label = labels[0]
    baseline = snapshots[baseline_label][1]
    detail: list[dict[str, Any]] = []
    changes: dict[str, Counter[str]] = {label: Counter() for label in labels[1:]}
    for hpo_id in benchmark_ids:
        baseline_status, baseline_normalized = _status(baseline, hpo_id)
        baseline_term = baseline.terms.get(baseline_normalized)
        for label in labels:
            _, snapshot = snapshots[label]
            status, normalized = _status(snapshot, hpo_id)
            term = snapshot.terms.get(normalized)
            row = {
                "input_hpo_id": hpo_id,
                "sources": "|".join(sorted(id_sources[hpo_id])),
                "snapshot": label,
                "data_version": snapshot.data_version,
                "status": status,
                "normalized_hpo_id": normalized,
                "label": term.label if term else "",
                "definition": term.definition if term else "",
                "synonyms": "|".join(term.synonyms) if term else "",
                "parents": "|".join(term.parents) if term else "",
                "replaced_by": "|".join(term.replaced_by) if term else "",
            }
            detail.append(row)
            if label == baseline_label:
                continue
            if status != baseline_status:
                changes[label]["status_changed"] += 1
            if normalized != baseline_normalized:
                changes[label]["normalized_id_changed"] += 1
            if (term.label if term else "") != (baseline_term.label if baseline_term else ""):
                changes[label]["label_changed"] += 1
            if set(term.synonyms if term else ()) != set(
                baseline_term.synonyms if baseline_term else ()
            ):
                changes[label]["synonyms_changed"] += 1
            if set(term.parents if term else ()) != set(
                baseline_term.parents if baseline_term else ()
            ):
                changes[label]["parents_changed"] += 1
            if (term.definition if term else "") != (
                baseline_term.definition if baseline_term else ""
            ):
                changes[label]["definition_changed"] += 1

    ensure_private_directory(args.private_detail.parent)
    with args.private_detail.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(detail)
    restrict_owner(args.private_detail)

    report = {
        "schema_version": "1.0",
        "baseline_snapshot": baseline_label,
        "benchmark_id_count": len(benchmark_ids),
        "snapshots": {
            label: {
                "data_version": snapshot.data_version,
                "sha256": sha256_file(path),
                "term_count": len(snapshot.terms),
                "benchmark_id_status": dict(
                    sorted(Counter(_status(snapshot, value)[0] for value in benchmark_ids).items())
                ),
                "reference_id_status": dict(
                    sorted(
                        Counter(
                            _status(snapshot, value)[0]
                            for value in benchmark_ids
                            if "reference" in id_sources[value]
                        ).items()
                    )
                ),
                "prediction_id_status": dict(
                    sorted(
                        Counter(
                            _status(snapshot, value)[0]
                            for value in benchmark_ids
                            if "prediction" in id_sources[value]
                        ).items()
                    )
                ),
                "lexical_ambiguity": _lexical_ambiguity(snapshot),
            }
            for label, (path, snapshot) in snapshots.items()
        },
        "changes_from_baseline": {
            label: dict(sorted(counts.items())) for label, counts in changes.items()
        },
        "private_detail": {
            "row_count": len(detail),
            "note": "Full labels, definitions, synonyms, and parents are outside Git.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
