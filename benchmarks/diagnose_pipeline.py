#!/usr/bin/env python3
"""Trace exact-set benchmark errors through extraction, retrieval, and mapping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rapidfuzz import fuzz

from rag_hpo.artifacts import load_artifacts, sha256_file
from rag_hpo.benchmark import (
    _metrics,
    load_hpo_aliases,
    load_prediction_sets,
    load_reference_sets,
    score_sets,
    summarize,
)
from rag_hpo.diagnostics import (
    OntologyRelation,
    RankedCandidate,
    distinct_candidates,
    hierarchy_match,
    parse_obo,
)
from rag_hpo.embeddings import create_backend
from rag_hpo.privacy import ensure_private_directory, restrict_owner


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError("prediction JSON must contain a list of objects")
        return value
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_inputs(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("Case") or row.get("patient_id") or "").strip(): str(
                row.get("clinical_note") or ""
            )
            for row in csv.DictReader(handle)
        }


def _read_reference_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            patient_id = str(row.get("Patient ID") or "").strip()
            if patient_id and row.get("hpo_term"):
                output[patient_id].append({key: str(value or "") for key, value in row.items()})
    return output


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(max(fuzz.token_set_ratio(left, right), fuzz.partial_ratio(left, right)))


def _best_extraction(
    description: str, rows: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    candidates = [
        (row, _similarity(description, str(row.get("phrase") or "")))
        for row in rows
        if row.get("phrase")
    ]
    if not candidates:
        return None, 0.0
    return max(candidates, key=lambda item: (item[1], str(item[0].get("phrase") or "")))


def _best_relation(
    predicted_ids: set[str], reference_id: str, ontology: Any
) -> tuple[str | None, OntologyRelation]:
    priority = {
        "predicted_descendant": 0,
        "predicted_ancestor": 1,
        "sibling": 2,
        "unrelated": 3,
    }
    options = [
        (predicted_id, ontology.relation(predicted_id, reference_id))
        for predicted_id in predicted_ids
    ]
    if not options:
        return None, OntologyRelation("unrelated", None)
    return min(
        options,
        key=lambda item: (
            priority.get(item[1].relation, 4),
            item[1].distance if item[1].distance is not None else 10**9,
            item[0],
        ),
    )


def _candidate_map(
    texts: list[str],
    *,
    entries: list[Any],
    matrix: np.ndarray,
    raw_search: int,
) -> dict[str, list[RankedCandidate]]:
    if not texts:
        return {}
    backend = create_backend("sapbert")
    queries = backend.encode(texts)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(np.asarray(matrix, dtype=np.float32))
    scores, indices = index.search(
        np.asarray(queries, dtype=np.float32), min(raw_search, len(entries))
    )
    return {
        text: distinct_candidates(
            entries,
            list(scores[row_index]),
            list(indices[row_index]),
            limit=64,
        )
        for row_index, text in enumerate(texts)
    }


def _candidate_by_id(candidates: list[RankedCandidate], hpo_id: str) -> RankedCandidate | None:
    return next((candidate for candidate in candidates if candidate.hpo_id == hpo_id), None)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_private_directory(path.parent)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    restrict_owner(path)


def _hierarchy_summary(
    scores: list[Any],
    ontology: Any,
    *,
    max_distance: int,
) -> dict[str, Any]:
    tp = fp = fn = 0
    relation_counts: Counter[str] = Counter()
    for score in scores:
        matches = hierarchy_match(
            score.predicted_ids,
            score.reference_ids,
            ontology,
            max_distance=max_distance,
        )
        tp += len(matches)
        fp += len(score.predicted_ids) - len(matches)
        fn += len(score.reference_ids) - len(matches)
        relation_counts.update(relation.relation for _, _, relation in matches)
    precision, recall, f1 = _metrics(tp, fp, fn)
    return {
        "max_parent_child_distance": max_distance,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_relation_counts": dict(sorted(relation_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--sanitized-output", type=Path, required=True)
    parser.add_argument("--raw-search", type=int, default=256)
    args = parser.parse_args()

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selection = manifest["selection"]
    selected_values = selection.get(
        "selected_case_ids",
        selection.get("confirmation_case_ids", []),
    )
    patient_ids = [str(value) for value in selected_values]
    if not patient_ids:
        parser.error("selection manifest does not contain case IDs")
    raw_rows = _read_rows(args.predictions)
    rows_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        patient_id = str(row.get("patient_id") or "").strip()
        if patient_id in patient_ids:
            rows_by_patient[patient_id].append(row)
    notes = _read_inputs(args.input)
    reference_rows = _read_reference_rows(args.references)
    aliases = load_hpo_aliases(args.ontology)
    predictions = load_prediction_sets(args.predictions, aliases)
    references = load_reference_sets(args.references, aliases)
    scores = score_sets(predictions, references, patient_ids=patient_ids)
    ontology = parse_obo(args.ontology)
    entries, matrix, vector_manifest = load_artifacts(args.vector_dir)

    diagnostic_texts = sorted(
        {
            str(row.get("phrase") or "").strip()
            for patient_id in patient_ids
            for row in rows_by_patient[patient_id]
            if row.get("phrase")
        }
        | {
            row.get("hpo_description", "").strip()
            for patient_id in patient_ids
            for row in reference_rows[patient_id]
            if row.get("hpo_description")
        }
    )
    candidate_map = _candidate_map(
        diagnostic_texts,
        entries=entries,
        matrix=matrix,
        raw_search=args.raw_search,
    )

    ledger: list[dict[str, Any]] = []
    review_pairs: list[dict[str, Any]] = []
    fn_stages: Counter[str] = Counter()
    fp_classes: Counter[str] = Counter()
    unique_in_raw_eight: list[int] = []
    near_tie_counts: list[int] = []
    oracle_hits: dict[str, int] = {
        "raw_1": 0,
        "raw_8": 0,
        "distinct_8": 0,
        "distinct_16": 0,
        "distinct_32": 0,
        "distinct_64": 0,
    }
    reference_total = 0

    for score in scores:
        patient_id = score.patient_id
        case_rows = rows_by_patient[patient_id]
        predicted_ids = set(score.predicted_ids)
        reference_ids = set(score.reference_ids)
        note = notes.get(patient_id, "")
        descriptions = {
            ontology.normalize(row["hpo_term"]): row.get("hpo_description", "").strip()
            for row in reference_rows[patient_id]
        }

        for row in case_rows:
            phrase = str(row.get("phrase") or "").strip()
            if not phrase:
                continue
            full_candidates = candidate_map[phrase]
            raw_eight = [candidate for candidate in full_candidates if candidate.raw_rank <= 8]
            unique_in_raw_eight.append(len(raw_eight))
            if raw_eight:
                near_tie_counts.append(
                    sum(candidate.score >= raw_eight[0].score - 0.02 for candidate in raw_eight)
                )

        for reference_id in sorted(reference_ids):
            reference_total += 1
            description = descriptions.get(reference_id, "")
            oracle = candidate_map.get(description, [])
            oracle_candidate = _candidate_by_id(oracle, reference_id)
            if oracle_candidate is not None:
                if oracle_candidate.raw_rank == 1:
                    oracle_hits["raw_1"] += 1
                if oracle_candidate.raw_rank <= 8:
                    oracle_hits["raw_8"] += 1
                for size in (8, 16, 32, 64):
                    if oracle_candidate.distinct_rank <= size:
                        oracle_hits[f"distinct_{size}"] += 1

            if reference_id in predicted_ids:
                related_prediction = reference_id
                relation = OntologyRelation("exact", 0)
            else:
                related_prediction, relation = _best_relation(predicted_ids, reference_id, ontology)
            aligned, alignment_score = _best_extraction(description, case_rows)
            category = str(aligned.get("category") or "") if aligned else ""
            phrase = str(aligned.get("phrase") or "") if aligned else ""
            selected_id = (
                ontology.normalize(str(aligned.get("hpo_id") or ""))
                if aligned and aligned.get("hpo_id")
                else ""
            )
            candidates = candidate_map.get(phrase, [])
            candidate = _candidate_by_id(candidates, reference_id)
            candidate_margin = (
                candidates[0].score - candidates[1].score if len(candidates) > 1 else None
            )
            raw_eight_ids = {value.hpo_id for value in candidates if value.raw_rank <= 8}

            if reference_id in predicted_ids:
                stage = "mapped_exact"
                confidence = "high"
            elif relation.relation in {"predicted_ancestor", "predicted_descendant"}:
                stage = "exact_scoring_disagreement"
                confidence = "high"
            elif aligned is None or alignment_score < 75:
                stage = "absent_extraction"
                confidence = "low" if alignment_score >= 60 else "medium"
            elif category != "Abnormal":
                stage = "excluded_category"
                confidence = "medium"
            elif reference_id in raw_eight_ids:
                stage = "mapper_choice_or_rejection"
                confidence = "medium"
            elif candidate is not None:
                stage = "retrieval_breadth"
                confidence = "medium"
            else:
                stage = "embedding_retrieval_miss"
                confidence = "medium"
            if reference_id not in predicted_ids:
                fn_stages[stage] += 1

            description_folded = description.casefold().strip()
            note_folded = note.casefold()
            note_score = _similarity(description, note)
            explicit_evidence = (
                "exact_text"
                if description_folded and description_folded in note_folded
                else "likely"
                if note_score >= 90
                else "uncertain"
            )
            ledger.append(
                {
                    "record_type": "reference",
                    "patient_id": patient_id,
                    "reference_id": reference_id,
                    "predicted_id": related_prediction or selected_id,
                    "reference_description": description,
                    "predicted_phrase": phrase,
                    "category": category,
                    "exact_status": "tp" if reference_id in predicted_ids else "fn",
                    "primary_classification": stage,
                    "classification_confidence": confidence,
                    "ontology_relation": relation.relation,
                    "ontology_distance": relation.distance if relation.distance is not None else "",
                    "extraction_alignment_score": round(alignment_score, 3),
                    "explicit_evidence": explicit_evidence,
                    "note_match_score": round(note_score, 3),
                    "candidate_raw_rank": candidate.raw_rank if candidate else "",
                    "candidate_distinct_rank": candidate.distinct_rank if candidate else "",
                    "candidate_score": candidate.score if candidate else "",
                    "candidate_margin": (
                        round(candidate_margin, 6) if candidate_margin is not None else ""
                    ),
                    "selected_id": selected_id,
                }
            )

        for predicted_id in sorted(predicted_ids):
            if predicted_id in reference_ids:
                classification = "true_positive"
                relation = OntologyRelation("exact", 0)
                related_reference = predicted_id
            else:
                relation_options = [
                    (reference_id, ontology.relation(predicted_id, reference_id))
                    for reference_id in reference_ids
                ]
                related_reference, relation = min(
                    relation_options,
                    key=lambda item: (
                        0
                        if item[1].relation == "predicted_descendant"
                        else 1
                        if item[1].relation == "predicted_ancestor"
                        else 2
                        if item[1].relation == "sibling"
                        else 3,
                        item[1].distance if item[1].distance is not None else 10**9,
                    ),
                )
                if relation.relation == "predicted_descendant":
                    classification = "greater_specificity"
                elif relation.relation == "predicted_ancestor":
                    classification = "lesser_specificity"
                elif relation.relation == "sibling":
                    classification = "related_sibling"
                else:
                    selected_rows = [
                        row
                        for row in case_rows
                        if ontology.normalize(str(row.get("hpo_id") or "")) == predicted_id
                    ]
                    phrase = str(selected_rows[0].get("phrase") or "") if selected_rows else ""
                    best_reference_score = max(
                        (_similarity(phrase, value) for value in descriptions.values()),
                        default=0.0,
                    )
                    classification = (
                        "wrong_mapping"
                        if best_reference_score >= 85
                        else "unresolved_possible_manual_omission"
                    )
                fp_classes[classification] += 1

            selected_rows = [
                row
                for row in case_rows
                if ontology.normalize(str(row.get("hpo_id") or "")) == predicted_id
            ]
            selected_row = selected_rows[0] if selected_rows else {}
            phrase = str(selected_row.get("phrase") or "")
            candidates = candidate_map.get(phrase, [])
            selected_candidate = _candidate_by_id(candidates, predicted_id)
            margin = candidates[0].score - candidates[1].score if len(candidates) > 1 else None
            ledger.append(
                {
                    "record_type": "prediction",
                    "patient_id": patient_id,
                    "reference_id": related_reference,
                    "predicted_id": predicted_id,
                    "reference_description": descriptions.get(related_reference, ""),
                    "predicted_phrase": phrase,
                    "category": str(selected_row.get("category") or ""),
                    "exact_status": "tp" if predicted_id in reference_ids else "fp",
                    "primary_classification": classification,
                    "classification_confidence": (
                        "high"
                        if classification
                        in {"true_positive", "greater_specificity", "lesser_specificity"}
                        else "low"
                    ),
                    "ontology_relation": relation.relation,
                    "ontology_distance": relation.distance if relation.distance is not None else "",
                    "extraction_alignment_score": "",
                    "explicit_evidence": "",
                    "note_match_score": "",
                    "candidate_raw_rank": (
                        selected_candidate.raw_rank if selected_candidate else ""
                    ),
                    "candidate_distinct_rank": (
                        selected_candidate.distinct_rank if selected_candidate else ""
                    ),
                    "candidate_score": (
                        selected_candidate.score
                        if selected_candidate
                        else selected_row.get("vector_score") or ""
                    ),
                    "candidate_margin": round(margin, 6) if margin is not None else "",
                    "selected_id": predicted_id,
                }
            )

            if predicted_id not in reference_ids and classification in {
                "greater_specificity",
                "lesser_specificity",
                "related_sibling",
                "wrong_mapping",
            }:
                review_id = hashlib.sha256(
                    f"{patient_id}|{predicted_id}|{related_reference}".encode()
                ).hexdigest()[:12]
                reference_term = ontology.terms.get(related_reference)
                predicted_term = ontology.terms.get(predicted_id)
                concepts = [
                    (
                        related_reference,
                        reference_term.label if reference_term else "",
                    ),
                    (
                        predicted_id,
                        predicted_term.label if predicted_term else phrase,
                    ),
                ]
                if int(review_id[-1], 16) % 2:
                    concepts.reverse()
                review_pairs.append(
                    {
                        "review_id": review_id,
                        "patient_id": patient_id,
                        "clinical_note": note,
                        "concept_a_id": concepts[0][0],
                        "concept_a_label": concepts[0][1],
                        "concept_b_id": concepts[1][0],
                        "concept_b_label": concepts[1][1],
                        "ontology_relation": relation.relation,
                        "ontology_distance": relation.distance or "",
                        "review_decision": "",
                        "review_notes": "",
                    }
                )

    ensure_private_directory(args.private_output_dir)
    _write_csv(args.private_output_dir / "stage_error_ledger.csv", ledger)
    _write_csv(args.private_output_dir / "blinded_review_packet.csv", review_pairs)

    prediction_rows = [row for row in ledger if row["record_type"] == "prediction"]

    def score_summary(status: str, column: str) -> dict[str, float | int]:
        values = [
            float(row[column])
            for row in prediction_rows
            if row["exact_status"] == status and row[column] != ""
        ]
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    strict_summary = summarize(scores)
    sanitized = {
        "schema_version": "1.0",
        "cohort": {
            "case_count": len(patient_ids),
            "selected_case_ids": patient_ids,
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
        },
        "provenance": {
            "input_sha256": sha256_file(args.input),
            "predictions_sha256": sha256_file(args.predictions),
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "ontology_data_version": ontology.data_version,
            "vector_manifest_sha256": sha256_file(args.vector_dir / "hpo_manifest.json"),
            "embedding_model": vector_manifest.embedding_model,
            "embedding_revision": vector_manifest.embedding_revision,
        },
        "strict_exact_set": strict_summary,
        "hierarchy_sensitivity": [
            _hierarchy_summary(scores, ontology, max_distance=distance) for distance in (1, 2)
        ],
        "false_negative_stage_counts": dict(sorted(fn_stages.items())),
        "false_positive_class_counts": dict(sorted(fp_classes.items())),
        "candidate_retrieval": {
            "reference_count": reference_total,
            "manual_description_oracle_recall": {
                key: count / reference_total for key, count in oracle_hits.items()
            },
            "manual_description_oracle_hits": oracle_hits,
            "distinct_ids_in_current_raw_top8": {
                "observation_count": len(unique_in_raw_eight),
                "minimum": min(unique_in_raw_eight),
                "median": statistics.median(unique_in_raw_eight),
                "maximum": max(unique_in_raw_eight),
                "distribution": dict(sorted(Counter(unique_in_raw_eight).items())),
            },
            "distinct_ids_within_0_02_of_top_score": {
                "minimum": min(near_tie_counts),
                "median": statistics.median(near_tie_counts),
                "maximum": max(near_tie_counts),
                "distribution": dict(sorted(Counter(near_tie_counts).items())),
            },
            "selected_prediction_scores": {
                "true_positive": score_summary("tp", "candidate_score"),
                "false_positive": score_summary("fp", "candidate_score"),
            },
            "selected_prediction_margins": {
                "true_positive": score_summary("tp", "candidate_margin"),
                "false_positive": score_summary("fp", "candidate_margin"),
            },
        },
        "private_artifacts": {
            "stage_error_ledger_rows": len(ledger),
            "blinded_review_rows": len(review_pairs),
            "note": "Stored outside Git; paths and note text are intentionally omitted.",
        },
        "limitations": [
            "Extraction-to-reference alignment is heuristic until lab adjudication.",
            "Historical workbook contains counts but not historical predicted IDs.",
            "Hierarchy sensitivity does not establish clinical equivalence.",
        ],
    }
    args.sanitized_output.parent.mkdir(parents=True, exist_ok=True)
    args.sanitized_output.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(sanitized, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
