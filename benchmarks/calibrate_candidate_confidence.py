#!/usr/bin/env python3
"""Calibrate bounded candidate-set confidence and freeze a precision-constrained policy."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.benchmark import (
    _maximum_group_matching,
    load_hpo_aliases,
    load_reference_groups,
)
from rag_hpo.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    STAGED_PIPELINE_SCHEMA_VERSION,
    CalibrationStratum,
    ConfidenceCalibration,
    evidence_stratum,
)
from rag_hpo.models import AnnotationResult, Category


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total == 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - margin) / denominator)


def _load_predictions(
    path: Path,
    aliases: dict[str, str],
) -> dict[str, list[AnnotationResult]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("predictions must be a JSON array")
    output: dict[str, list[AnnotationResult]] = defaultdict(list)
    seen: dict[str, set[frozenset[str]]] = defaultdict(set)
    for value in raw:
        result = AnnotationResult.model_validate(value)
        if (
            result.category is not Category.ABNORMAL
            or result.mapping_status != "mapped"
            or not result.candidate_hpo_ids
        ):
            continue
        candidates = {aliases.get(candidate, candidate) for candidate in result.candidate_hpo_ids}
        if len(candidates) > 3:
            raise ValueError("calibration requires candidate sets of no more than three IDs")
        identity = frozenset(candidates)
        if identity in seen[result.patient_id]:
            continue
        seen[result.patient_id].add(identity)
        output[result.patient_id].append(
            result.model_copy(update={"candidate_hpo_ids": sorted(candidates)})
        )
    return dict(output)


def _selected_case_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload.get("selection", payload)
    for key in ("selected_case_ids", "confirmation_case_ids", "case_ids"):
        values = selection.get(key)
        if isinstance(values, list) and values:
            case_ids = [str(value) for value in values]
            if len(case_ids) != len(set(case_ids)):
                raise ValueError("selection manifest contains duplicate case IDs")
            return case_ids
    raise ValueError("selection manifest does not contain a supported case-ID list")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.70)
    parser.add_argument(
        "--precision-criterion",
        choices=("point", "lower95"),
        default="point",
        help=(
            "Apply the precision floor to the point estimate, or to the more "
            "conservative Wilson 95%% lower bound."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.minimum_precision <= 1:
        parser.error("--minimum-precision must be greater than zero and at most one")

    aliases = load_hpo_aliases(args.ontology)
    references = load_reference_groups(args.references, aliases)
    predictions = _load_predictions(args.predictions, aliases)
    selected_case_ids = _selected_case_ids(args.selection_manifest)
    missing_references = [value for value in selected_case_ids if value not in references]
    if missing_references:
        raise ValueError(
            "selected cases are absent from the reference table: " + ", ".join(missing_references)
        )
    unexpected_predictions = sorted(set(predictions) - set(selected_case_ids))
    if unexpected_predictions:
        raise ValueError(
            "predictions contain cases outside the locked selection: "
            + ", ".join(unexpected_predictions)
        )
    observations: dict[str, list[int]] = defaultdict(list)
    total_references = sum(len(references[value]) for value in selected_case_ids)
    for patient_id in selected_case_ids:
        rows = predictions.get(patient_id, [])
        candidate_sets = [set(row.candidate_hpo_ids or []) for row in rows]
        matched = _maximum_group_matching(candidate_sets, references[patient_id])
        matched_predictions = set(matched.values())
        for index, row in enumerate(rows):
            # This is the requested benchmark truth: exact candidate-set overlap is 1.
            observations[evidence_stratum(row)].append(int(index in matched_predictions))

    strata: dict[str, CalibrationStratum] = {}
    for key, labels in sorted(observations.items()):
        count = len(labels)
        matched_count = sum(labels)
        strata[key] = CalibrationStratum(
            observations=count,
            gold_matches=matched_count,
            estimated_probability=(matched_count + 1) / (count + 2),
            lower_95=_wilson_lower(matched_count, count),
        )
    total_predictions = sum(value.observations for value in strata.values())
    total_matches = sum(value.gold_matches for value in strata.values())
    global_probability = (total_matches + 1) / (total_predictions + 2)

    ordered = sorted(
        strata,
        key=lambda key: (
            -strata[key].estimated_probability,
            -strata[key].lower_95,
            key,
        ),
    )
    best: tuple[float, float, int, list[str]] | None = None
    accepted: list[str] = []
    for index in range(1, len(ordered) + 1):
        included = ordered[:index]
        tp = sum(strata[key].gold_matches for key in included)
        predictions_count = sum(strata[key].observations for key in included)
        precision = tp / predictions_count if predictions_count else 0.0
        recall = tp / total_references if total_references else 0.0
        lower = _wilson_lower(tp, predictions_count)
        qualifying_precision = precision if args.precision_criterion == "point" else lower
        if qualifying_precision < args.minimum_precision:
            continue
        candidate = (recall, precision, -predictions_count, included)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
            accepted = included

    accepted_tp = sum(strata[key].gold_matches for key in accepted)
    accepted_predictions = sum(strata[key].observations for key in accepted)
    accepted_precision = accepted_tp / accepted_predictions if accepted_predictions else 0.0
    accepted_recall = accepted_tp / total_references if total_references else 0.0
    calibration = ConfidenceCalibration(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        pipeline_schema_version=STAGED_PIPELINE_SCHEMA_VERSION,
        prompt_bundle_sha256=sha256_file(args.prompt_file),
        artifact_manifest_sha256=sha256_file(args.vector_manifest),
        minimum_precision=args.minimum_precision,
        accepted_strata=accepted,
        strata=strata,
        global_probability=global_probability,
        selection={
            "objective": (
                "maximize exact candidate-set recall while the selected precision "
                "criterion meets the frozen floor"
            ),
            "precision_criterion": args.precision_criterion,
            "minimum_precision": args.minimum_precision,
            "accepted_stratum_count": len(accepted),
            "accepted_prediction_count": accepted_predictions,
            "accepted_gold_matches": accepted_tp,
            "accepted_precision": accepted_precision,
            "accepted_recall": accepted_recall,
            "accepted_precision_lower_95": _wilson_lower(accepted_tp, accepted_predictions),
            "total_reference_findings": total_references,
            "calibration_prediction_count": total_predictions,
            "calibration_gold_matches": total_matches,
            "selected_case_count": len(selected_case_ids),
        },
        provenance={
            "predictions_sha256": sha256_file(args.predictions),
            "references_sha256": sha256_file(args.references),
            "ontology_sha256": sha256_file(args.ontology),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
        },
        limits={
            "phenotype_confidence": (
                "not separately estimable without span-level human phenotype labels"
            ),
            "mapping_set_confidence": (
                "not separately estimable when extraction and mapping share one gold outcome"
            ),
        },
    )
    output = calibration.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["selection"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
