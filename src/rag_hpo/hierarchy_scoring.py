from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from rag_hpo.diagnostics import OntologyRelation, OntologySnapshot


@dataclass(frozen=True)
class LayeredCaseScore:
    patient_id: str
    max_distance: int
    exact_tp: int
    related_tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    exact_matches: list[str]
    related_matches: list[str]
    relation_counts: dict[str, int]


PredictionGroups = list[set[str]]
Match = tuple[int, str, str, OntologyRelation]


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _relation_for_group(
    prediction_group: set[str],
    group: set[str],
    ontology: OntologySnapshot,
    *,
    distance: int,
) -> tuple[str, str, OntologyRelation] | None:
    choices: list[tuple[str, str, OntologyRelation]] = []
    for prediction in sorted(prediction_group):
        for reference in sorted(group):
            relation = ontology.relation(prediction, reference)
            if distance == 0 and relation.relation == "exact":
                choices.append((prediction, reference, relation))
            elif relation.distance == distance and relation.relation in {
                "predicted_ancestor",
                "predicted_descendant",
                "sibling",
            }:
                choices.append((prediction, reference, relation))
    return choices[0] if choices else None


def _match_tier(
    predictions: PredictionGroups,
    available_predictions: set[int],
    groups: list[set[str]],
    available_groups: set[int],
    ontology: OntologySnapshot,
    *,
    distance: int,
) -> dict[int, Match]:
    edges: dict[int, list[tuple[int, str, str, OntologyRelation]]] = {}
    for prediction_index in sorted(available_predictions):
        values: list[tuple[int, str, str, OntologyRelation]] = []
        for group_index in sorted(available_groups):
            choice = _relation_for_group(
                predictions[prediction_index],
                groups[group_index],
                ontology,
                distance=distance,
            )
            if choice is not None:
                prediction, reference, relation = choice
                values.append((group_index, prediction, reference, relation))
        edges[prediction_index] = values

    matches: dict[int, Match] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for group_index, prediction, reference, relation in edges[prediction_index]:
            if group_index in visited:
                continue
            visited.add(group_index)
            existing = matches.get(group_index)
            if existing is None or augment(existing[0], visited):
                matches[group_index] = (
                    prediction_index,
                    prediction,
                    reference,
                    relation,
                )
                return True
        return False

    for prediction_index in sorted(available_predictions):
        augment(prediction_index, set())
    return matches


def score_layered_case(
    patient_id: str,
    predictions: set[str] | PredictionGroups,
    reference_groups: list[set[str]],
    ontology: OntologySnapshot,
    *,
    max_distance: int,
) -> LayeredCaseScore:
    """Score alternatives exactly first, then one-to-one ontology neighbors."""

    if max_distance not in {0, 1, 2}:
        raise ValueError("max_distance must be 0, 1, or 2")
    prediction_groups = (
        [{ontology.normalize(value)} for value in predictions]
        if isinstance(predictions, set)
        else [
            {ontology.normalize(value) for value in prediction_group}
            for prediction_group in predictions
        ]
    )
    groups = [{ontology.normalize(value) for value in group} for group in reference_groups]
    available_predictions = set(range(len(prediction_groups)))
    available_groups = set(range(len(groups)))
    all_matches: dict[int, Match] = {}
    for distance in range(max_distance + 1):
        tier = _match_tier(
            prediction_groups,
            available_predictions,
            groups,
            available_groups,
            ontology,
            distance=distance,
        )
        all_matches.update(tier)
        available_groups.difference_update(tier)
        available_predictions.difference_update(match[0] for match in tier.values())

    exact = [
        f"{prediction}>{reference}:exact:0"
        for _index, prediction, reference, relation in all_matches.values()
        if relation.relation == "exact"
    ]
    related = [
        f"{prediction}>{reference}:{relation.relation}:{relation.distance}"
        for _index, prediction, reference, relation in all_matches.values()
        if relation.relation != "exact"
    ]
    counts = Counter(match[3].relation for match in all_matches.values())
    tp = len(all_matches)
    fp = len(prediction_groups) - tp
    fn = len(groups) - tp
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return LayeredCaseScore(
        patient_id=patient_id,
        max_distance=max_distance,
        exact_tp=counts["exact"],
        related_tp=tp - counts["exact"],
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_matches=sorted(exact),
        related_matches=sorted(related),
        relation_counts=dict(sorted(counts.items())),
    )


def score_layered(
    predictions: dict[str, set[str]] | dict[str, PredictionGroups],
    references: dict[str, list[set[str]]],
    ontology: OntologySnapshot,
    *,
    max_distance: int,
    patient_ids: list[str] | None = None,
) -> list[LayeredCaseScore]:
    selected = patient_ids or sorted(
        references,
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    return [
        score_layered_case(
            patient_id,
            predictions.get(patient_id, set()),
            references.get(patient_id, []),
            ontology,
            max_distance=max_distance,
        )
        for patient_id in selected
    ]


def summarize_layered(scores: list[LayeredCaseScore]) -> dict[str, Any]:
    exact_tp = sum(value.exact_tp for value in scores)
    related_tp = sum(value.related_tp for value in scores)
    tp = exact_tp + related_tp
    fp = sum(value.fp for value in scores)
    fn = sum(value.fn for value in scores)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    relation_counts: Counter[str] = Counter()
    for value in scores:
        relation_counts.update(value.relation_counts)
    count = len(scores)
    return {
        "case_count": count,
        "max_distance": scores[0].max_distance if scores else 0,
        "micro": {
            "exact_tp": exact_tp,
            "related_tp": related_tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "macro": {
            "precision": sum(value.precision for value in scores) / count if count else 0.0,
            "recall": sum(value.recall for value in scores) / count if count else 0.0,
            "f1": sum(value.f1 for value in scores) / count if count else 0.0,
        },
        "relation_counts": dict(sorted(relation_counts.items())),
        "cases": [asdict(value) for value in scores],
    }
