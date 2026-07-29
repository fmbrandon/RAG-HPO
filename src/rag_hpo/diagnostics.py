from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import ArtifactEntry

_SYNONYM = re.compile(r'^synonym:\s+"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class OntologyTerm:
    hpo_id: str
    label: str
    definition: str
    synonyms: tuple[str, ...]
    parents: tuple[str, ...]
    alternate_ids: tuple[str, ...]
    obsolete: bool
    replaced_by: tuple[str, ...]


@dataclass(frozen=True)
class OntologyRelation:
    relation: str
    distance: int | None


@dataclass(frozen=True)
class RankedCandidate:
    hpo_id: str
    term: str
    score: float
    raw_rank: int
    distinct_rank: int
    source: str


class OntologySnapshot:
    def __init__(self, terms: dict[str, OntologyTerm], data_version: str) -> None:
        self.terms = terms
        self.data_version = data_version
        self.aliases: dict[str, str] = {}
        for hpo_id, term in terms.items():
            target = term.replaced_by[0] if term.obsolete and len(term.replaced_by) == 1 else hpo_id
            self.aliases[hpo_id] = target
            for alternate in term.alternate_ids:
                self.aliases[alternate] = target
        self._ancestor_cache: dict[str, dict[str, int]] = {}

    def normalize(self, hpo_id: str) -> str:
        return self.aliases.get(hpo_id, hpo_id)

    def ancestors(self, hpo_id: str) -> dict[str, int]:
        normalized = self.normalize(hpo_id)
        if normalized in self._ancestor_cache:
            return self._ancestor_cache[normalized]
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(normalized, 0)])
        while queue:
            current, distance = queue.popleft()
            term = self.terms.get(current)
            if term is None:
                continue
            for parent in term.parents:
                parent = self.normalize(parent)
                next_distance = distance + 1
                previous = distances.get(parent)
                if previous is None or next_distance < previous:
                    distances[parent] = next_distance
                    queue.append((parent, next_distance))
        self._ancestor_cache[normalized] = distances
        return distances

    def relation(self, predicted_id: str, reference_id: str) -> OntologyRelation:
        predicted = self.normalize(predicted_id)
        reference = self.normalize(reference_id)
        if predicted == reference:
            return OntologyRelation("exact", 0)
        predicted_ancestors = self.ancestors(predicted)
        reference_ancestors = self.ancestors(reference)
        if reference in predicted_ancestors:
            return OntologyRelation("predicted_descendant", predicted_ancestors[reference])
        if predicted in reference_ancestors:
            return OntologyRelation("predicted_ancestor", reference_ancestors[predicted])
        predicted_term = self.terms.get(predicted)
        reference_term = self.terms.get(reference)
        if predicted_term and reference_term:
            shared = set(predicted_term.parents) & set(reference_term.parents)
            if shared:
                return OntologyRelation("sibling", 2)
        return OntologyRelation("unrelated", None)


def parse_obo(path: Path) -> OntologySnapshot:
    terms: dict[str, OntologyTerm] = {}
    data_version = "not-recorded"
    current: dict[str, Any] | None = None

    def finish() -> None:
        nonlocal current
        if current is None or not current.get("hpo_id"):
            current = None
            return
        hpo_id = str(current["hpo_id"])
        terms[hpo_id] = OntologyTerm(
            hpo_id=hpo_id,
            label=str(current.get("label") or ""),
            definition=str(current.get("definition") or ""),
            synonyms=tuple(str(value) for value in current.get("synonyms", [])),
            parents=tuple(str(value) for value in current.get("parents", [])),
            alternate_ids=tuple(str(value) for value in current.get("alternate_ids", [])),
            obsolete=bool(current.get("obsolete", False)),
            replaced_by=tuple(str(value) for value in current.get("replaced_by", [])),
        )
        current = None

    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("data-version:"):
                data_version = line.partition(":")[2].strip()
            if line == "[Term]":
                finish()
                current = {
                    "synonyms": [],
                    "parents": [],
                    "alternate_ids": [],
                    "replaced_by": [],
                }
                continue
            if line.startswith("[") and line.endswith("]"):
                finish()
                continue
            if current is None:
                continue
            if line.startswith("id: HP:"):
                current["hpo_id"] = line.removeprefix("id: ").strip()
            elif line.startswith("name:"):
                current["label"] = line.removeprefix("name:").strip()
            elif line.startswith("def:"):
                match = re.match(r'^def:\s+"((?:[^"\\]|\\.)*)"', line)
                current["definition"] = match.group(1) if match else ""
            elif match := _SYNONYM.match(line):
                current["synonyms"].append(match.group(1).replace(r"\"", '"'))
            elif line.startswith("is_a: HP:"):
                current["parents"].append(line.removeprefix("is_a: ").partition(" !")[0].strip())
            elif line.startswith("alt_id: HP:"):
                current["alternate_ids"].append(line.removeprefix("alt_id: ").strip())
            elif line == "is_obsolete: true":
                current["obsolete"] = True
            elif line.startswith("replaced_by: HP:"):
                current["replaced_by"].append(line.removeprefix("replaced_by: ").strip())
    finish()
    if not terms:
        raise ValueError(f"no HPO terms were parsed from {path}")
    return OntologySnapshot(terms, data_version)


def distinct_candidates(
    entries: Sequence[ArtifactEntry],
    scores: Sequence[float],
    indices: Sequence[int],
    *,
    limit: int,
    raw_limit: int | None = None,
) -> list[RankedCandidate]:
    output: list[RankedCandidate] = []
    seen: set[str] = set()
    for raw_rank, (score, index) in enumerate(zip(scores, indices, strict=True), start=1):
        if raw_limit is not None and raw_rank > raw_limit:
            break
        if index < 0:
            continue
        entry = entries[index]
        if entry.hp_id in seen:
            continue
        seen.add(entry.hp_id)
        output.append(
            RankedCandidate(
                hpo_id=entry.hp_id,
                term=entry.term,
                score=round(float(score), 6),
                raw_rank=raw_rank,
                distinct_rank=len(output) + 1,
                source=entry.source,
            )
        )
        if len(output) == limit:
            break
    return output


def hierarchy_match(
    predicted_ids: Iterable[str],
    reference_ids: Iterable[str],
    ontology: OntologySnapshot,
    *,
    max_distance: int,
) -> list[tuple[str, str, OntologyRelation]]:
    predicted_set = {ontology.normalize(value) for value in predicted_ids}
    reference_set = {ontology.normalize(value) for value in reference_ids}
    exact_ids = predicted_set & reference_set
    exact_matches = [(hpo_id, hpo_id, OntologyRelation("exact", 0)) for hpo_id in sorted(exact_ids)]
    predicted = sorted(predicted_set - exact_ids)
    references = sorted(reference_set - exact_ids)
    candidates: dict[str, list[tuple[int, str, OntologyRelation]]] = {}
    for predicted_id in predicted:
        edges = []
        for reference_id in references:
            relation = ontology.relation(predicted_id, reference_id)
            if (
                relation.relation in {"predicted_ancestor", "predicted_descendant"}
                and relation.distance is not None
                and relation.distance <= max_distance
            ):
                edges.append((relation.distance or 0, reference_id, relation))
        candidates[predicted_id] = sorted(edges)

    matched_reference: dict[str, tuple[str, OntologyRelation]] = {}

    def augment(predicted_id: str, visited: set[str]) -> bool:
        for _, reference_id, relation in candidates[predicted_id]:
            if reference_id in visited:
                continue
            visited.add(reference_id)
            existing = matched_reference.get(reference_id)
            if existing is None or augment(existing[0], visited):
                matched_reference[reference_id] = (predicted_id, relation)
                return True
        return False

    for predicted_id in predicted:
        augment(predicted_id, set())
    hierarchy_matches = sorted(
        (
            predicted_id,
            reference_id,
            relation,
        )
        for reference_id, (predicted_id, relation) in matched_reference.items()
    )
    return exact_matches + hierarchy_matches


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(indexed)
    running = 0.0
    total = len(indexed)
    for rank, (original_index, value) in enumerate(indexed):
        candidate = min(1.0, value * (total - rank))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted
