from __future__ import annotations

from pathlib import Path

from rag_hpo.artifacts import ArtifactEntry
from rag_hpo.diagnostics import (
    distinct_candidates,
    hierarchy_match,
    holm_adjust,
    parse_obo,
)

OBO = """format-version: 1.2
data-version: hp/releases/2025-05-06

[Term]
id: HP:0000001
name: Root

[Term]
id: HP:0000002
name: Parent
is_a: HP:0000001 ! Root

[Term]
id: HP:0000003
name: Child A
alt_id: HP:9000003
synonym: "Third" EXACT []
is_a: HP:0000002 ! Parent

[Term]
id: HP:0000004
name: Child B
is_a: HP:0000002 ! Parent

[Term]
id: HP:0000005
name: obsolete Child
is_obsolete: true
replaced_by: HP:0000004
"""


def test_ontology_relations_and_hierarchy_matching(tmp_path: Path) -> None:
    path = tmp_path / "hp.obo"
    path.write_text(OBO, encoding="utf-8")
    ontology = parse_obo(path)
    assert ontology.data_version == "hp/releases/2025-05-06"
    assert ontology.normalize("HP:9000003") == "HP:0000003"
    assert ontology.normalize("HP:0000005") == "HP:0000004"
    assert ontology.relation("HP:0000003", "HP:0000002").relation == "predicted_descendant"
    assert ontology.relation("HP:0000002", "HP:0000003").relation == "predicted_ancestor"
    assert ontology.relation("HP:0000003", "HP:0000004").relation == "sibling"
    matches = hierarchy_match(
        {"HP:0000003", "HP:0000004"},
        {"HP:0000002", "HP:0000004"},
        ontology,
        max_distance=1,
    )
    assert len(matches) == 2
    assert sum(relation.relation == "exact" for _, _, relation in matches) == 1


def test_distinct_candidates_do_not_count_duplicate_hpo_ids() -> None:
    entries = [
        ArtifactEntry(hp_id="HP:1", phrase="one", term="One", source="label"),
        ArtifactEntry(hp_id="HP:1", phrase="first", term="One", source="synonym"),
        ArtifactEntry(hp_id="HP:2", phrase="two", term="Two", source="label"),
    ]
    candidates = distinct_candidates(entries, [0.9, 0.8, 0.7], [0, 1, 2], limit=8)
    assert [candidate.hpo_id for candidate in candidates] == ["HP:1", "HP:2"]
    assert [candidate.raw_rank for candidate in candidates] == [1, 3]
    raw_two = distinct_candidates(
        entries,
        [0.9, 0.8, 0.7],
        [0, 1, 2],
        limit=8,
        raw_limit=2,
    )
    assert [candidate.hpo_id for candidate in raw_two] == ["HP:1"]


def test_holm_adjustment_is_monotonic_in_sorted_p_values() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
