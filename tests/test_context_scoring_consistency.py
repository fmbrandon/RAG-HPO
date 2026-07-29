from __future__ import annotations

from pathlib import Path

import pytest

from rag_hpo.consistency import evaluate_consistency, load_dispositions
from rag_hpo.diagnostics import parse_obo
from rag_hpo.hierarchy_scoring import (
    score_layered,
    score_layered_case,
    summarize_layered,
)

OBO = """format-version: 1.2
data-version: test

[Term]
id: HP:0000001
name: Root

[Term]
id: HP:0000002
name: Parent
is_a: HP:0000001 ! Root

[Term]
id: HP:0000003
name: Child
is_a: HP:0000002 ! Parent

[Term]
id: HP:0000004
name: Sibling
is_a: HP:0000002 ! Parent
"""


def test_layered_scoring_preserves_exact_and_catalogs_neighbors(tmp_path: Path) -> None:
    path = tmp_path / "hp.obo"
    path.write_text(OBO, encoding="utf-8")
    ontology = parse_obo(path)

    exact = score_layered_case(
        "1",
        {"HP:0000003", "HP:0000004"},
        [{"HP:0000002"}, {"HP:0000004"}],
        ontology,
        max_distance=0,
    )
    assert (exact.exact_tp, exact.related_tp, exact.fp, exact.fn) == (1, 0, 1, 1)

    one_edge = score_layered_case(
        "1",
        {"HP:0000003", "HP:0000004"},
        [{"HP:0000002"}, {"HP:0000004"}],
        ontology,
        max_distance=1,
    )
    assert (one_edge.exact_tp, one_edge.related_tp, one_edge.fp, one_edge.fn) == (
        1,
        1,
        0,
        0,
    )
    assert one_edge.relation_counts["predicted_descendant"] == 1

    sibling = score_layered_case(
        "1",
        {"HP:0000004"},
        [{"HP:0000003"}],
        ontology,
        max_distance=2,
    )
    assert sibling.relation_counts == {"sibling": 1}

    summary = summarize_layered(
        score_layered(
            {"1": {"HP:0000004"}},
            {"1": [{"HP:0000003"}]},
            ontology,
            max_distance=2,
            patient_ids=["1"],
        )
    )
    assert summary["micro"]["related_tp"] == 1
    assert summary["macro"]["f1"] == 1.0

    with pytest.raises(ValueError, match="max_distance"):
        score_layered_case("1", set(), [], ontology, max_distance=3)


def test_consistency_harness_reports_stable_repeated_runs() -> None:
    predictions = {
        "1": {"HP:0000001", "HP:0000002"},
        "2": {"HP:0000003"},
    }
    references = {
        "1": [{"HP:0000001"}, {"HP:0000002"}],
        "2": [{"HP:0000003"}],
    }
    report = evaluate_consistency(
        [predictions, predictions, predictions],
        references,
        patient_ids=["1", "2"],
        dispositions=[
            {
                ("1", "HP:0000001"): "accepted",
                ("1", "HP:0000002"): "accepted",
                ("2", "HP:0000003"): "accepted",
                ("outside-selection", "HP:9999999"): "review",
            }
        ]
        * 3,
    )
    assert report["all_gates_pass"] is True
    assert report["mean_pairwise_case_jaccard"] == 1.0
    assert report["all_run_id_recurrence"] == 1.0
    assert report["disposition_agreement"] == 1.0


def test_consistency_loads_dispositions_and_reports_instability(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        """
[
  {"patient_id":"1","hpo_id":"HP:9000001","review_status":"review"},
  {"patient_id":"1","hpo_id":"HP:9000001","review_status":"accepted"},
  {"patient_id":"2","hpo_id":"HP:0000002","review_status":"unexpected"},
  {"patient_id":"","hpo_id":"HP:0000003","review_status":"accepted"}
]
""".strip(),
        encoding="utf-8",
    )
    dispositions = load_dispositions(path, {"HP:9000001": "HP:0000001"})
    assert dispositions == {
        ("1", "HP:0000001"): "accepted",
        ("2", "HP:0000002"): "review",
    }

    report = evaluate_consistency(
        [{"1": {"HP:0000001"}}, {"1": {"HP:0000002"}}],
        {"1": [{"HP:0000001"}]},
        patient_ids=["1"],
    )
    assert report["all_gates_pass"] is False
    assert report["disposition_agreement"] is None
    assert report["all_run_id_recurrence"] == 0.0

    with pytest.raises(ValueError, match="at least two"):
        evaluate_consistency([{}], {}, patient_ids=[])
