from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_hpo.benchmark import (
    load_hpo_aliases,
    load_prediction_sets,
    load_reference_sets,
    score_sets,
    summarize,
    write_benchmark_report,
)
from rag_hpo.statistics import paired_inference

OBO = """format-version: 1.2

[Term]
id: HP:0000001
name: First
alt_id: HP:9000001

[Term]
id: HP:0000002
name: Second
"""


def test_exact_set_scoring_normalizes_alt_ids_and_deduplicates(tmp_path: Path) -> None:
    ontology = tmp_path / "hp.obo"
    ontology.write_text(OBO, encoding="utf-8")
    references = tmp_path / "references.csv"
    references.write_text(
        "Patient ID,hpo_term\n1,HP:0000001\n1,HP:0000002\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "patient_id,hpo_id,mapping_status\n"
        "1,HP:9000001,mapped\n"
        "1,HP:9000001,mapped\n"
        "1,HP:9999999,mapped\n",
        encoding="utf-8",
    )
    aliases = load_hpo_aliases(ontology)
    scores = score_sets(
        load_prediction_sets(predictions, aliases),
        load_reference_sets(references, aliases),
        patient_ids=["1"],
    )
    score = scores[0]
    assert score.true_positive_ids == ["HP:0000001"]
    assert score.false_positive_ids == ["HP:9999999"]
    assert score.false_negative_ids == ["HP:0000002"]
    assert (score.tp, score.fp, score.fn) == (1, 1, 1)
    assert summarize(scores)["micro"]["f1"] == 0.5


def test_benchmark_reports_are_byte_reproducible(tmp_path: Path) -> None:
    ontology = tmp_path / "hp.obo"
    ontology.write_text(OBO, encoding="utf-8")
    references = tmp_path / "references.csv"
    references.write_text("Patient ID,hpo_term\n1,HP:0000001\n", encoding="utf-8")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps([{"patient_id": "1", "hpo_id": "HP:0000001", "mapping_status": "mapped"}]),
        encoding="utf-8",
    )
    aliases = load_hpo_aliases(ontology)
    scores = score_sets(
        load_prediction_sets(predictions, aliases),
        load_reference_sets(references, aliases),
    )
    metadata = {
        "model": "mock-model",
        "prompt_version": "1.0",
        "vector_manifest_sha256": "abc",
    }
    first = write_benchmark_report(
        scores,
        tmp_path / "first",
        predictions_path=predictions,
        references_path=references,
        ontology_path=ontology,
        metadata=metadata,
    )
    second = write_benchmark_report(
        scores,
        tmp_path / "second",
        predictions_path=predictions,
        references_path=references,
        ontology_path=ontology,
        metadata=metadata,
    )
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()


def test_exported_reference_corpora_are_separate_and_case_68_is_repaired() -> None:
    root = Path(__file__).parents[1] / "benchmarks" / "references"
    aliases = {"HP:0001513": "HP:0001513"}
    csc = load_reference_sets(root / "csc_manual_annotations.csv", aliases)
    assert len(csc["1"]) == 8

    with (Path(__file__).parents[1] / "Test_Cases.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        csc_rows = {row["Case"]: row["clinical_note"] for row in csv.DictReader(handle)}
    with (root / "gsc_input.csv").open(encoding="utf-8-sig", newline="") as handle:
        gsc_rows = {row["patient_id"]: row["clinical_note"] for row in csv.DictReader(handle)}
    with (root / "csc_input.csv").open(encoding="utf-8-sig", newline="") as handle:
        authoritative_csc = {row["Case"]: row["clinical_note"] for row in csv.DictReader(handle)}
    assert csc_rows["1"] != gsc_rows["1"]
    assert csc_rows["67"] != csc_rows["68"]
    assert csc_rows["68"] == authoritative_csc["68"]
    assert len(csc["68"]) == 10


def test_paired_inference_detects_consistent_improvement() -> None:
    result = paired_inference([0.1] * 30, seed=7, iterations=1_000)
    assert result["mean_difference_ci95_lower"] == 0.1
    assert result["one_sided_sign_flip_p"] < 0.05


def test_stored_benchmark_reports_contain_no_credentials() -> None:
    result_dir = Path(__file__).parents[1] / "benchmarks" / "results"
    for path in result_dir.glob("*.json"):
        content = path.read_text(encoding="utf-8").lower()
        assert "gsk_" not in content
        assert '"api_key"' not in content
