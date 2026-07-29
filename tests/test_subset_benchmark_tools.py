from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = _load_script("bootstrap_subset_metrics")
selector = _load_script("prepare_stratified_subset")
runner = _load_script("run_benchmark")
corpus_runner = _load_script("run_corpus_evaluation")


def test_rank_bins_cover_requested_range_deterministically() -> None:
    case_ids = ["3", "1", "4", "2", "5", "6"]
    bins = selector._rank_bins(
        case_ids,
        {"1": 10, "2": 10, "3": 20, "4": 30, "5": 40, "6": 50},
        3,
    )
    assert bins == {"1": 0, "2": 0, "3": 1, "4": 1, "5": 2, "6": 2}


def test_stratum_allocation_is_complete_and_within_capacity() -> None:
    strata = {(0, 0): ["1", "2", "3"], (0, 1): ["4"], (1, 1): ["5", "6"]}
    allocation = selector._allocate(strata, 4)
    assert sum(allocation.values()) == 4
    assert all(0 <= allocation[key] <= len(strata[key]) for key in strata)
    assert all(allocation[key] >= 1 for key in strata)


def test_bootstrap_aggregate_recomputes_micro_and_macro_metrics() -> None:
    aggregate = bootstrap._aggregate(
        [
            {"tp": 2, "fp": 1, "fn": 0, "precision": 2 / 3, "recall": 1, "f1": 0.8},
            {"tp": 1, "fp": 0, "fn": 2, "precision": 1, "recall": 1 / 3, "f1": 0.5},
        ]
    )
    assert aggregate["tp"] == 3
    assert aggregate["fp"] == 1
    assert aggregate["fn"] == 2
    assert aggregate["precision"] == 0.75
    assert aggregate["recall"] == 0.6
    assert aggregate["f1"] == pytest.approx(2 / 3)


def test_runner_reads_locked_subset_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        '{"selection": {"selected_case_ids": [3, "7", 11]}}',
        encoding="utf-8",
    )
    assert runner._selection_ids(manifest) == ["3", "7", "11"]


def test_corpus_runner_writes_only_selected_rows_with_private_permissions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.csv"
    source.write_text(
        "patient_id,clinical_note\n1,first note\n2,second note\n",
        encoding="utf-8",
    )
    output = tmp_path / "selected.csv"
    selected = corpus_runner._write_selected_input(source, output, ["2"])
    assert selected == ["2"]
    assert output.read_text(encoding="utf-8").splitlines() == [
        "patient_id,clinical_note",
        "2,second note",
    ]
    assert output.stat().st_mode & 0o077 == 0
