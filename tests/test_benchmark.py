from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks/recompute_metrics.py"
    spec = importlib.util.spec_from_file_location("recompute_metrics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_metric_calculation() -> None:
    module = _module()
    precision, recall, f1 = module.compute(5, 2, 3)
    assert precision == pytest.approx(5 / 7)
    assert recall == pytest.approx(5 / 8)
    assert f1 == pytest.approx(2 * precision * recall / (precision + recall))
    assert module.compute(0, 0, 0) == (0.0, 0.0, 0.0)


def test_published_workbook_yields_tidy_rows() -> None:
    module = _module()
    workbook = Path(__file__).parents[1] / "RAG-HPO Tests and Data Analysis copy.xlsx"
    rows = module.extract(workbook)
    assert rows
    assert all(row.patient_id for row in rows)
    assert all(0 <= row.precision <= 1 for row in rows)
    assert all(0 <= row.recall <= 1 for row in rows)
    assert all(0 <= row.f1 <= 1 for row in rows)
