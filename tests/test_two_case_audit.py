from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "audit_two_case_pipeline.py"
SPEC = importlib.util.spec_from_file_location("audit_two_case_pipeline", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("alpha\r\nbeta\rcharlie", "alpha\nbeta\ncharlie"),
        ("  smart “quote”  ", "smart “quote”"),
        ("Cafe\u0301", "Café"),
        ("non-ASCII µm", "non-ASCII µm"),
    ],
)
def test_canonicalize_note(raw: str, expected: str) -> None:
    assert AUDIT.canonicalize_note(raw) == expected


def test_config_diff_is_path_specific_and_deterministic() -> None:
    assert AUDIT.config_diff(
        {"temperature": 0.2, "nested": {"mode": "old"}},
        {"temperature": 0.0, "nested": {"mode": "new"}},
    ) == [
        {"path": "nested.mode", "old": "old", "new": "new"},
        {"path": "temperature", "old": 0.2, "new": 0.0},
    ]


def test_ids_at_obeys_checkpoint_limit() -> None:
    record = {
        "dense_top64": [
            {"hpo_id": "HP:1"},
            {"hpo_id": "HP:2"},
            {"hpo_id": "HP:3"},
        ]
    }
    assert AUDIT.ids_at(record, "dense_top64", 2) == {"HP:1", "HP:2"}
