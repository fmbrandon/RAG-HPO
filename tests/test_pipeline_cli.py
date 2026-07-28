from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rag_hpo.artifacts import ArtifactEntry, write_artifacts
from rag_hpo.cli import main
from rag_hpo.models import (
    AnnotationInput,
    HPOMapping,
    PhenotypeExtraction,
)
from rag_hpo.pipeline import AnnotationPipeline, load_csv_inputs


class FakeBackend:
    name = "fake"
    model_id = "fake-model"
    revision = "fake-revision"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def request(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> tuple[Any, str]:
        self.calls += 1
        if response_model is PhenotypeExtraction:
            value = PhenotypeExtraction.model_validate(
                {
                    "phenotypes": [
                        {"phrase": "café fever", "category": "Abnormal"},
                        {"phrase": "normal hearing", "category": "Normal"},
                    ]
                }
            )
            return value, value.model_dump_json()
        assert response_model is HPOMapping
        value = HPOMapping(hpo_id="HP:1")
        return value, value.model_dump_json()


def _vectors(path: Path) -> None:
    write_artifacts(
        path,
        entries=[
            ArtifactEntry(
                hp_id="HP:1",
                phrase="Fever",
                term="Fever",
                source="test",
            )
        ],
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        hpo_source="test",
        hpo_sha256="a",
        addons_sha256=None,
        parser="test",
        parser_version="1",
        embedding_backend="fake",
        embedding_model="fake-model",
        embedding_revision="fake-revision",
    )


def test_pipeline_maps_abnormal_and_preserves_unicode(tmp_path: Path) -> None:
    vector_dir = tmp_path / "vectors"
    output_dir = tmp_path / "output"
    _vectors(vector_dir)
    provider = FakeProvider()
    pipeline = AnnotationPipeline(
        provider=provider,  # type: ignore[arg-type]
        vector_dir=vector_dir,
        output_dir=output_dir,
        resume=False,
        keep_state=False,
        keep_raw_responses=False,
        backend=FakeBackend(),
    )
    results = pipeline.run([AnnotationInput(patient_id="001-A", clinical_note="café fever")])
    assert [value.mapping_status for value in results] == [
        "mapped",
        "not_mapped_category",
    ]
    assert results[0].phrase == "café fever"
    assert results[0].hpo_id == "HP:1"
    assert not (output_dir / ".rag-hpo-state.sqlite3").exists()
    exported = json.loads((output_dir / "rag_hpo_results.json").read_text())
    assert exported[0]["patient_id"] == "001-A"


def test_pipeline_retains_state_on_row_failure(tmp_path: Path) -> None:
    vector_dir = tmp_path / "vectors"
    output_dir = tmp_path / "output"
    _vectors(vector_dir)

    class BrokenProvider(FakeProvider):
        def request(self, **_: object) -> tuple[Any, str]:
            raise ValueError("bad row")

    pipeline = AnnotationPipeline(
        provider=BrokenProvider(),  # type: ignore[arg-type]
        vector_dir=vector_dir,
        output_dir=output_dir,
        resume=False,
        keep_state=False,
        keep_raw_responses=False,
        backend=FakeBackend(),
    )
    results = pipeline.run([AnnotationInput(patient_id="1", clinical_note="note")])
    assert results[0].mapping_status == "error"
    assert (output_dir / ".rag-hpo-state.sqlite3").exists()


def test_csv_loader_returns_row_level_errors(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Case", "clinical_note"])
        writer.writeheader()
        writer.writerow({"Case": "001", "clinical_note": "valid"})
        writer.writerow({"Case": "002", "clinical_note": "   "})
    rows, errors = load_csv_inputs(path)
    assert rows[0].patient_id == "001"
    assert errors[0].patient_id == "002"
    assert errors[0].error_code == "invalid_input"


def test_csv_loader_requires_schema(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_text("wrong\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clinical_note"):
        load_csv_inputs(path)


def test_doctor_json_without_key(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.delenv("RAG_HPO_API_KEY", raising=False)
    code = main(["doctor", "--skip-provider", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["ok"] is True
    assert any(check["name"] == "provider-configuration" for check in report["checks"])


def test_cli_rejects_blank_manual_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: Any,
) -> None:
    monkeypatch.setenv("RAG_HPO_API_KEY", "test")
    code = main(
        [
            "annotate",
            "--text",
            " ",
            "--vector-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert code == 2
    assert "ERROR" in capsys.readouterr().err
