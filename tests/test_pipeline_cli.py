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


def test_load_inputs_txt_file_and_directory(tmp_path: Path) -> None:
    from rag_hpo.pipeline import load_inputs

    # Single .txt file
    txt_path = tmp_path / "patient_101.txt"
    txt_path.write_text("Patient presents with fever and cough.", encoding="utf-8")
    rows, _errors = load_inputs(txt_path)
    assert len(rows) == 1
    assert rows[0].patient_id == "patient_101"
    assert rows[0].clinical_note == "Patient presents with fever and cough."

    # Directory of .txt files
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "note1.txt").write_text("Note 1 content", encoding="utf-8")
    (notes_dir / "note2.txt").write_text("Note 2 content", encoding="utf-8")

    rows_dir, _ = load_inputs(notes_dir)
    assert len(rows_dir) == 2
    assert [r.patient_id for r in rows_dir] == ["note1", "note2"]


def test_load_inputs_flexible_csv_columns(tmp_path: Path) -> None:
    from rag_hpo.pipeline import load_inputs

    # CSV with 'text' column alias instead of 'clinical_note'
    csv_path = tmp_path / "input_alias.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "text"])
        writer.writeheader()
        writer.writerow({"patient_id": "p1", "text": "Patient has short stature."})

    rows, _ = load_inputs(csv_path)
    assert len(rows) == 1
    assert rows[0].patient_id == "p1"
    assert rows[0].clinical_note == "Patient has short stature."


def test_load_inputs_docx(tmp_path: Path) -> None:
    import zipfile

    from rag_hpo.pipeline import load_inputs

    docx_path = tmp_path / "patient_202.docx"
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        "  <w:body>\n"
        "    <w:p><w:r><w:t>Patient presents with severe microcephaly.</w:t></w:r></w:p>\n"
        "  </w:body>\n"
        "</w:document>"
    )
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)

    rows, _errors = load_inputs(docx_path)
    assert len(rows) == 1
    assert rows[0].patient_id == "patient_202"
    assert rows[0].clinical_note == "Patient presents with severe microcephaly."


def test_load_inputs_excel(tmp_path: Path) -> None:
    import openpyxl

    from rag_hpo.pipeline import load_inputs

    excel_path = tmp_path / "clinical_records.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["patient_id", "note"])
    ws.append(["P-100", "Patient has short stature and microcephaly."])
    wb.save(excel_path)

    rows, _errors = load_inputs(excel_path)
    assert len(rows) == 1
    assert rows[0].patient_id == "P-100"
    assert rows[0].clinical_note == "Patient has short stature and microcephaly."
