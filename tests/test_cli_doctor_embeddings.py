from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fastembed
import numpy as np
import pytest
import sentence_transformers

import rag_hpo.cli as cli
import rag_hpo.doctor as doctor_module
from rag_hpo.artifacts import ArtifactEntry, ArtifactManifest, write_artifacts
from rag_hpo.embeddings import (
    FastEmbedBackend,
    SapBERTBackend,
    create_backend,
    normalize_rows,
)
from rag_hpo.models import AnnotationResult, Category


def _manifest() -> ArtifactManifest:
    return ArtifactManifest(
        created_at="2026-01-01T00:00:00+00:00",
        hpo_source="test",
        hpo_sha256="a",
        addons_sha256=None,
        parser="test",
        parser_version="1",
        embedding_backend="fake",
        embedding_model="fake",
        embedding_revision="fake",
        dtype="float32",
        dimension=2,
        metadata_count=1,
        vector_count=1,
        metadata_sha256="b",
        vectors_sha256="c",
    )


def _write_vectors(path: Path) -> None:
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
        embedding_model="fake",
        embedding_revision="fake",
    )


def test_doctor_validates_vectors_and_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = tmp_path / "vectors"
    _write_vectors(vectors)
    monkeypatch.setenv("RAG_HPO_API_KEY", "test")
    called: list[bool] = []

    class Provider:
        def __init__(self, _: object) -> None:
            pass

        def __enter__(self) -> Provider:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def health_check(self) -> None:
            called.append(True)

    monkeypatch.setattr(doctor_module, "OpenAICompatibleProvider", Provider)
    report = doctor_module.run_doctor(
        vector_dir=vectors,
        output_dir=tmp_path / "output",
        check_provider=True,
    )
    assert report.ok
    assert called == [True]
    assert {check.status for check in report.checks} == {"pass"}


def test_doctor_reports_bad_vectors_and_missing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_HPO_API_KEY", raising=False)
    report = doctor_module.run_doctor(
        vector_dir=tmp_path / "missing",
        output_dir=tmp_path / "output",
        check_provider=True,
    )
    assert not report.ok
    assert sum(check.status == "fail" for check in report.checks) == 2


def test_cli_vectorize_reports_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setattr(cli, "vectorize", lambda **_: _manifest())
    code = cli.main(
        [
            "vectorize",
            "--output-dir",
            str(tmp_path),
            "--limit",
            "1",
            "--json",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["dimension"] == 2


def test_cli_annotate_uses_manual_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setenv("RAG_HPO_API_KEY", "test")

    class Provider:
        def __init__(self, _: object) -> None:
            pass

        def __enter__(self) -> Provider:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    class Pipeline:
        def __init__(self, **_: object) -> None:
            pass

        def run(
            self,
            rows: list[object],
            *,
            initial_errors: list[object],
        ) -> list[AnnotationResult]:
            assert len(rows) == 1
            assert initial_errors == []
            return [
                AnnotationResult(
                    patient_id="abc",
                    phrase="fever",
                    category=Category.ABNORMAL,
                    mapping_status="no_candidate_fit",
                )
            ]

    monkeypatch.setattr(cli, "OpenAICompatibleProvider", Provider)
    monkeypatch.setattr(cli, "AnnotationPipeline", Pipeline)
    code = cli.main(
        [
            "annotate",
            "--text",
            "Unicode café",
            "--patient-id",
            "abc",
            "--vector-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--json",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["error_rows"] == 0


def test_cli_stdin_and_invalid_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("stdin note"))
    args = cli.build_parser().parse_args(
        [
            "annotate",
            "--input",
            "-",
            "--vector-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
        ]
    )
    rows, errors = cli._load_annotation_source(args)
    assert rows[0].clinical_note == "stdin note"
    assert errors == []
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["vectorize", "--output-dir", str(tmp_path), "--limit", "0"])


def test_embedding_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix = normalize_rows(np.asarray([[3.0, 4.0]], dtype=np.float32))
    assert np.allclose(matrix, [[0.6, 0.8]])
    with pytest.raises(ValueError, match="zero"):
        normalize_rows(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="unknown"):
        create_backend("missing")

    class SentenceModel:
        def __init__(self, model_id: str, revision: str) -> None:
            assert model_id and revision

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            return np.asarray([[2.0, 0.0] for _ in texts])

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", SentenceModel)
    sapbert = SapBERTBackend()
    assert np.allclose(sapbert.encode(["x"]), [[1.0, 0.0]])

    class FastModel:
        def __init__(self, model_name: str) -> None:
            assert model_name

        def embed(self, texts: list[str]) -> list[np.ndarray]:
            return [np.asarray([0.0, 2.0]) for _ in texts]

    monkeypatch.setattr(fastembed, "TextEmbedding", FastModel)
    fast = FastEmbedBackend()
    assert np.allclose(fast.encode(["x"]), [[0.0, 1.0]])
