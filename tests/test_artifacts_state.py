from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rag_hpo.artifacts import (
    ArtifactEntry,
    load_artifacts,
    sha256_file,
    write_artifacts,
)
from rag_hpo.models import AnnotationResult, Category
from rag_hpo.state import PipelineState, StateMismatchError


def _write(tmp_path: Path) -> None:
    write_artifacts(
        tmp_path,
        entries=[
            ArtifactEntry(
                hp_id="HP:1",
                phrase="Fever",
                term="Fever",
                source="test",
            )
        ],
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        hpo_source="test.obo",
        hpo_sha256="a" * 64,
        addons_sha256=None,
        parser="test",
        parser_version="1",
        embedding_backend="fake",
        embedding_model="fake-model",
        embedding_revision="fake-revision",
    )


def _manifest(tmp_path: Path) -> dict[str, object]:
    return json.loads((tmp_path / "hpo_manifest.json").read_text(encoding="utf-8"))


def _replace_manifest(tmp_path: Path, manifest: dict[str, object]) -> None:
    (tmp_path / "hpo_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _replace_vectors(tmp_path: Path, vectors: np.ndarray) -> None:
    vector_path = tmp_path / "hpo_embedded.npz"
    np.savez_compressed(vector_path, emb=vectors)
    manifest = _manifest(tmp_path)
    manifest["vectors_sha256"] = sha256_file(vector_path)
    _replace_manifest(tmp_path, manifest)


def test_artifact_roundtrip_and_permissions(tmp_path: Path) -> None:
    _write(tmp_path)
    entries, matrix, manifest = load_artifacts(tmp_path)
    assert entries[0].hp_id == "HP:1"
    assert matrix.shape == (1, 2)
    assert manifest.dimension == 2
    assert os.stat(tmp_path / "hpo_manifest.json").st_mode & 0o777 == 0o600


def test_artifact_payload_hashes_are_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write(first)
    _write(second)
    first_manifest = json.loads((first / "hpo_manifest.json").read_text())
    second_manifest = json.loads((second / "hpo_manifest.json").read_text())
    assert first_manifest["metadata_sha256"] == second_manifest["metadata_sha256"]
    assert first_manifest["vectors_sha256"] == second_manifest["vectors_sha256"]


def test_artifact_count_mismatch_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="counts"):
        write_artifacts(
            tmp_path,
            entries=[],
            vectors=np.asarray([[1.0]], dtype=np.float32),
            hpo_source="test",
            hpo_sha256="a",
            addons_sha256=None,
            parser="test",
            parser_version="1",
            embedding_backend="fake",
            embedding_model="fake",
            embedding_revision="fake",
        )


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (np.asarray([1.0], dtype=np.float32), "two-dimensional"),
        (np.asarray([[np.nan]], dtype=np.float32), "non-finite"),
    ],
)
def test_artifact_writer_rejects_invalid_vectors(
    tmp_path: Path,
    vectors: np.ndarray,
    message: str,
) -> None:
    entries = [
        ArtifactEntry(
            hp_id="HP:1",
            phrase="Fever",
            term="Fever",
            source="test",
        )
    ]
    with pytest.raises(ValueError, match=message):
        write_artifacts(
            tmp_path,
            entries=entries,
            vectors=vectors,
            hpo_source="test",
            hpo_sha256="a",
            addons_sha256=None,
            parser="test",
            parser_version="1",
            embedding_backend="fake",
            embedding_model="fake",
            embedding_revision="fake",
        )


def test_artifact_tampering_fails(tmp_path: Path) -> None:
    _write(tmp_path)
    metadata = tmp_path / "hpo_meta.json"
    metadata.write_text(metadata.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_artifacts(tmp_path)


def test_artifact_vector_hash_tampering_fails(tmp_path: Path) -> None:
    _write(tmp_path)
    vectors = tmp_path / "hpo_embedded.npz"
    vectors.write_bytes(vectors.read_bytes() + b" ")
    with pytest.raises(ValueError, match="vector hash"):
        load_artifacts(tmp_path)


def test_artifact_schema_mismatches_fail(tmp_path: Path) -> None:
    _write(tmp_path)
    manifest = _manifest(tmp_path)
    manifest["schema_version"] = "99"
    _replace_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="unsupported artifact schema"):
        load_artifacts(tmp_path)

    _write(tmp_path)
    metadata_path = tmp_path / "hpo_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "99"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest = _manifest(tmp_path)
    manifest["metadata_sha256"] = sha256_file(metadata_path)
    _replace_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="metadata schema"):
        load_artifacts(tmp_path)


def test_artifact_shape_count_and_finite_mismatches_fail(tmp_path: Path) -> None:
    _write(tmp_path)
    _replace_vectors(tmp_path, np.asarray([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="metadata and vector counts"):
        load_artifacts(tmp_path)

    _write(tmp_path)
    manifest = _manifest(tmp_path)
    manifest["metadata_count"] = 2
    _replace_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="manifest counts"):
        load_artifacts(tmp_path)

    _write(tmp_path)
    manifest = _manifest(tmp_path)
    manifest["dimension"] = 3
    _replace_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="shape or dtype"):
        load_artifacts(tmp_path)

    _write(tmp_path)
    _replace_vectors(tmp_path, np.asarray([[np.nan, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        load_artifacts(tmp_path)


def test_state_roundtrip_and_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    result = AnnotationResult(
        patient_id="001",
        phrase="Fever",
        category=Category.ABNORMAL,
        hpo_id="HP:1",
        hpo_term="Fever",
        vector_score=1.0,
        mapping_status="mapped",
    )
    with PipelineState(
        path,
        input_sha256="input",
        artifact_sha256="artifact",
        pipeline_version="0.2.0",
        resume=False,
    ) as state:
        assert state.completed(0, "note") is None
        state.save_complete(0, "001", "note", [result])
        restored = state.completed(0, "note")
        assert restored == [result]
    assert os.stat(path).st_mode & 0o777 == 0o600

    with pytest.raises(StateMismatchError):
        PipelineState(
            path,
            input_sha256="different",
            artifact_sha256="artifact",
            pipeline_version="0.2.0",
            resume=True,
        )

    with PipelineState(
        path,
        input_sha256="input",
        artifact_sha256="artifact",
        pipeline_version="0.2.0",
        resume=False,
    ):
        pass
    with PipelineState(
        path,
        input_sha256="input",
        artifact_sha256="artifact",
        pipeline_version="0.2.0",
        resume=True,
    ):
        pass


def test_state_error_does_not_restore_as_complete(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with PipelineState(
        path,
        input_sha256="input",
        artifact_sha256="artifact",
        pipeline_version="0.2.0",
        resume=False,
    ) as state:
        state.save_error(0, "1", "note", "timeout", "provider timed out")
        assert state.completed(0, "note") is None
        row = state.connection.execute("SELECT results_json, error_code FROM rows").fetchone()
        assert row == (None, "timeout")


def test_state_rejects_changed_note_hash(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with PipelineState(
        path,
        input_sha256="input",
        artifact_sha256="artifact",
        pipeline_version="0.2.0",
        resume=False,
    ) as state:
        state.save_error(0, "1", "original", "timeout", "provider timed out")
        with pytest.raises(StateMismatchError, match="note hash changed"):
            state.completed(0, "changed")


def test_manifest_contains_no_source_content(tmp_path: Path) -> None:
    _write(tmp_path)
    manifest = json.loads((tmp_path / "hpo_manifest.json").read_text())
    assert "api_key" not in manifest
    assert "clinical_note" not in manifest
