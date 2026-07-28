from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np

from rag_hpo.artifacts import ArtifactEntry, write_artifacts


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "build_vector_bundle.py"
    spec = importlib.util.spec_from_file_location("build_vector_bundle", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_release_vector_bundle_is_reproducible_across_build_timestamps(
    tmp_path: Path,
) -> None:
    module = _module()
    first_vectors = tmp_path / "first-vectors"
    second_vectors = tmp_path / "second-vectors"
    _vectors(first_vectors)
    _vectors(second_vectors)
    first_bundle = tmp_path / "first.zip"
    second_bundle = tmp_path / "second.zip"
    module.build(first_vectors, first_bundle)
    module.build(second_vectors, second_bundle)
    assert first_bundle.read_bytes() == second_bundle.read_bytes()

    with zipfile.ZipFile(first_bundle) as archive:
        manifest = json.loads(archive.read("hpo_manifest.json"))
    assert manifest["created_at"] == "1970-01-01T00:00:00+00:00"
