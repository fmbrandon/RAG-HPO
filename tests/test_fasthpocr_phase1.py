from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_hpo import fasthpocr
from rag_hpo.fasthpocr import (
    FastHPOAnnotation,
    FastHPORecognizer,
    build_fasthpocr_index,
)


def test_fasthpocr_annotation_adapter_preserves_span_and_identifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index = tmp_path / "hp.index"
    index.write_text("{}", encoding="utf-8")

    class Annotation:
        def getTextSpan(self) -> str:
            return "short stature"

        def getHPOUri(self) -> str:
            return "HP:0004322"

        def getHPOLabel(self) -> str:
            return "Short stature"

        def getStartOffset(self) -> int:
            return 5

        def getEndOffset(self) -> int:
            return 18

    class Annotator:
        def __init__(self, _path: str) -> None:
            pass

        def annotate(self, text: str, *, longestMatch: bool) -> list[Annotation]:
            assert text == "Has short stature."
            assert longestMatch is True
            return [Annotation()]

    class Module:
        HPOAnnotator = Annotator

    monkeypatch.setattr("rag_hpo.fasthpocr.importlib.import_module", lambda _: Module)
    recognizer = FastHPORecognizer(index, longest_match=True)
    assert recognizer.annotate("Has short stature.") == [
        FastHPOAnnotation(
            phrase="short stature",
            hpo_id="HP:0004322",
            hpo_term="Short stature",
            start_offset=5,
            end_offset=18,
        )
    ]


def test_addon_file_shape_is_compatible_with_current_repository() -> None:
    addons = Path(__file__).parents[1] / "HPO_addons.csv"
    with addons.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert {"HP_ID", "info"} <= set(reader.fieldnames)


def test_build_fasthpocr_index_with_addons_and_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ontology = tmp_path / "hp.obo"
    ontology.write_text("format-version: 1.2\n", encoding="utf-8")
    addons = tmp_path / "addons.csv"
    addons.write_text(
        "HP_ID,info\nHP:0000001,first phrase\ninvalid,ignored\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class IndexHPO:
        def __init__(
            self,
            hpo_path: str,
            output_dir: str,
            *,
            indexConfig: dict[str, object],
        ) -> None:
            captured["hpo_path"] = hpo_path
            captured["output_dir"] = output_dir
            captured["config"] = indexConfig

        def index(self) -> None:
            output_dir = Path(str(captured["output_dir"]))
            external = Path(str(captured["config"]["externalSynFile"]))  # type: ignore[index]
            captured["external_content"] = external.read_text(encoding="utf-8")
            (output_dir / "hp.index").write_text('{"ok": true}', encoding="utf-8")
            print("fake FastHPOCR index complete")

    monkeypatch.setattr(fasthpocr.importlib.metadata, "version", lambda _: "0.1.4")
    monkeypatch.setattr(
        fasthpocr,
        "_expected_manifest",
        lambda _ontology, _addons, config, _registry: {
            "schema_version": "1.0",
            "fast_hpo_cr_version": "0.1.4",
            "ontology_sha256": "ontology",
            "addons_sha256": "addons",
            "config": dict(sorted(config.items())),
            "resources": {},
        },
    )
    monkeypatch.setattr(
        fasthpocr.importlib,
        "import_module",
        lambda _: SimpleNamespace(IndexHPO=IndexHPO),
    )

    index_dir = tmp_path / "index"
    index_path, manifest, reused = build_fasthpocr_index(
        ontology_path=ontology,
        index_dir=index_dir,
        addons_path=addons,
    )
    assert reused is False
    assert index_path.is_file()
    assert manifest["external_synonym_count"] == 1
    assert captured["external_content"] == "HP:0000001=first phrase\n"
    assert "fake FastHPOCR index complete" in (index_dir / "fasthpocr_index_build.log").read_text(
        encoding="utf-8"
    )

    repeated_path, repeated_manifest, repeated_reused = build_fasthpocr_index(
        ontology_path=ontology,
        index_dir=index_dir,
        addons_path=addons,
    )
    assert repeated_path == index_path
    assert repeated_manifest == manifest
    assert repeated_reused is True

    index_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="index hash"):
        build_fasthpocr_index(
            ontology_path=ontology,
            index_dir=index_dir,
            addons_path=addons,
        )


def test_fasthpocr_index_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ontology = tmp_path / "hp.obo"
    ontology.write_text("format-version: 1.2\n", encoding="utf-8")
    monkeypatch.setattr(fasthpocr.importlib.metadata, "version", lambda _: "9.9")
    with pytest.raises(RuntimeError, match=r"0\.1\.4"):
        build_fasthpocr_index(ontology_path=ontology, index_dir=tmp_path / "index")

    monkeypatch.setattr(fasthpocr.importlib.metadata, "version", lambda _: "0.1.4")
    with pytest.raises(FileNotFoundError, match="ontology"):
        build_fasthpocr_index(
            ontology_path=tmp_path / "missing.obo",
            index_dir=tmp_path / "index",
        )
    with pytest.raises(FileNotFoundError, match="add-on"):
        build_fasthpocr_index(
            ontology_path=ontology,
            index_dir=tmp_path / "index",
            addons_path=tmp_path / "missing.csv",
        )
    with pytest.raises(FileNotFoundError, match="index"):
        FastHPORecognizer(tmp_path / "missing.index")


def test_existing_index_with_different_provenance_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ontology = tmp_path / "hp.obo"
    ontology.write_text("format-version: 1.2\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "hp.index").write_text("{}", encoding="utf-8")
    (index_dir / "fasthpocr_index_manifest.json").write_text(
        json.dumps({"schema_version": "old"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(fasthpocr.importlib.metadata, "version", lambda _: "0.1.4")
    monkeypatch.setattr(
        fasthpocr,
        "_expected_manifest",
        lambda *_: {
            "schema_version": "1.0",
            "fast_hpo_cr_version": "0.1.4",
            "ontology_sha256": "ontology",
            "addons_sha256": None,
            "config": {},
            "resources": {},
        },
    )
    with pytest.raises(ValueError, match="provenance differs"):
        build_fasthpocr_index(ontology_path=ontology, index_dir=index_dir)
