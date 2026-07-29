from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rag_hpo import fasthpocr, ontology
from rag_hpo.artifacts import load_artifacts, sha256_file
from rag_hpo.fasthpocr import FastHPORecognizer
from rag_hpo.registry import (
    LEXICAL_NAME,
    REGISTRY_MANIFEST_NAME,
    REGISTRY_NAME,
    build_registry,
    load_registry_bundle,
    normalize_phrase,
    registry_phrase_index,
    validate_utf8_file,
    write_registry_bundle,
)

OBO = """format-version: 1.2
ontology: hp
data-version: hp/releases/2026-07-01

[Term]
id: HP:0000001
name: All

[Term]
id: HP:0000118
name: Phenotypic abnormality
is_a: HP:0000001 ! All

[Term]
id: HP:0001000
name: Breathing difficulty
alt_id: HP:9001000
def: "Difficulty breathing." []
synonym: "Difficult breathing" EXACT []
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0001001
name: Labored respiration
synonym: "Difficult breathing" RELATED []
is_a: HP:0000118 ! Phenotypic abnormality
"""


def _files(tmp_path: Path) -> tuple[Path, Path]:
    obo = tmp_path / "hp.obo"
    obo.write_text(OBO, encoding="utf-8")
    addons = tmp_path / "addons.csv"
    addons.write_text(
        "HP_ID,info\nHP:9001000,Hard to breathe\n",
        encoding="utf-8",
    )
    return obo, addons


def test_registry_is_deterministic_and_preserves_provenance(tmp_path: Path) -> None:
    obo, addons = _files(tmp_path)
    first = build_registry(obo, addons_path=addons, limit=None)
    second = build_registry(obo, addons_path=addons, limit=None)
    assert first == second
    assert first.data_version == "hp/releases/2026-07-01"

    concept = next(item for item in first.concepts if item.hp_id == "HP:0001000")
    assert concept.alternate_ids == ["HP:9001000"]
    assert concept.parents == ["HP:0000118"]
    assert {phrase.scope for phrase in concept.phrases} == {"LABEL", "EXACT"}
    assert first.extensions[0].hp_id == "HP:0001000"
    assert first.extensions[0].source == "rag-hpo-addon"

    ambiguity = next(
        group
        for group in first.ambiguity_groups
        if group.normalized_phrase == "difficult breathing"
    )
    assert ambiguity.hp_ids == ["HP:0001000", "HP:0001001"]
    assert registry_phrase_index(first)["difficult breathing"] == (
        "HP:0001000",
        "HP:0001001",
    )

    hashes: list[tuple[str, str]] = []
    for name in ("first", "second"):
        output = tmp_path / name
        write_registry_bundle(
            output,
            registry=first,
            hpo_source="pinned-hp.obo",
            hpo_sha256=sha256_file(obo),
            addons_sha256=sha256_file(addons),
        )
        loaded, registry_manifest, lexical_manifest = load_registry_bundle(output)
        assert loaded == first
        hashes.append((registry_manifest.registry_sha256, lexical_manifest.lexical_sha256))
    assert hashes[0] == hashes[1]


def test_registry_validation_rejects_encoding_and_tampering(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.obo"
    invalid.write_bytes(b"name: invalid " + bytes([0xFF]))
    with pytest.raises(ValueError, match="UTF-8"):
        validate_utf8_file(invalid)

    mojibake = tmp_path / "mojibake.obo"
    mojibake.write_text("name: patientâ€™s finding", encoding="utf-8")
    with pytest.raises(ValueError, match="mojibake"):
        validate_utf8_file(mojibake)

    obo, addons = _files(tmp_path)
    registry = build_registry(obo, addons_path=addons, limit=None)
    output = tmp_path / "registry"
    write_registry_bundle(
        output,
        registry=registry,
        hpo_source="pinned-hp.obo",
        hpo_sha256=sha256_file(obo),
        addons_sha256=sha256_file(addons),
    )
    lexical = output / LEXICAL_NAME
    lexical.write_text(lexical.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="lexical index hash"):
        load_registry_bundle(output)


def test_vectorize_links_dense_artifacts_to_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obo, addons = _files(tmp_path)

    class Backend:
        name = "fake"
        model_id = "fake-model"
        revision = "fixed"

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(ontology, "create_backend", lambda _: Backend())
    output = tmp_path / "vectors"
    manifest = ontology.vectorize(
        output_dir=output,
        obo_file=obo,
        addons_path=addons,
        backend_name="fake",
        offline=True,
    )
    assert manifest.registry_sha256 == sha256_file(output / REGISTRY_NAME)
    assert manifest.registry_manifest_sha256 == sha256_file(output / REGISTRY_MANIFEST_NAME)
    entries, matrix, loaded_manifest = load_artifacts(output)
    assert len(entries) == matrix.shape[0]
    assert loaded_manifest.registry_sha256 == manifest.registry_sha256

    registry_path = output / REGISTRY_NAME
    registry_path.write_text(
        registry_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="registry"):
        load_artifacts(output)


def test_fasthpocr_adapter_emits_deterministic_ambiguity_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obo, addons = _files(tmp_path)
    registry = build_registry(obo, addons_path=addons, limit=None)
    index = tmp_path / "hp.index"
    index.write_text("{}", encoding="utf-8")

    class Annotation:
        def __init__(self, phrase: str, uri: str) -> None:
            self.phrase = phrase
            self.uri = uri

        def getTextSpan(self) -> str:
            return self.phrase

        def getHPOUri(self) -> str:
            return self.uri

        def getHPOLabel(self) -> str:
            return "upstream label"

        def getStartOffset(self) -> int:
            return 0

        def getEndOffset(self) -> int:
            return len(self.phrase)

    class Annotator:
        def __init__(self, _: str) -> None:
            pass

        def annotate(self, text: str, *, longestMatch: bool) -> list[Annotation]:
            assert longestMatch
            if text == "alias":
                return [Annotation("unregistered wording", "HP:9001000")]
            return [Annotation("Difficult breathing", "HP:0001001")]

    monkeypatch.setattr(
        fasthpocr.importlib,
        "import_module",
        lambda _: SimpleNamespace(HPOAnnotator=Annotator),
    )
    recognizer = FastHPORecognizer(index, longest_match=True, registry=registry)
    ambiguous = recognizer.annotate("ambiguous")[0]
    assert ambiguous.hpo_id == ""
    assert ambiguous.candidate_hpo_ids == ("HP:0001000", "HP:0001001")
    assert ambiguous.resolution == "ambiguous"

    alias = recognizer.annotate("alias")[0]
    assert alias.hpo_id == "HP:0001000"
    assert alias.hpo_term == "Breathing difficulty"
    assert alias.resolution == "registry"


def test_fasthpocr_adapter_retains_morphological_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obo, addons = _files(tmp_path)
    registry = build_registry(obo, addons_path=addons, limit=None)
    index = tmp_path / "hp.index"
    index.write_text(
        json.dumps(
            {
                "clusters": {
                    "C1": ["difficult", "difficulty"],
                    "C2": ["breathing", "respiration"],
                },
                "termData": [
                    {
                        "uri": hp_id,
                        "labels": [{"tokenSet": ["C1", "C2"]}],
                    }
                    for hp_id in ("HP:0001000", "HP:0001001")
                ],
            }
        ),
        encoding="utf-8",
    )

    class Annotation:
        def getTextSpan(self) -> str:
            return "Difficult respiration"

        def getHPOUri(self) -> str:
            return "HP:0001001"

        def getHPOLabel(self) -> str:
            return "upstream"

        def getStartOffset(self) -> int:
            return 0

        def getEndOffset(self) -> int:
            return 21

    class Annotator:
        def __init__(self, _: str) -> None:
            pass

        def annotate(self, _: str, *, longestMatch: bool) -> list[Annotation]:
            assert longestMatch
            return [Annotation()]

    monkeypatch.setattr(
        fasthpocr.importlib,
        "import_module",
        lambda _: SimpleNamespace(HPOAnnotator=Annotator),
    )
    annotation = FastHPORecognizer(
        index,
        longest_match=True,
        registry=registry,
    ).annotate("Difficult respiration")[0]
    assert annotation.hpo_id == ""
    assert annotation.candidate_hpo_ids == ("HP:0001000", "HP:0001001")
    assert annotation.resolution == "ambiguous"


def test_semantic_index_fingerprint_ignores_container_order(tmp_path: Path) -> None:
    label_one = {
        "native": True,
        "originalLabel": "Breathing difficulty",
        "tokens": ["C1", "C2"],
        "tokenSet": ["C2", "C1"],
    }
    label_two = {
        "native": False,
        "originalLabel": "Difficult breathing",
        "tokens": ["C2"],
        "tokenSet": ["C2"],
    }
    first = {
        "clusters": {"C1": ["breathing"], "C2": ["difficult", "difficulty"]},
        "termData": [
            {
                "uri": "HP:0001000",
                "categories": ["HP:2", "HP:1"],
                "labels": [label_one, label_two],
            }
        ],
    }
    second = {
        "termData": [
            {
                "labels": [label_two, label_one],
                "categories": ["HP:1", "HP:2"],
                "uri": "HP:0001000",
            }
        ],
        "clusters": {"C2": ["difficulty", "difficult"], "C1": ["breathing"]},
    }
    paths = []
    for name, value in (("first", first), ("second", second)):
        path = tmp_path / f"{name}.index"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    one = fasthpocr._semantic_index_fingerprint(paths[0])
    two = fasthpocr._semantic_index_fingerprint(paths[1])
    assert one == two
    assert one["term_count"] == 1
    assert one["label_count"] == 2


def test_phrase_normalization_is_unicode_and_punctuation_stable() -> None:
    assert normalize_phrase(" Café-au-lait! ") == "café au lait"
