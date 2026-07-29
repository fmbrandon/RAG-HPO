from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rag_hpo.artifacts import ArtifactEntry, sha256_file, write_artifacts
from rag_hpo.assertion import analyze_assertion
from rag_hpo.cli import build_parser
from rag_hpo.config import ProviderConfig
from rag_hpo.models import (
    AnnotationInput,
    Candidate,
    Category,
    MappingDecisionBatch,
    PhenotypeExtraction,
)
from rag_hpo.registry import (
    LEXICAL_MANIFEST_NAME,
    LEXICAL_NAME,
    REGISTRY_MANIFEST_NAME,
    REGISTRY_NAME,
    HPORegistry,
    RegistryConcept,
    RegistryPhrase,
    write_registry_bundle,
)
from rag_hpo.retrieval import HybridCandidateRetriever
from rag_hpo.staged_pipeline import (
    AnnotationMode,
    MappingPromptMode,
    Mention,
    StagedAnnotationPipeline,
    build_context_packet,
    sentence_spans,
)


class FakeBackend:
    name = "fake"
    model_id = "fake-model"
    revision = "fake-revision"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _phrase(hp_id: str, value: str, key: str) -> RegistryPhrase:
    return RegistryPhrase(
        record_key=key * 64,
        phrase=value,
        normalized_phrase=value.casefold(),
        source="hpo-label",
        scope="LABEL",
    )


def _registry() -> HPORegistry:
    return HPORegistry(
        data_version="test",
        concepts=[
            RegistryConcept(
                hp_id="HP:0000001",
                label="Fever",
                definition="Elevated body temperature.",
                phrases=[
                    _phrase("HP:0000001", "Fever", "1"),
                    _phrase("HP:0000001", "High temperature", "2"),
                ],
            ),
            RegistryConcept(
                hp_id="HP:0000002",
                label="Short stature",
                definition="Reduced body height.",
                phrases=[_phrase("HP:0000002", "Short stature", "3")],
            ),
            RegistryConcept(
                hp_id="HP:0000003",
                label="Old fever",
                obsolete=True,
                phrases=[_phrase("HP:0000003", "Old fever", "4")],
            ),
        ],
    )


def _vector_bundle(path: Path) -> tuple[list[ArtifactEntry], np.ndarray]:
    registry = _registry()
    write_registry_bundle(
        path,
        registry=registry,
        hpo_source="test",
        hpo_sha256="hpo",
        addons_sha256=None,
    )
    entries = [
        ArtifactEntry(
            hp_id="HP:0000001",
            phrase="Fever",
            term="Fever",
            source="test",
        ),
        ArtifactEntry(
            hp_id="HP:0000001",
            phrase="High temperature",
            term="Fever",
            source="test",
        ),
        ArtifactEntry(
            hp_id="HP:0000002",
            phrase="Short stature",
            term="Short stature",
            source="test",
        ),
        ArtifactEntry(
            hp_id="HP:0000003",
            phrase="Old fever",
            term="Old fever",
            source="test",
        ),
    ]
    matrix = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.8, 0.2], [1.0, 0.0]],
        dtype=np.float32,
    )
    write_artifacts(
        path,
        entries=entries,
        vectors=matrix,
        hpo_source="test",
        hpo_sha256="hpo",
        addons_sha256=None,
        parser="test",
        parser_version="1",
        embedding_backend="fake",
        embedding_model="fake-model",
        embedding_revision="fake-revision",
        registry_schema_version=registry.schema_version,
        registry_sha256=sha256_file(path / REGISTRY_NAME),
        registry_manifest_sha256=sha256_file(path / REGISTRY_MANIFEST_NAME),
        lexical_sha256=sha256_file(path / LEXICAL_NAME),
        lexical_manifest_sha256=sha256_file(path / LEXICAL_MANIFEST_NAME),
    )
    return entries, matrix


def test_hybrid_retrieval_returns_distinct_active_ids(tmp_path: Path) -> None:
    entries, matrix = _vector_bundle(tmp_path)
    retriever = HybridCandidateRetriever(
        registry=_registry(),
        entries=entries,
        matrix=matrix,
        backend=FakeBackend(),
    )
    candidates = retriever.retrieve("fever", distinct_limit=16)
    assert [value.hpo_id for value in candidates] == [
        "HP:0000001",
        "HP:0000002",
    ]
    assert candidates[0].lexical_rank == 1
    assert candidates[0].source_methods == ["sapbert", "lexical"]


def test_sentence_spans_preserve_absolute_offsets() -> None:
    text = " Fever occurred. Short stature was present!\nNormal hearing."
    values = sentence_spans(text)
    assert [text[value.start : value.end] for value in values] == [
        "Fever occurred.",
        "Short stature was present!",
        "Normal hearing.",
    ]


def test_context_packet_adds_bounded_adjacent_context_only_for_a_reason() -> None:
    note = "History:\nThe child was evaluated. This was associated with fever. Hearing was normal."
    start = note.index("fever")
    mention = Mention(
        phrase="fever",
        start=start,
        end=start + len("fever"),
        category=Category.ABNORMAL,
    )
    packet = build_context_packet(
        note,
        mention,
        analyze_assertion(note, mention.start, mention.end),
        [
            Candidate(hpo_id="HP:0000001", term="Fever", score=0.03),
            Candidate(hpo_id="HP:0000002", term="Recurrent fever", score=0.02),
        ],
    )
    assert packet.section == "History"
    assert packet.subject == "patient"
    assert packet.extended_context_reason == "context-cue"
    assert packet.preceding_sentence
    assert packet.following_sentence == "Hearing was normal."


class StagedProvider:
    def __init__(
        self,
        *,
        incomplete_mapping: bool = False,
        usage_requests: int = 0,
    ) -> None:
        self.calls: list[str] = []
        self.incomplete_mapping = incomplete_mapping
        self.usage = {
            "requests": usage_requests,
            "input_tokens": usage_requests * 10,
            "output_tokens": usage_requests * 2,
            "total_tokens": usage_requests * 12,
        }

    def request(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> tuple[Any, str]:
        del temperature
        if response_model is PhenotypeExtraction:
            if "missed" in system_message:
                self.calls.append("audit")
                value = PhenotypeExtraction.model_validate(
                    {
                        "phenotypes": [
                            {
                                "phrase": "short stature",
                                "category": "Abnormal",
                            }
                        ]
                    }
                )
            else:
                self.calls.append("extract")
                value = PhenotypeExtraction.model_validate(
                    {
                        "phenotypes": [
                            {
                                "phrase": "Fever",
                                "category": "Abnormal",
                            }
                        ]
                    }
                )
            return value, value.model_dump_json()
        assert response_model is MappingDecisionBatch
        self.calls.append("map")
        payload = json.loads(user_message)
        decisions = []
        if not self.incomplete_mapping:
            for item in payload["items"]:
                selected = "HP:0000001" if item["phrase"].casefold() == "fever" else "HP:0000002"
                decisions.append(
                    {
                        "mention_id": item["mention_id"],
                        "hpo_id": selected,
                        "verdict": "supported",
                        "confidence": "medium",
                    }
                )
        value = MappingDecisionBatch.model_validate({"decisions": decisions})
        return value, value.model_dump_json()


def test_staged_pipeline_runs_two_passes_and_batched_mapping(tmp_path: Path) -> None:
    vector_dir = tmp_path / "vectors"
    output_dir = tmp_path / "output"
    _vector_bundle(vector_dir)
    provider = StagedProvider()
    pipeline = StagedAnnotationPipeline(
        provider=provider,  # type: ignore[arg-type]
        vector_dir=vector_dir,
        output_dir=output_dir,
        mode=AnnotationMode.BALANCED,
        recognizers={"native"},
        fasthpocr_index=None,
        resume=False,
        keep_state=False,
        keep_raw_responses=False,
        include_evidence_text=False,
        offline=False,
        backend=FakeBackend(),
    )
    results = pipeline.run(
        [
            AnnotationInput(
                patient_id="A-1",
                clinical_note="Fever and short stature were present.",
            )
        ]
    )
    mapped = [value for value in results if value.mapping_status == "mapped"]
    assert {value.hpo_id for value in mapped} == {"HP:0000001"}
    assert all(value.review_status == "accepted" for value in mapped)
    assert all(value.evidence_text is None for value in mapped)
    review = [value for value in results if value.review_status == "review"]
    assert review[0].candidate_hpo_ids
    assert "HP:0000002" in review[0].candidate_hpo_ids
    assert provider.calls == ["extract", "audit", "map"]
    exported = json.loads((output_dir / "rag_hpo_results.json").read_text())
    assert exported[0]["evidence_start"] == 0
    assert "source_methods" in exported[0]
    run_manifest = (output_dir / "rag_hpo_run_manifest.json").read_text(encoding="utf-8")
    assert "Fever and short stature" not in run_manifest
    assert json.loads(run_manifest)["accepted_rows"] == 1


def test_native_mode_requires_no_provider_and_retains_assertion_flag(
    tmp_path: Path,
) -> None:
    vector_dir = tmp_path / "vectors"
    _vector_bundle(vector_dir)
    pipeline = StagedAnnotationPipeline(
        provider=None,
        vector_dir=vector_dir,
        output_dir=tmp_path / "output",
        mode=AnnotationMode.NATIVE,
        recognizers={"native"},
        fasthpocr_index=None,
        resume=False,
        keep_state=False,
        keep_raw_responses=False,
        include_evidence_text=True,
        offline=True,
        backend=FakeBackend(),
    )
    results = pipeline.run(
        [AnnotationInput(patient_id="1", clinical_note="The patient denies fever.")]
    )
    assert results[0].hpo_id == "HP:0000001"
    assert results[0].review_status == "review"
    assert results[0].assertion_status == "negated"
    assert results[0].evidence_text


def test_incomplete_batched_mapping_is_a_row_error(tmp_path: Path) -> None:
    vector_dir = tmp_path / "vectors"
    _vector_bundle(vector_dir)
    pipeline = StagedAnnotationPipeline(
        provider=StagedProvider(incomplete_mapping=True),  # type: ignore[arg-type]
        vector_dir=vector_dir,
        output_dir=tmp_path / "output",
        mode=AnnotationMode.BALANCED,
        recognizers={"native"},
        fasthpocr_index=None,
        resume=False,
        keep_state=False,
        keep_raw_responses=False,
        include_evidence_text=False,
        offline=False,
        backend=FakeBackend(),
    )
    results = pipeline.run([AnnotationInput(patient_id="1", clinical_note="Fever was present.")])
    assert results[0].mapping_status == "error"
    assert results[0].error_message
    assert "incomplete" in results[0].error_message


def test_resume_manifest_accumulates_attempt_usage(tmp_path: Path) -> None:
    vector_dir = tmp_path / "vectors"
    output_dir = tmp_path / "output"
    _vector_bundle(vector_dir)
    row = AnnotationInput(patient_id="1", clinical_note="Fever was present.")
    for resume, requests in ((False, 2), (True, 3)):
        pipeline = StagedAnnotationPipeline(
            provider=StagedProvider(usage_requests=requests),  # type: ignore[arg-type]
            vector_dir=vector_dir,
            output_dir=output_dir,
            mode=AnnotationMode.BALANCED,
            recognizers={"native"},
            fasthpocr_index=None,
            resume=resume,
            keep_state=True,
            keep_raw_responses=False,
            include_evidence_text=False,
            offline=False,
            backend=FakeBackend(),
        )
        pipeline.run([row])

    manifest = json.loads((output_dir / "rag_hpo_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_attempt_count"] == 2
    assert manifest["provider_usage"]["requests"] == 5
    assert manifest["provider_usage"]["total_tokens"] == 60
    assert len(manifest["attempts"]) == 2


def test_cli_exposes_modes_and_repeatable_recognizers() -> None:
    args = build_parser().parse_args(
        [
            "annotate",
            "--text",
            "fever",
            "--vector-dir",
            "vectors",
            "--output-dir",
            "out",
            "--mode",
            "high-recall",
            "--recognizer",
            "native",
            "--recognizer",
            "fasthpocr",
            "--offline",
            "--no-model",
            "--mapping-prompt",
            "one-shot",
        ]
    )
    assert args.mode == "high-recall"
    assert args.recognizer == ["native", "fasthpocr"]
    assert args.offline is True
    assert args.no_model is True
    assert args.mapping_prompt == MappingPromptMode.ONE_SHOT.value


def test_provider_config_allows_only_http_loopback() -> None:
    assert (
        ProviderConfig(api_key="x", base_url="http://127.0.0.1:8000/v1").base_url
        == "http://127.0.0.1:8000/v1"
    )
