from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rag_hpo import registry as registry_module
from rag_hpo.artifacts import (
    ArtifactEntry,
    load_artifacts,
    sha256_file,
    write_artifacts,
)
from rag_hpo.assertion import analyze_assertion
from rag_hpo.cli import build_parser
from rag_hpo.config import ProviderConfig
from rag_hpo.models import (
    AnnotationInput,
    Candidate,
    Category,
    FinalCategoryDecisionBatch,
    MappingSetDecisionBatch,
    PhenotypeSpanExtraction,
)
from rag_hpo.provider import ProviderError
from rag_hpo.registry import (
    LEXICAL_MANIFEST_NAME,
    LEXICAL_NAME,
    REGISTRY_MANIFEST_NAME,
    REGISTRY_NAME,
    HPORegistry,
    RegistryConcept,
    RegistryPhrase,
    load_registry_bundle,
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


def test_linked_artifact_manifest_rejects_incomplete_and_unknown_registry(
    tmp_path: Path,
) -> None:
    _vector_bundle(tmp_path)
    manifest_path = tmp_path / "hpo_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["registry_sha256"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete registry provenance"):
        load_artifacts(tmp_path)

    _vector_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["registry_schema_version"] = "99"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported registry schema"):
        load_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("manifest_kind", "field", "message"),
    [
        ("registry", "registry_sha256", "registry identity"),
        ("lexical", "lexical_sha256", "lexical identity"),
    ],
)
def test_linked_artifact_rejects_loaded_manifest_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_kind: str,
    field: str,
    message: str,
) -> None:
    _vector_bundle(tmp_path)
    registry, registry_manifest, lexical_manifest = load_registry_bundle(tmp_path)
    if manifest_kind == "registry":
        registry_manifest = registry_manifest.model_copy(update={field: "mismatch"})
    else:
        lexical_manifest = lexical_manifest.model_copy(update={field: "mismatch"})
    monkeypatch.setattr(
        registry_module,
        "load_registry_bundle",
        lambda _: (registry, registry_manifest, lexical_manifest),
    )
    with pytest.raises(ValueError, match=message):
        load_artifacts(tmp_path)


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
        final_category: str = "Abnormal",
        ambiguous_mapping: bool = False,
        invalid_mapping: bool = False,
        incomplete_final: bool = False,
        reject_mapping: bool = False,
        reject_final: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.incomplete_mapping = incomplete_mapping
        self.final_category = final_category
        self.ambiguous_mapping = ambiguous_mapping
        self.invalid_mapping = invalid_mapping
        self.incomplete_final = incomplete_final
        self.reject_mapping = reject_mapping
        self.reject_final = reject_final
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
        if response_model is PhenotypeSpanExtraction:
            if "missed" in system_message:
                self.calls.append("audit")
                value = PhenotypeSpanExtraction.model_validate(
                    {
                        "phenotypes": [
                            {
                                "phrase": "short stature",
                            }
                        ]
                    }
                )
            else:
                self.calls.append("extract")
                value = PhenotypeSpanExtraction.model_validate(
                    {
                        "phenotypes": [
                            {
                                "phrase": "Fever",
                            }
                        ]
                    }
                )
            return value, value.model_dump_json()
        payload = json.loads(user_message)
        if response_model is FinalCategoryDecisionBatch:
            self.calls.append("categorize")
            if self.reject_final:
                raise ProviderError("invalid_request", "rejected", 400)
            value = FinalCategoryDecisionBatch.model_validate(
                {
                    "decisions": [
                        {
                            "mention_id": item["mention_id"],
                            "category": self.final_category,
                            "confidence": "high",
                        }
                        for item in ([] if self.incomplete_final else payload["items"])
                    ]
                }
            )
            return value, value.model_dump_json()
        assert response_model is MappingSetDecisionBatch
        self.calls.append("map")
        if self.reject_mapping:
            raise ProviderError("invalid_request", "rejected", 400)
        decisions = []
        if not self.incomplete_mapping:
            for item in payload["items"]:
                selected = "HP:0000001" if item["phrase"].casefold() == "fever" else "HP:0000002"
                if self.invalid_mapping:
                    selected = "HP:9999999"
                decisions.append(
                    {
                        "mention_id": item["mention_id"],
                        "candidate_hpo_ids": (
                            ["HP:0000001", "HP:0000002"] if self.ambiguous_mapping else [selected]
                        ),
                        "verdict": "ambiguous" if self.ambiguous_mapping else "supported",
                        "confidence": "medium",
                    }
                )
        value = MappingSetDecisionBatch.model_validate({"decisions": decisions})
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
    assert {value.hpo_id for value in mapped} == {"HP:0000001", "HP:0000002"}
    assert all(value.review_status == "accepted" for value in mapped)
    assert all(value.evidence_text is None for value in mapped)
    assert provider.calls == ["extract", "audit", "map", "categorize"]
    exported = json.loads((output_dir / "rag_hpo_results.json").read_text())
    assert exported[0]["evidence_start"] == 0
    assert "source_methods" in exported[0]
    run_manifest = (output_dir / "rag_hpo_run_manifest.json").read_text(encoding="utf-8")
    assert "Fever and short stature" not in run_manifest
    assert json.loads(run_manifest)["accepted_rows"] == 2


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
    assert results[0].hpo_id is None
    assert results[0].category is Category.NORMAL
    assert results[0].mapping_status == "not_mapped_category"
    assert results[0].review_status == "accepted"
    assert results[0].assertion_status == "negated"
    assert results[0].evidence_text


def test_clinical_modifier_ids_are_never_standalone_mappings(tmp_path: Path) -> None:
    vector_dir = tmp_path / "vectors"
    _vector_bundle(vector_dir)
    pipeline = StagedAnnotationPipeline(
        provider=StagedProvider(),  # type: ignore[arg-type]
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
    # The fixture calls Fever a modifier to exercise the production filter
    # without adding benchmark-specific HPO content to the test registry.
    pipeline._modifier_ids = {"HP:0000001"}
    results = pipeline.run([AnnotationInput(patient_id="1", clinical_note="Fever was present.")])
    assert not any(result.hpo_id == "HP:0000001" for result in results)
    assert results[0].modifier_hpo_ids == ["HP:0000001"]
    assert "HP:0000001" not in (results[0].retrieval_candidate_hpo_ids or [])


def test_final_categorization_can_remove_a_false_abnormal_mapping(
    tmp_path: Path,
) -> None:
    vector_dir = tmp_path / "vectors"
    _vector_bundle(vector_dir)
    pipeline = StagedAnnotationPipeline(
        provider=StagedProvider(final_category="Normal"),  # type: ignore[arg-type]
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
    results = pipeline.run([AnnotationInput(patient_id="1", clinical_note="Fever was considered.")])
    assert results[0].category is Category.NORMAL
    assert results[0].category_confidence == "high"
    assert results[0].hpo_id is None
    assert results[0].mapping_status == "not_mapped_category"
    assert results[0].assertion_status == "normal"


def test_nested_context_span_merges_into_single_phenotype_span() -> None:
    mentions = StagedAnnotationPipeline._merge_mentions(
        [
            Mention(
                phrase="Possible scoliosis",
                start=0,
                end=18,
                methods={"model-pass-1"},
            ),
            Mention(
                phrase="scoliosis",
                start=9,
                end=18,
                methods={"native"},
                recognizer_ids={"HP:0002650"},
            ),
        ]
    )
    assert len(mentions) == 1
    assert mentions[0].phrase == "Possible scoliosis"
    assert mentions[0].methods == {"model-pass-1", "native"}
    assert mentions[0].recognizer_ids == {"HP:0002650"}
    assert mentions[0].phrase_variants == {"Possible scoliosis", "scoliosis"}


def test_non_contiguous_coordinated_span_retains_exact_evidence_segments() -> None:
    note = "There were multiple millimetric cutaneous and some larger subcutaneous neurofibromas."
    mentions = StagedAnnotationPipeline._validated_extraction(
        note,
        PhenotypeSpanExtraction.model_validate(
            {"phenotypes": [{"phrase": "multiple millimetric cutaneous neurofibromas"}]}
        ),
        "model-pass-1",
    )
    assert len(mentions) == 1
    assert mentions[0].phrase == "multiple millimetric cutaneous neurofibromas"
    assert [note[start:end] for start, end in mentions[0].evidence_segments] == [
        "multiple millimetric cutaneous",
        "neurofibromas",
    ]


def test_independent_imaging_clause_does_not_inherit_normal_assertion() -> None:
    note = (
        "Thyroid ultrasound was within normal limits and abdominal CT was "
        "significant for left-sided mesenteric mass."
    )
    start = note.index("mesenteric mass")
    decision = analyze_assertion(note, start, start + len("mesenteric mass"))
    assert decision.status == "affirmed"


def test_short_phrase_retrieval_batches_phrase_and_local_context(tmp_path: Path) -> None:
    class RecordingBackend(FakeBackend):
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts: list[str]) -> np.ndarray:
            self.calls.append(texts)
            return super().encode(texts)

    entries, matrix = _vector_bundle(tmp_path)
    backend = RecordingBackend()
    retriever = HybridCandidateRetriever(
        registry=_registry(),
        entries=entries,
        matrix=matrix,
        backend=backend,
    )
    retriever.retrieve_many(
        ["tubular adenoma"],
        contexts=["tubular adenoma Colonoscopy found polyps. Biopsy showed tubular adenoma."],
        distinct_limit=16,
    )
    retriever.retrieve_many(
        ["tubular adenoma"],
        contexts=["tubular adenoma Colonoscopy found polyps. Biopsy showed tubular adenoma."],
        distinct_limit=16,
    )
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == "tubular adenoma"
    assert "Colonoscopy" in backend.calls[0][1]
    assert retriever.cache_info()["candidate_hits"] == 1
    assert retriever.cache_info()["candidate_entries"] == 1


def test_context_only_top_match_survives_sixteen_candidate_bound() -> None:
    dense = {f"HP:{index:07d}": (index, 1.0 - index / 100) for index in range(1, 21)}
    context_id = "HP:9999999"
    ranked = HybridCandidateRetriever._fuse(
        dense,
        {},
        {context_id: (1, 0.95)},
    )
    assert context_id in [candidate.hpo_id for candidate in ranked[:16]]


def test_assertion_cues_inside_extracted_span_are_retained() -> None:
    normal = analyze_assertion("Hearing is normal.", 0, 17)
    family = analyze_assertion("His mother had seizures.", 0, 23)
    uncertain = analyze_assertion("Possible scoliosis.", 0, 18)
    assert normal.status == "normal"
    assert family.status == "family_history"
    assert uncertain.status == "uncertain"


def test_ambiguous_mapping_retains_at_most_three_alternatives_as_one_finding(
    tmp_path: Path,
) -> None:
    vector_dir = tmp_path / "vectors"
    _vector_bundle(vector_dir)
    pipeline = StagedAnnotationPipeline(
        provider=StagedProvider(ambiguous_mapping=True),  # type: ignore[arg-type]
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
    assert len(results) == 1
    assert results[0].hpo_id is None
    assert results[0].candidate_hpo_ids == ["HP:0000001", "HP:0000002"]
    assert results[0].mapping_status == "mapped"
    assert results[0].mapping_verdict == "ambiguous"
    assert results[0].review_status == "accepted"


def test_incomplete_batched_mapping_is_retained_for_review(tmp_path: Path) -> None:
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
    assert results[0].mapping_status == "no_candidate_fit"
    assert results[0].review_status == "review"
    assert results[0].error_code == "incomplete_mapping_decision"


def test_out_of_scope_mapping_id_is_discarded_without_failing_row(tmp_path: Path) -> None:
    vector_dir = tmp_path / "vectors"
    _vector_bundle(vector_dir)
    pipeline = StagedAnnotationPipeline(
        provider=StagedProvider(invalid_mapping=True),  # type: ignore[arg-type]
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
    assert results[0].mapping_status == "no_candidate_fit"
    assert results[0].candidate_hpo_ids == []
    assert results[0].review_status == "review"
    assert results[0].error_code == "mapping_candidate_out_of_scope"


def test_incomplete_final_category_keeps_abnormal_finding_for_review(
    tmp_path: Path,
) -> None:
    vector_dir = tmp_path / "vectors"
    _vector_bundle(vector_dir)
    pipeline = StagedAnnotationPipeline(
        provider=StagedProvider(incomplete_final=True),  # type: ignore[arg-type]
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
    assert results[0].category is Category.ABNORMAL
    assert results[0].category_confidence == "low"
    assert results[0].review_status == "review"
    assert results[0].error_code == "incomplete_final_category"


@pytest.mark.parametrize("rejected_stage", ["mapping", "final"])
def test_unsplittable_provider_400_is_a_finding_level_review(
    tmp_path: Path,
    rejected_stage: str,
) -> None:
    vector_dir = tmp_path / "vectors"
    _vector_bundle(vector_dir)
    pipeline = StagedAnnotationPipeline(
        provider=StagedProvider(
            reject_mapping=rejected_stage == "mapping",
            reject_final=rejected_stage == "final",
        ),  # type: ignore[arg-type]
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
    assert all(result.mapping_status != "error" for result in results)
    assert results[0].category is Category.ABNORMAL
    assert results[0].review_status == "review"
    assert results[0].error_code in {
        "incomplete_mapping_decision",
        "incomplete_final_category",
    }


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


def test_internal_retry_reuses_successful_provider_stages_and_retrieval(
    tmp_path: Path,
) -> None:
    class RetryFinalProvider(StagedProvider):
        def __init__(self) -> None:
            super().__init__()
            self.failed_final = False

        def request(
            self,
            *,
            system_message: str,
            user_message: str,
            response_model: type[Any],
            temperature: float = 0.2,
        ) -> tuple[Any, str]:
            if response_model is FinalCategoryDecisionBatch and not self.failed_final:
                self.calls.append("categorize")
                self.failed_final = True
                raise ProviderError("rate_limit", "retry later", 429)
            return super().request(
                system_message=system_message,
                user_message=user_message,
                response_model=response_model,
                temperature=temperature,
            )

    vector_dir = tmp_path / "vectors"
    output_dir = tmp_path / "output"
    _vector_bundle(vector_dir)
    provider = RetryFinalProvider()
    pipeline = StagedAnnotationPipeline(
        provider=provider,  # type: ignore[arg-type]
        vector_dir=vector_dir,
        output_dir=output_dir,
        mode=AnnotationMode.BALANCED,
        recognizers={"native"},
        fasthpocr_index=None,
        resume=True,
        keep_state=True,
        keep_raw_responses=False,
        include_evidence_text=False,
        offline=False,
        backend=FakeBackend(),
    )
    results = pipeline.run(
        [AnnotationInput(patient_id="1", clinical_note="Fever was present.")],
        max_attempts=2,
    )

    assert all(result.mapping_status != "error" for result in results)
    assert provider.calls == ["extract", "map", "categorize", "categorize"]
    manifest = json.loads((output_dir / "rag_hpo_run_manifest.json").read_text())
    assert manifest["run_attempt_count"] == 2
    assert manifest["cache"]["provider_response_hits"] == 2
    assert manifest["cache"]["candidate_hits"] >= 1


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
            "--max-row-attempts",
            "3",
        ]
    )
    assert args.mode == "high-recall"
    assert args.recognizer == ["native", "fasthpocr"]
    assert args.offline is True
    assert args.no_model is True
    assert args.mapping_prompt == MappingPromptMode.ONE_SHOT.value
    assert args.max_row_attempts == 3


def test_provider_config_allows_only_http_loopback() -> None:
    assert (
        ProviderConfig(api_key="x", base_url="http://127.0.0.1:8000/v1").base_url
        == "http://127.0.0.1:8000/v1"
    )
