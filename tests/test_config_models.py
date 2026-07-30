from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_hpo.config import DEFAULT_BASE_URL, DEFAULT_MODEL, ProviderConfig, ResponseMode
from rag_hpo.models import (
    AnnotationInput,
    Category,
    MappingDecision,
    MappingDecisionBatch,
    MappingSetDecision,
    MappingSetDecisionBatch,
    Phenotype,
    PhenotypeExtraction,
    PhenotypeSpan,
    _normalize_batch_dict,
)


def test_provider_config_uses_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_HPO_API_KEY", "test-key")
    config = ProviderConfig.from_env()
    assert config.base_url == DEFAULT_BASE_URL
    assert config.model == DEFAULT_MODEL
    assert config.response_mode is ResponseMode.STRICT
    assert config.redacted()["api_key"] == "configured"  # pragma: allowlist secret
    assert "test-key" not in str(config.redacted())


def test_provider_config_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_HPO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="RAG_HPO_API_KEY"):
        ProviderConfig.from_env()


def test_provider_config_rejects_insecure_url() -> None:
    with pytest.raises(ValidationError, match="HTTPS"):
        ProviderConfig(api_key="x", base_url="http://example.test/v1")


def test_unicode_and_string_ids_are_preserved() -> None:
    value = AnnotationInput(
        patient_id="001-A",
        clinical_note="café \N{EN DASH} µ",
    )
    assert value.patient_id == "001-A"
    assert value.clinical_note == "café \N{EN DASH} µ"


@pytest.mark.parametrize("field", ["patient_id", "clinical_note"])
def test_annotation_input_rejects_blanks(field: str) -> None:
    values = {"patient_id": "1", "clinical_note": "text"}
    values[field] = "   "
    with pytest.raises(ValidationError):
        AnnotationInput(**values)


def test_phenotype_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PhenotypeExtraction.model_validate(
            {
                "phenotypes": [
                    {
                        "phrase": "fever",
                        "category": "Abnormal",
                        "unexpected": True,
                    }
                ]
            }
        )


def test_phenotype_strips_phrase() -> None:
    phenotype = Phenotype(phrase="  fever  ", category=Category.ABNORMAL)
    assert phenotype.phrase == "fever"


def test_phenotype_rejects_whitespace_only_phrase() -> None:
    with pytest.raises(ValidationError, match="phrase must not be blank"):
        Phenotype(phrase="   ", category=Category.ABNORMAL)


def test_public_phenotype_categories_are_exactly_three() -> None:
    assert {value.value for value in Category} == {
        "Abnormal",
        "Normal",
        "Family History",
    }
    with pytest.raises(ValidationError):
        Phenotype(phrase="possible fever", category="Suspected")


def test_mapping_alternative_set_is_bounded_and_verdict_consistent() -> None:
    value = MappingSetDecision(
        mention_id="m1",
        candidate_hpo_ids=["HP:0000001", "HP:0000002"],
        verdict="ambiguous",
        confidence="medium",
    )
    assert len(value.candidate_hpo_ids) == 2
    with pytest.raises(ValidationError, match="at most 3"):
        MappingSetDecision(
            mention_id="m1",
            candidate_hpo_ids=[
                "HP:0000001",
                "HP:0000002",
                "HP:0000003",
                "HP:0000004",
            ],
            verdict="ambiguous",
            confidence="medium",
        )
    with pytest.raises(ValidationError, match="cannot retain"):
        MappingSetDecision(
            mention_id="m1",
            candidate_hpo_ids=["HP:0000001"],
            verdict="unsupported",
            confidence="high",
        )
    with pytest.raises(ValidationError, match="distinct"):
        MappingSetDecision(
            mention_id="m1",
            candidate_hpo_ids=["HP:0000001", "HP:0000001"],
            verdict="ambiguous",
            confidence="medium",
        )
    with pytest.raises(ValidationError, match="require candidate"):
        MappingSetDecision(
            mention_id="m1",
            candidate_hpo_ids=[],
            verdict="supported",
            confidence="medium",
        )


def test_span_only_extraction_strips_and_rejects_blank_phrases() -> None:
    assert PhenotypeSpan(phrase="  fever ").phrase == "fever"
    with pytest.raises(ValidationError, match="phrase must not be blank"):
        PhenotypeSpan(phrase="   ")


def test_mapping_decision_and_batch_normalization() -> None:
    d1 = MappingDecision.model_validate({"mention_id": "m1", "hpo_id": "HP:0001234"})
    assert d1.verdict == "supported"
    assert d1.confidence == "medium"

    d2 = MappingDecision.model_validate({"mention_id": "m2", "hpo_id": None})
    assert d2.verdict == "unsupported"

    b1 = MappingDecisionBatch.model_validate({"results": [{"mention_id": "m1", "hpo_id": "HP:0001234"}]})
    assert len(b1.decisions) == 1

    b2 = MappingDecisionBatch.model_validate({"decision": {"mention_id": "m1", "hpo_id": "HP:0001234"}})
    assert len(b2.decisions) == 1

    b3 = MappingDecisionBatch.model_validate({"items": [{"mention_id": "m1", "hpo_id": "HP:0001234"}]})
    assert len(b3.decisions) == 1

    b4 = MappingDecisionBatch.model_validate({"m1": {"hpo_id": "HP:0001234"}})
    assert len(b4.decisions) == 1

    ms = MappingSetDecision.model_validate({"mention_id": "m1", "candidate_hpo_ids": ["HP:0001234"]})
    assert ms.verdict == "supported"
    assert ms.confidence == "medium"

    ms_un = MappingSetDecision.model_validate({"mention_id": "m1", "candidate_hpo_ids": []})
    assert ms_un.verdict == "unsupported"

    b5 = MappingSetDecisionBatch.model_validate({"results": [{"mention_id": "m1", "candidate_hpo_ids": ["HP:0001234"]}]})
    assert len(b5.decisions) == 1


def test_models_edge_cases_for_100_percent_coverage() -> None:
    with pytest.raises(ValidationError):
        MappingDecision.model_validate(123)

    d = MappingDecision.model_validate({"mention_id": "m1", "verdict": "supported", "confidence": "high", "hpo_id": "HP:0001234"})
    assert d.confidence == "high"

    assert _normalize_batch_dict([1, 2, 3]) == [1, 2, 3]
    assert _normalize_batch_dict({"empty": 123}) == {"empty": 123}

    b2 = MappingDecisionBatch.model_validate({"m1": {"mention_id": "m1", "hpo_id": "HP:0001234"}})
    assert len(b2.decisions) == 1

    with pytest.raises(ValidationError):
        MappingSetDecision.model_validate(123)

    ms = MappingSetDecision.model_validate({"mention_id": "m1", "candidate_hpo_ids": ["HP:0001234"], "verdict": "supported", "confidence": "high"})
    assert ms.confidence == "high"
