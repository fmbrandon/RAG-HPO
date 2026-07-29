from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_hpo.config import DEFAULT_BASE_URL, DEFAULT_MODEL, ProviderConfig, ResponseMode
from rag_hpo.models import (
    AnnotationInput,
    Category,
    MappingSetDecision,
    Phenotype,
    PhenotypeExtraction,
    PhenotypeSpan,
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
