from __future__ import annotations

import unicodedata
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Category(StrEnum):
    ABNORMAL = "Abnormal"
    NORMAL = "Normal"
    FAMILY_HISTORY = "Family History"


class Phenotype(StrictModel):
    phrase: str = Field(min_length=1)
    category: Category

    @field_validator("phrase")
    @classmethod
    def strip_phrase(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("phrase must not be blank")
        return value


class PhenotypeExtraction(StrictModel):
    phenotypes: list[Phenotype]


class PhenotypeSpan(StrictModel):
    model_config = ConfigDict(extra="ignore")
    phrase: str = Field(min_length=1)

    @field_validator("phrase")
    @classmethod
    def strip_phrase(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("phrase must not be blank")
        return value


class PhenotypeSpanExtraction(StrictModel):
    phenotypes: list[PhenotypeSpan]


class HPOMapping(StrictModel):
    hpo_id: str | None


class SpanPhenotype(Phenotype):
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class SpanPhenotypeExtraction(StrictModel):
    phenotypes: list[SpanPhenotype]


class MappingDecision(StrictModel):
    model_config = ConfigDict(extra="ignore")
    mention_id: str
    hpo_id: str | None = None
    verdict: Literal["supported", "unsupported", "ambiguous"] = "supported"
    confidence: Literal["high", "medium", "low"] = "medium"

    @model_validator(mode="before")
    @classmethod
    def fill_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "verdict" not in data:
                data["verdict"] = "supported" if data.get("hpo_id") else "unsupported"
            if "confidence" not in data:
                data["confidence"] = "medium"
        return data


def _normalize_batch_dict(data: Any) -> Any:
    if isinstance(data, dict):
        if "decisions" not in data:
            if "results" in data and isinstance(data["results"], list):
                data["decisions"] = data["results"]
            elif "decision" in data:
                dec = data["decision"]
                data["decisions"] = dec if isinstance(dec, list) else [dec]
            elif "items" in data and isinstance(data["items"], list):
                data["decisions"] = data["items"]
            else:
                items = []
                for k, v in data.items():
                    if isinstance(v, dict):
                        item = dict(v)
                        if "mention_id" not in item:
                            item["mention_id"] = k
                        items.append(item)
                if items:
                    data["decisions"] = items
    return data


class MappingDecisionBatch(StrictModel):
    model_config = ConfigDict(extra="ignore")
    decisions: list[MappingDecision]

    @model_validator(mode="before")
    @classmethod
    def normalize_batch(cls, data: Any) -> Any:
        return _normalize_batch_dict(data)


class MappingSetDecision(StrictModel):
    model_config = ConfigDict(extra="ignore")
    mention_id: str
    candidate_hpo_ids: list[str] = Field(default_factory=list, max_length=3)
    verdict: Literal["supported", "unsupported", "ambiguous"] = "supported"
    confidence: Literal["high", "medium", "low"] = "medium"

    @model_validator(mode="before")
    @classmethod
    def fill_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "verdict" not in data:
                c_ids = data.get("candidate_hpo_ids")
                data["verdict"] = "supported" if c_ids else "unsupported"
            if "confidence" not in data:
                data["confidence"] = "medium"
        return data

    @field_validator("candidate_hpo_ids")
    @classmethod
    def require_distinct_candidates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("candidate_hpo_ids must be distinct")
        return value

    @model_validator(mode="after")
    def validate_verdict_candidates(self) -> MappingSetDecision:
        if self.verdict == "unsupported" and self.candidate_hpo_ids:
            raise ValueError("unsupported decisions cannot retain candidate IDs")
        if self.verdict != "unsupported" and not self.candidate_hpo_ids:
            raise ValueError("supported or ambiguous decisions require candidate IDs")
        return self


class MappingSetDecisionBatch(StrictModel):
    model_config = ConfigDict(extra="ignore")
    decisions: list[MappingSetDecision]

    @model_validator(mode="before")
    @classmethod
    def normalize_batch(cls, data: Any) -> Any:
        return _normalize_batch_dict(data)


class FinalCategoryDecision(StrictModel):
    model_config = ConfigDict(extra="ignore")
    mention_id: str
    category: Category
    confidence: Literal["high", "medium", "low"] = "medium"


class FinalCategoryDecisionBatch(StrictModel):
    model_config = ConfigDict(extra="ignore")
    decisions: list[FinalCategoryDecision]

    @model_validator(mode="before")
    @classmethod
    def normalize_batch(cls, data: Any) -> Any:
        return _normalize_batch_dict(data)


class AnnotationInput(StrictModel):
    patient_id: str
    clinical_note: str

    @field_validator("patient_id", "clinical_note")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        value = unicodedata.normalize("NFC", value.replace("\r\n", "\n")).strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class AnnotationResult(StrictModel):
    patient_id: str
    phrase: str
    category: Category | None
    hpo_id: str | None = None
    hpo_term: str | None = None
    vector_score: float | None = None
    mapping_status: Literal[
        "mapped",
        "no_candidate_fit",
        "not_mapped_category",
        "error",
    ]
    error_code: str | None = None
    error_message: str | None = None
    evidence_start: int | None = None
    evidence_end: int | None = None
    assertion_status: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    category_confidence: Literal["high", "medium", "low"] | None = None
    review_status: Literal["accepted", "review", "rejected"] | None = None
    source_methods: list[str] | None = None
    candidate_hpo_ids: list[str] | None = None
    retrieval_candidate_hpo_ids: list[str] | None = None
    modifier_hpo_ids: list[str] | None = None
    evidence_segments: list[tuple[int, int]] | None = None
    mapping_verdict: Literal["supported", "unsupported", "ambiguous"] | None = None
    phenotype_confidence: float | None = Field(default=None, ge=0, le=1)
    mapping_set_confidence: float | None = Field(default=None, ge=0, le=1)
    overall_confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_basis: str | None = None
    evidence_text: str | None = None


class Candidate(StrictModel):
    hpo_id: str
    term: str
    score: float
    definition: str | None = None
    synonyms: list[str] | None = None
    parents: list[str] | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    context_dense_rank: int | None = None
    context_dense_score: float | None = None
    source_methods: list[str] | None = None


class DoctorCheck(StrictModel):
    name: str
    status: Literal["pass", "warn", "fail"]
    detail: str
    action: str | None = None


class DoctorReport(StrictModel):
    ok: bool
    checks: list[DoctorCheck]
