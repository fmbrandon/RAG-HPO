from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Category(StrEnum):
    ABNORMAL = "Abnormal"
    NORMAL = "Normal"
    FAMILY_HISTORY = "Family History"
    OTHER = "Other"
    SUSPECTED = "Suspected"


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


class HPOMapping(StrictModel):
    hpo_id: str | None


class AnnotationInput(StrictModel):
    patient_id: str
    clinical_note: str

    @field_validator("patient_id", "clinical_note")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class AnnotationResult(StrictModel):
    patient_id: str
    phrase: str
    category: Category
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


class Candidate(StrictModel):
    hpo_id: str
    term: str
    score: float


class DoctorCheck(StrictModel):
    name: str
    status: Literal["pass", "warn", "fail"]
    detail: str
    action: str | None = None


class DoctorReport(StrictModel):
    ok: bool
    checks: list[DoctorCheck]
