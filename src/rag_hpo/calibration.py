from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_hpo.models import AnnotationResult, Category

CALIBRATION_SCHEMA_VERSION = "1.0"
STAGED_PIPELINE_SCHEMA_VERSION = "3.0"


class CalibrationStratum(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: int = Field(ge=1)
    gold_matches: int = Field(ge=0)
    estimated_probability: float = Field(ge=0, le=1)
    lower_95: float = Field(ge=0, le=1)


class ConfidenceCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = CALIBRATION_SCHEMA_VERSION
    pipeline_schema_version: str
    prompt_bundle_sha256: str
    artifact_manifest_sha256: str
    minimum_precision: float = Field(ge=0, le=1)
    accepted_strata: list[str]
    strata: dict[str, CalibrationStratum]
    global_probability: float = Field(ge=0, le=1)
    selection: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    limits: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        prompt_bundle_sha256: str,
        artifact_manifest_sha256: str,
    ) -> ConfidenceCalibration:
        value = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if value.pipeline_schema_version != STAGED_PIPELINE_SCHEMA_VERSION:
            raise ValueError("confidence calibration pipeline identity does not match")
        if value.prompt_bundle_sha256 != prompt_bundle_sha256:
            raise ValueError("confidence calibration prompt identity does not match")
        if value.artifact_manifest_sha256 != artifact_manifest_sha256:
            raise ValueError("confidence calibration artifact identity does not match")
        return value

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


def evidence_stratum(result: AnnotationResult) -> str:
    methods = set(result.source_methods or [])
    has_model = any(value.startswith("model-pass") for value in methods)
    has_recognizer = bool({"native", "fasthpocr"} & methods)
    if has_model and has_recognizer:
        lane = "model+recognizer"
    elif "model-pass-1" in methods:
        lane = "model-pass-1"
    elif "model-pass-2" in methods:
        lane = "model-pass-2"
    elif has_recognizer:
        lane = "recognizer"
    else:
        lane = "other"
    return "|".join(
        (
            lane,
            result.mapping_verdict or "none",
            result.confidence or "none",
            result.category_confidence or "none",
            result.assertion_status or "none",
            f"set-{len(result.candidate_hpo_ids or [])}",
        )
    )


def apply_calibration(
    results: list[AnnotationResult],
    calibration: ConfidenceCalibration | None,
) -> list[AnnotationResult]:
    if calibration is None:
        return results
    accepted = set(calibration.accepted_strata)
    output: list[AnnotationResult] = []
    for result in results:
        if (
            result.category is not Category.ABNORMAL
            or result.mapping_status != "mapped"
            or not result.candidate_hpo_ids
        ):
            output.append(result)
            continue
        key = evidence_stratum(result)
        stratum = calibration.strata.get(key)
        probability = (
            stratum.estimated_probability if stratum is not None else calibration.global_probability
        )
        output.append(
            result.model_copy(
                update={
                    "overall_confidence": probability,
                    "confidence_basis": f"held-out-calibration:{calibration.identity}",
                    "review_status": "accepted" if key in accepted else "review",
                }
            )
        )
    return output
