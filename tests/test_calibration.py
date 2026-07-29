from __future__ import annotations

from pathlib import Path

import pytest

from rag_hpo.calibration import (
    STAGED_PIPELINE_SCHEMA_VERSION,
    CalibrationStratum,
    ConfidenceCalibration,
    apply_calibration,
    evidence_stratum,
)
from rag_hpo.models import AnnotationResult, Category


def _result() -> AnnotationResult:
    return AnnotationResult(
        patient_id="1",
        phrase="fever",
        category=Category.ABNORMAL,
        hpo_id="HP:0000001",
        hpo_term="Fever",
        mapping_status="mapped",
        assertion_status="affirmed",
        confidence="high",
        category_confidence="high",
        review_status="review",
        source_methods=["model-pass-1", "native"],
        candidate_hpo_ids=["HP:0000001"],
        mapping_verdict="supported",
    )


def test_held_out_calibration_controls_confidence_and_disposition() -> None:
    result = _result()
    key = evidence_stratum(result)
    calibration = ConfidenceCalibration(
        pipeline_schema_version=STAGED_PIPELINE_SCHEMA_VERSION,
        prompt_bundle_sha256="a",
        artifact_manifest_sha256="b",
        minimum_precision=0.70,
        accepted_strata=[key],
        strata={
            key: CalibrationStratum(
                observations=20,
                gold_matches=17,
                estimated_probability=18 / 22,
                lower_95=0.64,
            )
        },
        global_probability=0.5,
    )
    calibrated = apply_calibration([result], calibration)[0]
    assert calibrated.overall_confidence == 18 / 22
    assert calibrated.review_status == "accepted"
    assert calibrated.confidence_basis
    assert calibrated.phenotype_confidence is None
    assert calibrated.mapping_set_confidence is None


def test_unseen_evidence_stays_in_review_with_global_probability() -> None:
    result = _result()
    calibration = ConfidenceCalibration(
        pipeline_schema_version=STAGED_PIPELINE_SCHEMA_VERSION,
        prompt_bundle_sha256="a",
        artifact_manifest_sha256="b",
        minimum_precision=0.70,
        accepted_strata=[],
        strata={},
        global_probability=0.4,
    )
    calibrated = apply_calibration([result], calibration)[0]
    assert calibrated.overall_confidence == 0.4
    assert calibrated.review_status == "review"


def test_calibration_load_requires_matching_prompt_and_artifact(
    tmp_path: Path,
) -> None:
    calibration = ConfidenceCalibration(
        pipeline_schema_version=STAGED_PIPELINE_SCHEMA_VERSION,
        prompt_bundle_sha256="prompt",
        artifact_manifest_sha256="artifact",
        minimum_precision=0.70,
        accepted_strata=[],
        strata={},
        global_probability=0.4,
    )
    path = tmp_path / "calibration.json"
    path.write_text(calibration.model_dump_json(), encoding="utf-8")
    loaded = ConfidenceCalibration.load(
        path,
        prompt_bundle_sha256="prompt",
        artifact_manifest_sha256="artifact",
    )
    assert loaded.identity == calibration.identity
    with pytest.raises(ValueError, match="prompt identity"):
        ConfidenceCalibration.load(
            path,
            prompt_bundle_sha256="wrong",
            artifact_manifest_sha256="artifact",
        )
    with pytest.raises(ValueError, match="artifact identity"):
        ConfidenceCalibration.load(
            path,
            prompt_bundle_sha256="prompt",
            artifact_manifest_sha256="wrong",
        )
    incompatible = calibration.model_copy(update={"pipeline_schema_version": "old"})
    path.write_text(incompatible.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="pipeline identity"):
        ConfidenceCalibration.load(
            path,
            prompt_bundle_sha256="prompt",
            artifact_manifest_sha256="artifact",
        )


@pytest.mark.parametrize(
    ("methods", "lane"),
    [
        (["model-pass-1"], "model-pass-1"),
        (["model-pass-2"], "model-pass-2"),
        (["native"], "recognizer"),
        ([], "other"),
    ],
)
def test_evidence_strata_distinguish_pipeline_lanes(
    methods: list[str],
    lane: str,
) -> None:
    result = _result().model_copy(update={"source_methods": methods})
    assert evidence_stratum(result).startswith(f"{lane}|")


def test_calibration_leaves_nonmapped_results_unchanged() -> None:
    calibration = ConfidenceCalibration(
        pipeline_schema_version=STAGED_PIPELINE_SCHEMA_VERSION,
        prompt_bundle_sha256="a",
        artifact_manifest_sha256="b",
        minimum_precision=0.70,
        accepted_strata=[],
        strata={},
        global_probability=0.4,
    )
    result = _result().model_copy(update={"mapping_status": "no_candidate_fit"})
    assert apply_calibration([result], calibration) == [result]
    assert apply_calibration([result], None) == [result]
