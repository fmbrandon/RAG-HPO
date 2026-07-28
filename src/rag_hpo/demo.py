from __future__ import annotations

import json
from pathlib import Path

from rag_hpo.bundle import BUNDLE_DOWNLOAD_BYTES, resolve_vector_dir
from rag_hpo.config import ProviderConfig
from rag_hpo.doctor import run_doctor
from rag_hpo.models import AnnotationInput
from rag_hpo.pipeline import AnnotationPipeline
from rag_hpo.provider import OpenAICompatibleProvider

DEMO_NOTE = (
    "A 34-year-old adult has recurrent migraine headaches and excessive thirst. "
    "Examination documents obesity. Hearing is normal."
)


def run_demo(
    *,
    vector_dir: Path | None,
    output_dir: Path,
    as_json: bool,
) -> int:
    if vector_dir is None and not as_json:
        mebibytes = BUNDLE_DOWNLOAD_BYTES / (1024 * 1024)
        print(
            f"First run: the validated vector bundle is about {mebibytes:.0f} MiB; "
            "allow at least 500 MB free and several minutes."
        )
    try:
        resolved_vectors = resolve_vector_dir(vector_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        failure = {
            "ok": False,
            "error": "vectors_unavailable",
            "detail": str(exc),
        }
        if as_json:
            print(json.dumps(failure, indent=2))
        else:
            print(f"Demo cannot start: {exc}")
        return 2
    report = run_doctor(
        vector_dir=resolved_vectors,
        output_dir=output_dir,
        check_provider=True,
    )
    if not report.ok:
        if as_json:
            print(report.model_dump_json(indent=2))
        else:
            print("The demo cannot start until these checks are fixed:")
            for check in report.checks:
                if check.status == "fail":
                    print(f"- {check.detail}")
                    if check.action:
                        print(f"  Fix: {check.action}")
        return 1

    config = ProviderConfig.from_env()
    with OpenAICompatibleProvider(config) as provider:
        pipeline = AnnotationPipeline(
            provider=provider,
            vector_dir=resolved_vectors,
            output_dir=output_dir,
            resume=False,
            keep_state=False,
            keep_raw_responses=False,
        )
        results = pipeline.run(
            [AnnotationInput(patient_id="synthetic-demo-1", clinical_note=DEMO_NOTE)]
        )
    errors = sum(result.mapping_status == "error" for result in results)
    mapped = [result.hpo_id for result in results if result.hpo_id]
    summary = {
        "ok": errors == 0,
        "synthetic_input": True,
        "result_rows": len(results),
        "mapped_hpo_ids": mapped,
        "output_csv": str(output_dir / "rag_hpo_results.csv"),
        "output_json": str(output_dir / "rag_hpo_results.json"),
    }
    if as_json:
        print(json.dumps(summary, indent=2))
    else:
        print("Demo complete using a built-in synthetic note.")
        print(f"Mapped HPO IDs: {', '.join(mapped) if mapped else 'none selected'}")
        print(f"CSV results: {summary['output_csv']}")
        print(f"JSON results: {summary['output_json']}")
        print("Each row is one extracted phrase; only Abnormal rows are mapped.")
    return 0 if errors == 0 else 4
