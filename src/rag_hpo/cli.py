from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from rag_hpo.config import DEFAULT_BASE_URL, DEFAULT_MODEL, ProviderConfig, ResponseMode
from rag_hpo.demo import run_demo
from rag_hpo.doctor import run_doctor
from rag_hpo.models import AnnotationInput, AnnotationResult
from rag_hpo.ontology import DEFAULT_HPO_URL, vectorize
from rag_hpo.pipeline import AnnotationPipeline, load_inputs
from rag_hpo.provider import OpenAICompatibleProvider
from rag_hpo.staged_pipeline import (
    AnnotationMode,
    MappingPromptMode,
    StagedAnnotationPipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-hpo",
        description="Retrieval-augmented Human Phenotype Ontology annotation.",
    )
    parser.add_argument("--version", action="version", version="rag-hpo 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Diagnose configuration and artifacts.")
    doctor.add_argument("--vector-dir", type=Path)
    doctor.add_argument("--output-dir", type=Path, default=Path("rag_hpo_output"))
    doctor.add_argument("--base-url")
    doctor.add_argument("--model")
    doctor.add_argument("--skip-provider", action="store_true")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    demo = commands.add_parser("demo", help="Run the beginner-safe synthetic demonstration.")
    demo.add_argument("--vector-dir", type=Path)
    demo.add_argument("--output-dir", type=Path, default=Path("rag_hpo_demo_output"))
    demo.add_argument("--json", action="store_true", dest="as_json")

    vector = commands.add_parser("vectorize", help="Build reproducible HPO vectors.")
    vector.add_argument("--obo-file", type=Path)
    vector.add_argument("--obo-url", default=DEFAULT_HPO_URL)
    vector.add_argument("--hpo-addons", type=Path)
    vector.add_argument("--output-dir", type=Path, required=True)
    vector.add_argument("--backend", choices=("sapbert", "fastembed"), default="sapbert")
    vector.add_argument("--limit", type=_positive_int)
    vector.add_argument("--refresh", action="store_true")
    vector.add_argument("--offline", action="store_true")
    vector.add_argument("--json", action="store_true", dest="as_json")

    annotate = commands.add_parser("annotate", help="Extract and map phenotypes.")
    source = annotate.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", help="CSV path, or '-' for clinical-note text on stdin.")
    annotate.add_argument("--patient-id", default="1")
    annotate.add_argument("--vector-dir", type=Path, required=True)
    annotate.add_argument("--output-dir", type=Path, required=True)
    annotate.add_argument(
        "--base-url",
        default=os.environ.get("RAG_HPO_BASE_URL", DEFAULT_BASE_URL),
    )
    annotate.add_argument("--model", default=os.environ.get("RAG_HPO_MODEL", DEFAULT_MODEL))
    annotate.add_argument(
        "--response-mode",
        choices=[mode.value for mode in ResponseMode],
        default=os.environ.get("RAG_HPO_RESPONSE_MODE", ResponseMode.STRICT.value),
    )
    annotate.add_argument("--resume", action="store_true")
    annotate.add_argument("--keep-state", action="store_true")
    annotate.add_argument(
        "--max-row-attempts",
        type=_positive_int,
        default=1,
        help=(
            "Retry failed rows inside the loaded process so models, indexes, and "
            "successful stage responses are reused (default: 1)."
        ),
    )
    annotate.add_argument("--raw-responses", action="store_true")
    annotate.add_argument(
        "--mode",
        choices=[mode.value for mode in AnnotationMode],
        default=AnnotationMode.MODEL.value,
    )
    annotate.add_argument(
        "--recognizer",
        action="append",
        choices=("native", "fasthpocr"),
        default=[],
    )
    annotate.add_argument("--fasthpocr-index", type=Path)
    annotate.add_argument("--offline", action="store_true")
    annotate.add_argument("--no-model", action="store_true")
    annotate.add_argument("--include-evidence-text", action="store_true")
    annotate.add_argument(
        "--confidence-calibration",
        type=Path,
        help="Apply a hash-matched held-out confidence calibration policy.",
    )
    annotate.add_argument(
        "--mapping-prompt",
        choices=[mode.value for mode in MappingPromptMode],
        default=MappingPromptMode.ZERO_SHOT.value,
        help="Use a context-aware zero-shot prompt or one fixed synthetic example.",
    )
    annotate.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _doctor(args: argparse.Namespace) -> int:
    report = run_doctor(
        vector_dir=args.vector_dir,
        output_dir=args.output_dir,
        check_provider=not args.skip_provider,
        base_url=args.base_url,
        model=args.model,
    )
    if args.as_json:
        print(report.model_dump_json(indent=2))
    else:
        for check in report.checks:
            print(f"{check.status.upper():4} {check.name}: {check.detail}")
            if check.action and check.status != "pass":
                print(f"     Fix: {check.action}")
    return 0 if report.ok else 1


def _vectorize(args: argparse.Namespace) -> int:
    manifest = vectorize(
        output_dir=args.output_dir,
        obo_file=args.obo_file,
        obo_url=args.obo_url,
        addons_path=args.hpo_addons,
        backend_name=args.backend,
        limit=args.limit,
        refresh=args.refresh,
        offline=args.offline,
    )
    if args.as_json:
        print(manifest.model_dump_json(indent=2))
    else:
        print(
            f"Created {manifest.vector_count} vectors "
            f"({manifest.dimension} dimensions) in {args.output_dir}"
        )
    return 0


def _load_annotation_source(
    args: argparse.Namespace,
) -> tuple[list[AnnotationInput], list[AnnotationResult]]:
    if args.text is not None:
        return [AnnotationInput(patient_id=args.patient_id, clinical_note=args.text)], []
    if args.input == "-":
        return [
            AnnotationInput(
                patient_id=args.patient_id,
                clinical_note=sys.stdin.read(),
            )
        ], []
    return load_inputs(Path(args.input))


def _annotate(args: argparse.Namespace) -> int:
    if args.raw_responses:
        print(
            "WARNING: raw provider responses may contain sensitive clinical text.",
            file=sys.stderr,
        )
    if args.include_evidence_text:
        print(
            "WARNING: evidence text may contain sensitive clinical information.",
            file=sys.stderr,
        )
    mode = AnnotationMode(args.mode)
    if mode is AnnotationMode.MODEL and args.no_model:
        raise ValueError("--no-model cannot be used with --mode model")
    uses_provider = mode is AnnotationMode.MODEL or (
        mode in {AnnotationMode.BALANCED, AnnotationMode.HIGH_RECALL} and not args.no_model
    )
    if args.offline and uses_provider:
        host = urlparse(args.base_url).hostname
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("--offline model modes require a loopback --base-url")
    rows, errors = _load_annotation_source(args)
    if not rows and errors:
        from rag_hpo.export import export_results

        export_results(errors, args.output_dir)
        return 4
    provider_context: Any
    if uses_provider:
        config = ProviderConfig.from_env(
            base_url=args.base_url,
            model=args.model,
            response_mode=ResponseMode(args.response_mode),
        )
        provider_context = OpenAICompatibleProvider(config)
    else:
        provider_context = contextlib.nullcontext(None)
    with provider_context as provider:
        if mode is AnnotationMode.MODEL:
            if provider is None:
                raise RuntimeError("model mode requires a provider")
            pipeline: AnnotationPipeline | StagedAnnotationPipeline = AnnotationPipeline(
                provider=provider,
                vector_dir=args.vector_dir,
                output_dir=args.output_dir,
                resume=args.resume,
                keep_state=args.keep_state,
                keep_raw_responses=args.raw_responses,
                offline=args.offline,
            )
        else:
            recognizers = {"native", *args.recognizer}
            if mode is AnnotationMode.FASTHPOCR:
                recognizers.add("fasthpocr")
            pipeline = StagedAnnotationPipeline(
                provider=provider,
                vector_dir=args.vector_dir,
                output_dir=args.output_dir,
                mode=mode,
                recognizers=recognizers,
                fasthpocr_index=args.fasthpocr_index,
                resume=args.resume,
                keep_state=args.keep_state,
                keep_raw_responses=args.raw_responses,
                include_evidence_text=args.include_evidence_text,
                offline=args.offline,
                confidence_calibration=args.confidence_calibration,
                mapping_prompt=MappingPromptMode(args.mapping_prompt),
            )
        if isinstance(pipeline, StagedAnnotationPipeline):
            results = pipeline.run(
                rows,
                initial_errors=errors,
                max_attempts=args.max_row_attempts,
            )
        else:
            if args.max_row_attempts != 1:
                raise ValueError("--max-row-attempts requires a staged annotation mode")
            results = pipeline.run(rows, initial_errors=errors)
    failures = sum(result.mapping_status == "error" for result in results)
    summary = {
        "input_rows": len(rows) + len(errors),
        "result_rows": len(results),
        "error_rows": failures,
        "accepted_rows": sum(
            result.review_status in (None, "accepted") and result.mapping_status == "mapped"
            for result in results
        ),
        "review_rows": sum(result.review_status == "review" for result in results),
        "rejected_rows": sum(result.review_status == "rejected" for result in results),
        "output_csv": str(args.output_dir / "rag_hpo_results.csv"),
        "output_json": str(args.output_dir / "rag_hpo_results.json"),
    }
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Wrote {len(results)} results with {failures} errors to {args.output_dir}")
    return 0 if failures == 0 else 4


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "demo":
            return run_demo(
                vector_dir=args.vector_dir,
                output_dir=args.output_dir,
                as_json=args.as_json,
            )
        if args.command == "vectorize":
            return _vectorize(args)
        if args.command == "annotate":
            return _annotate(args)
    except (FileNotFoundError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
