#!/usr/bin/env python3
"""Run resumable annotation and exact scoring for one explicit benchmark cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess  # nosec B404 - fixed local Python entry points, never a shell
import sys
from pathlib import Path
from urllib.parse import urlparse

from rag_hpo.config import DEFAULT_BASE_URL, DEFAULT_MODEL
from rag_hpo.privacy import ensure_private_directory, restrict_owner

REPOSITORY = Path(__file__).resolve().parents[1]
REFERENCES = REPOSITORY / "benchmarks" / "references"
CORPORA = {
    "csc": (
        REFERENCES / "csc_input.csv",
        REFERENCES / "csc_manual_annotations.csv",
    ),
    "gsc": (
        REFERENCES / "gsc_input.csv",
        REFERENCES / "gsc_manual_annotations.csv",
    ),
}


def _selection_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selection = raw.get("selection", raw)
    for key in ("selected_case_ids", "confirmation_case_ids", "case_ids"):
        values = selection.get(key)
        if isinstance(values, list):
            return [str(value) for value in values]
    raise ValueError("selection manifest does not contain a recognized case-ID list")


def _write_selected_input(
    source: Path,
    destination: Path,
    case_ids: list[str] | None,
) -> list[str]:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {
        str(row.get("patient_id") or row.get("Case")): {
            "patient_id": str(row.get("patient_id") or row.get("Case")),
            "clinical_note": row["clinical_note"],
        }
        for row in rows
    }
    selected = list(by_id) if case_ids is None else case_ids
    missing = [case_id for case_id in selected if case_id not in by_id]
    if missing:
        raise ValueError(f"input corpus is missing selected case IDs: {missing}")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "clinical_note"])
        writer.writeheader()
        writer.writerows(by_id[case_id] for case_id in selected)
    restrict_owner(destination)
    return selected


def _run(command: list[str], *, allowed_returncodes: set[int] | None = None) -> int:
    completed = subprocess.run(command, cwd=REPOSITORY, check=False)  # noqa: S603
    allowed = allowed_returncodes or {0}
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same resumable balanced pipeline and alternative-aware exact "
            "scoring for a subset or a complete CSC/GSC corpus."
        )
    )
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--case-id", action="append", dest="case_ids")
    selection.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("RAG_HPO_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("RAG_HPO_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--confirm-external-transmission",
        action="store_true",
        help="Confirm that the selected published notes may be sent to a remote provider.",
    )
    parser.add_argument(
        "--high-recall",
        action="store_true",
        help="Use 32 distinct candidates instead of the balanced 16-candidate policy.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Retry only unfinished/error rows this many times (default: 3).",
    )
    args = parser.parse_args()
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")

    hostname = urlparse(args.base_url).hostname
    remote = hostname not in {"localhost", "127.0.0.1", "::1"}
    if remote and not args.confirm_external_transmission:
        parser.error("remote evaluation requires --confirm-external-transmission")
    if remote and not os.environ.get("RAG_HPO_API_KEY"):
        parser.error("RAG_HPO_API_KEY is not set")

    ensure_private_directory(args.output_dir)
    input_source, reference_source = CORPORA[args.corpus]
    if args.selection_manifest is not None:
        case_ids = _selection_ids(args.selection_manifest)
    elif args.case_ids:
        case_ids = [str(value) for value in args.case_ids]
    else:
        case_ids = None

    selected_input = args.output_dir / f"{args.corpus}-selected-input.csv"
    selected_ids = _write_selected_input(input_source, selected_input, case_ids)
    generated_selection = args.output_dir / f"{args.corpus}-selection.json"
    generated_selection.write_text(
        json.dumps(
            {
                "selection": {
                    "corpus": args.corpus.upper(),
                    "complete_corpus": args.all,
                    "selected_case_ids": selected_ids,
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    restrict_owner(generated_selection)

    annotation_dir = args.output_dir / "annotations"
    score_dir = args.output_dir / "score"
    annotate_command = [
        sys.executable,
        "-m",
        "rag_hpo.cli",
        "annotate",
        "--input",
        str(selected_input),
        "--vector-dir",
        str(args.vector_dir),
        "--output-dir",
        str(annotation_dir),
        "--mode",
        "high-recall" if args.high_recall else "balanced",
        "--recognizer",
        "native",
        "--mapping-prompt",
        "zero-shot",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--resume",
        "--keep-state",
        "--json",
    ]
    annotation_returncode = 4
    for attempt in range(1, args.max_attempts + 1):
        annotation_returncode = _run(annotate_command, allowed_returncodes={0, 4})
        if annotation_returncode == 0:
            break
        print(
            f"Annotation attempt {attempt} retained row errors; "
            "retrying only unfinished rows from the private checkpoint.",
            file=sys.stderr,
        )
    if annotation_returncode != 0:
        raise RuntimeError(
            "annotation still has row errors after the configured attempts; "
            "rerun the identical command to resume"
        )
    _run(
        [
            sys.executable,
            str(REPOSITORY / "benchmarks" / "run_benchmark.py"),
            "--predictions",
            str(annotation_dir / "rag_hpo_results.json"),
            "--input",
            str(selected_input),
            "--references",
            str(reference_source),
            "--ontology",
            str(args.ontology),
            "--prompt-file",
            str(REPOSITORY / "src" / "rag_hpo" / "data" / "system_prompts.json"),
            "--vector-manifest",
            str(args.vector_dir / "hpo_manifest.json"),
            "--selection-manifest",
            str(generated_selection),
            "--accepted-only",
            "--model",
            args.model,
            "--prompt-version",
            "staged-pipeline-3.2",
            "--output-dir",
            str(score_dir),
        ]
    )
    layered_report = score_dir / "layered.json"
    _run(
        [
            sys.executable,
            str(REPOSITORY / "benchmarks" / "score_layered.py"),
            "--predictions",
            str(annotation_dir / "rag_hpo_results.json"),
            "--references",
            str(reference_source),
            "--ontology",
            str(args.ontology),
            "--selection-manifest",
            str(generated_selection),
            "--accepted-only",
            "--output",
            str(layered_report),
        ]
    )
    print(
        json.dumps(
            {
                "annotation_results": str(annotation_dir / "rag_hpo_results.json"),
                "cases": len(selected_ids),
                "corpus": args.corpus,
                "resume_state": str(annotation_dir / ".rag-hpo-state.sqlite3"),
                "score_report": str(score_dir / "benchmark_report.json"),
                "hierarchy_sensitivity_report": str(layered_report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
