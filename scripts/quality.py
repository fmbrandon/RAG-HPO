#!/usr/bin/env python3
"""Canonical repository quality validation suite for local dev and CI."""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False, env=env)  # noqa: S603  # nosec B603
    return result.returncode


def run_license_check(python_exe: str) -> int:
    print("\n$ pip-licenses + check_licenses.py", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temporary:
        temporary_path = Path(temporary.name)

    try:
        cmd_licenses = [
            python_exe,
            "-m",
            "piplicenses",
            "--format=json",
            f"--output-file={temporary_path}",
        ]
        res1 = subprocess.run(cmd_licenses, cwd=ROOT, check=False)  # noqa: S603  # nosec B603
        if res1.returncode != 0:
            return res1.returncode

        cmd_check = [python_exe, "scripts/check_licenses.py", str(temporary_path)]
        res2 = subprocess.run(cmd_check, cwd=ROOT, check=False)  # noqa: S603  # nosec B603
        return res2.returncode
    finally:
        temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run canonical quality suite (Ruff, mypy, Bandit, secrets, licenses)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast code quality checks (Ruff, mypy, Bandit, secret scan).",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip network-dependent pip-audit step.",
    )
    parser.add_argument(
        "--no-notebook-exec",
        action="store_true",
        help="Skip executing Jupyter notebooks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    python_exe = sys.executable

    steps: list[tuple[str, list[str]]] = [
        ("ruff-check", [python_exe, "-m", "ruff", "check", "."]),
        ("ruff-format", [python_exe, "-m", "ruff", "format", "--check", "."]),
        (
            "nbqa-ruff",
            [
                python_exe,
                "-m",
                "nbqa",
                "ruff",
                "RAG-HPO.ipynb",
                "legacy_original_app/HPO_Vectorization.ipynb",
            ],
        ),
        ("mypy", [python_exe, "-m", "mypy", "src/rag_hpo"]),
        ("bandit", [python_exe, "-m", "bandit", "-c", "pyproject.toml", "-r", "src/rag_hpo"]),
        ("check-secrets", [python_exe, "scripts/check_secrets.py"]),
    ]

    if not args.fast:
        if not args.no_audit:
            steps.append(("pip-audit", [python_exe, "-m", "pip_audit"]))

        if not args.no_notebook_exec:
            steps.extend(
                [
                    (
                        "notebook-exec-main",
                        [
                            python_exe,
                            "-m",
                            "jupyter",
                            "nbconvert",
                            "--execute",
                            "--to",
                            "notebook",
                            "--stdout",
                            "RAG-HPO.ipynb",
                        ],
                    ),
                    (
                        "notebook-exec-legacy",
                        [
                            python_exe,
                            "-m",
                            "jupyter",
                            "nbconvert",
                            "--execute",
                            "--to",
                            "notebook",
                            "--stdout",
                            "legacy_original_app/HPO_Vectorization.ipynb",
                        ],
                    ),
                ]
            )

    for name, cmd in steps:
        ret = run_cmd(cmd)
        if ret != 0:
            print(f"\nQuality check failed at step '{name}' (exit code {ret}).", file=sys.stderr)
            return ret

    if not args.fast:
        ret = run_license_check(python_exe)
        if ret != 0:
            print("\nQuality check failed at step 'licenses'.", file=sys.stderr)
            return ret

    print("\nAll quality checks passed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
