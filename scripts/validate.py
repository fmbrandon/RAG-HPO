#!/usr/bin/env python3
"""Canonical repository validation entrypoint reproducing local and CI gates."""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)  # noqa: S603  # nosec B603
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run canonical developer validation (quality, pytest, build, pip check)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast validation (fast quality suite + pytest without notebook re-execution).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip python package build step.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    python_exe = sys.executable

    # 1. Run Quality Suite
    quality_cmd = [python_exe, "scripts/quality.py"]
    if args.fast:
        quality_cmd.append("--fast")

    ret = run_cmd(quality_cmd)
    if ret != 0:
        print("\nValidation failed at quality checks step.", file=sys.stderr)
        return ret

    # 2. Run Test Suite & Coverage
    pytest_cmd = [
        python_exe,
        "-m",
        "pytest",
        "--cov=rag_hpo",
        "--cov-branch",
        "--cov-report=xml",
        "--cov-report=json:coverage.json",
    ]
    ret = run_cmd(pytest_cmd)
    if ret != 0:
        print("\nValidation failed at pytest step.", file=sys.stderr)
        return ret

    # 3. Check Coverage Targets
    cov_target_cmd = [python_exe, "scripts/check_coverage_targets.py", "coverage.json"]
    ret = run_cmd(cov_target_cmd)
    if ret != 0:
        print("\nValidation failed at coverage target check.", file=sys.stderr)
        return ret

    # 4. Build Package
    if not args.skip_build:
        build_cmd = [python_exe, "-m", "build"]
        ret = run_cmd(build_cmd)
        if ret != 0:
            print("\nValidation failed at python -m build step.", file=sys.stderr)
            return ret

    # 5. Pip Check
    pip_check_cmd = [python_exe, "-m", "pip", "check"]
    ret = run_cmd(pip_check_cmd)
    if ret != 0:
        print("\nValidation failed at pip check step.", file=sys.stderr)
        return ret

    print("\nAll validation checks passed cleanly. Local workspace reproduces CI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
