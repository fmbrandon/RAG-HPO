#!/usr/bin/env python3
"""Canonical repository secret scanner wrapper around detect-secrets."""

from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

EXCLUDE_PATTERN = (
    r"(^|/)requirements(/|\.txt$)|"
    r"(^|/)benchmarks/results/.*\.json$|"
    r"(^|/)benchmarks/provenance/.*\.json$|"
    r"(^|/)benchmarks/prompts/historical-2025-07\.json$|"
    r"(^|/)artifacts/.*|"
    r"\.(pptx|xlsx)$"
)


def run_secret_scan() -> int:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temporary:
        temporary_path = Path(temporary.name)

    try:
        command = [
            sys.executable,
            "-m",
            "detect_secrets",
            "scan",
            "--exclude-files",
            EXCLUDE_PATTERN,
        ]
        result = subprocess.run(  # noqa: S603  # nosec B603
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"detect-secrets invocation failed: {result.stderr}", file=sys.stderr)
            return result.returncode

        data = json.loads(result.stdout)
        results = data.get("results", {})

        violations = 0
        for filename, findings in sorted(results.items()):
            for finding in findings:
                violations += 1
                line_no = finding.get("line_number")
                secret_type = finding.get("type")
                print(f"Secret detected: {filename}:{line_no} [{secret_type}]")

        if violations > 0:
            print(f"\nFailed: {violations} potential secret(s) found.", file=sys.stderr)
            return 1

        print("Secret scan passed cleanly.")
        return 0
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(run_secret_scan())
