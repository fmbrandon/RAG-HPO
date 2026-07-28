#!/usr/bin/env python3
"""Enforce branch-aware coverage targets for safety-critical modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGETS = (
    "src/rag_hpo/artifacts.py",
    "src/rag_hpo/config.py",
    "src/rag_hpo/models.py",
    "src/rag_hpo/state.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures: list[str] = []
    for target in TARGETS:
        summary = report["files"][target]["summary"]
        covered = float(summary["percent_covered"])
        if covered != 100.0:
            failures.append(f"{target}: {covered:.2f}%")
    if failures:
        parser.error("coverage targets below 100%: " + ", ".join(failures))
    print("Safety-critical module branch coverage: 100%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
