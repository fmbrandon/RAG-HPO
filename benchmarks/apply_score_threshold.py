#!/usr/bin/env python3
"""Apply a frozen discovery threshold to prediction rows for sensitivity scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between zero and one")

    rows: list[dict[str, Any]] = json.loads(args.predictions.read_text(encoding="utf-8"))
    removed = 0
    output = []
    for row in rows:
        score = row.get("vector_score")
        if (
            row.get("mapping_status") == "mapped"
            and isinstance(score, int | float)
            and float(score) < args.threshold
        ):
            row = {
                **row,
                "hpo_id": None,
                "hpo_term": None,
                "mapping_status": "no_candidate_fit",
            }
            removed += 1
        output.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "threshold": args.threshold,
                "input_rows": len(rows),
                "removed_mapped_rows": removed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
