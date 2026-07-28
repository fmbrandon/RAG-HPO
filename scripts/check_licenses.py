#!/usr/bin/env python3
"""Fail on unreviewed GPL/LGPL dependency licenses."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ALLOWLIST = {"chardet"}
COPYLEFT = re.compile(r"\b(?:A?GPL|LGPL)\b", re.IGNORECASE)


def violations(inventory: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        item
        for item in inventory
        if item.get("Name", "").casefold() not in ALLOWLIST
        and COPYLEFT.search(item.get("License", ""))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    args = parser.parse_args()
    data = json.loads(args.inventory.read_text(encoding="utf-8"))
    failures = violations(data)
    if failures:
        for item in failures:
            print(f"Unreviewed copyleft dependency: {item['Name']} ({item['License']})")
        return 1
    print("Dependency license policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
