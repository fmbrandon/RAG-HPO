#!/usr/bin/env python3
"""Repair repository CSC Case 68 from the immutable workbook-derived export."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_CASES = ROOT / "Test_Cases.csv"
CSC_INPUT = ROOT / "benchmarks" / "references" / "csc_input.csv"
CSC_REFERENCES = ROOT / "benchmarks" / "references" / "csc_manual_annotations.csv"


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def main() -> int:
    input_fields, authoritative_rows = _read_rows(CSC_INPUT)
    if input_fields != ["Case", "clinical_note"]:
        raise ValueError(f"unexpected CSC input columns: {input_fields}")
    authoritative = {row["Case"]: row["clinical_note"] for row in authoritative_rows}
    true_case_68 = authoritative.get("68", "").strip()
    if not true_case_68:
        raise ValueError("the workbook-derived CSC input has no nonblank Case 68")

    _, reference_rows = _read_rows(CSC_REFERENCES)
    case_68_ids = {
        row["hpo_term"].strip()
        for row in reference_rows
        if row["Patient ID"].strip() == "68" and row["hpo_term"].strip()
    }
    if not case_68_ids:
        raise ValueError("the workbook-derived manual annotations have no Case 68 IDs")

    fields, test_rows = _read_rows(TEST_CASES)
    by_case = {row["Case"]: row for row in test_rows}
    if "67" not in by_case or "68" not in by_case:
        raise ValueError("Test_Cases.csv must contain both Cases 67 and 68")

    current = by_case["68"]["clinical_note"]
    if current != true_case_68 and current != by_case["67"]["clinical_note"]:
        raise ValueError(
            "Case 68 is neither the known duplicated Case 67 note nor the "
            "authoritative workbook-derived note; refusing to overwrite it"
        )

    changed_note = current != true_case_68
    by_case["68"]["clinical_note"] = true_case_68
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(test_rows)
    canonical_bytes = b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")
    if TEST_CASES.read_bytes() == canonical_bytes:
        print(f"Case 68 is already synchronized ({len(case_68_ids)} manual HPO IDs).")
        return 0

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".Test_Cases.", suffix=".csv", dir=TEST_CASES.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes)
        os.replace(temporary_name, TEST_CASES)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    verb = "Synchronized" if changed_note else "Canonicalized"
    print(
        f"{verb} Test_Cases.csv Case 68 from CSC Input "
        f"({len(true_case_68)} characters; {len(case_68_ids)} manual HPO IDs)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
