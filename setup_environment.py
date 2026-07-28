#!/usr/bin/env python3
"""Create the project environment without modifying the system interpreter."""

from __future__ import annotations

import os
import subprocess  # nosec B404
import sys
from pathlib import Path

MINIMUM = (3, 11)
MAXIMUM = (3, 14)
ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / ".venv"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603  # nosec B603


def main() -> int:
    if not (MINIMUM <= sys.version_info[:2] < MAXIMUM):
        print("RAG-HPO requires Python >=3.11,<3.14; Python 3.12 is recommended.", file=sys.stderr)
        return 2

    if not ENVIRONMENT.exists():
        run([sys.executable, "-m", "venv", str(ENVIRONMENT)])

    python = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(python), "-m", "pip", "install", "-e", ".[vectorize,notebook]"])
    run([str(python), "-m", "pip", "check"])

    activate = r".\.venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    print(f"Environment ready. Activate with: {activate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
