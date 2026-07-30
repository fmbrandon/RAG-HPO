#!/usr/bin/env python3
"""Create the environment, extract HF models, and build HPO vector DB."""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

MINIMUM = (3, 11)
MAXIMUM = (3, 14)
ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / ".venv"
DEFAULT_VECTOR_DIR = ROOT / "artifacts" / "hpo"
DEFAULT_ADDONS = ROOT / "HPO_addons.csv"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603  # nosec B603


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap RAG-HPO environment, preload HF models, and build vector DB."
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=DEFAULT_VECTOR_DIR,
        help=f"Target directory for HPO vector artifacts (default: {DEFAULT_VECTOR_DIR})",
    )
    parser.add_argument(
        "--backend",
        choices=("sapbert", "fastembed"),
        default="sapbert",
        help="Embedding model backend to extract and use (default: sapbert)",
    )
    parser.add_argument(
        "--skip-vectorize",
        action="store_true",
        help="Skip downloading HF embedding models and creating initial HPO vector DB.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (MINIMUM <= sys.version_info[:2] < MAXIMUM):
        print("RAG-HPO requires Python >=3.11,<3.14; Python 3.12 is recommended.", file=sys.stderr)
        return 2

    if not ENVIRONMENT.exists():
        print(f"Creating virtual environment in {ENVIRONMENT}...")
        run([sys.executable, "-m", "venv", str(ENVIRONMENT)])

    python = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    print("Upgrading core build tooling (pip, setuptools, wheel)...")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    print("Installing RAG-HPO package with vectorize dependencies...")
    run([str(python), "-m", "pip", "install", "-e", ".[vectorize,notebook]"])

    print("Verifying environment dependency integrity...")
    run([str(python), "-m", "pip", "check"])

    if not args.skip_vectorize:
        print(f"\n1. Extracting/preloading '{args.backend}' embedding model from Hugging Face...")
        preload_code = (
            "from rag_hpo.embeddings import create_backend\n"
            f"backend = create_backend('{args.backend}')\n"
            "print(f'Hugging Face model loaded: {backend.model_id} ({backend.revision})')\n"
        )
        run([str(python), "-c", preload_code])

        print(f"\n2. Executing initial run to build HPO vector database in {args.vector_dir}...")
        vectorize_cmd = [
            str(python),
            "-m",
            "rag_hpo.cli",
            "vectorize",
            "--output-dir",
            str(args.vector_dir),
            "--backend",
            args.backend,
        ]
        if DEFAULT_ADDONS.is_file():
            vectorize_cmd.extend(["--hpo-addons", str(DEFAULT_ADDONS)])
        run(vectorize_cmd)

        # Cache copy to default cache root so `rag-hpo demo` runs seamlessly without extra flags
        cache_script = (
            "import shutil, sys\n"
            "from pathlib import Path\n"
            "from rag_hpo.bundle import default_cache_root\n"
            "src = Path(sys.argv[1])\n"
            "cache = default_cache_root()\n"
            "cache.mkdir(parents=True, exist_ok=True)\n"
            "for item in src.iterdir():\n"
            "    if item.is_file():\n"
            "        shutil.copy2(item, cache / item.name)\n"
            "print(f'Populated default cache bundle at {cache}')\n"
        )
        run([str(python), "-c", cache_script, str(args.vector_dir)])

    activate = r".\.venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    print(f"\nEnvironment setup complete! Activate with: {activate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
