#!/usr/bin/env python3
"""Build the deterministic HPO registry without creating embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag_hpo.artifacts import sha256_file
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.registry import (
    LEXICAL_MANIFEST_NAME,
    LEXICAL_NAME,
    REGISTRY_MANIFEST_NAME,
    REGISTRY_NAME,
    build_registry,
    write_registry_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--addons", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    ensure_private_directory(args.output_dir)
    started = time.perf_counter()
    registry = build_registry(
        args.ontology,
        addons_path=args.addons,
        limit=args.limit,
    )
    registry_manifest, lexical_manifest = write_registry_bundle(
        args.output_dir,
        registry=registry,
        hpo_source=str(args.ontology.resolve()),
        hpo_sha256=sha256_file(args.ontology),
        addons_sha256=sha256_file(args.addons) if args.addons else None,
    )
    summary = {
        "schema_version": "1.0",
        "elapsed_seconds": time.perf_counter() - started,
        "registry_manifest": registry_manifest.model_dump(mode="json"),
        "lexical_manifest": lexical_manifest.model_dump(mode="json"),
        "artifact_sha256": {
            REGISTRY_NAME: sha256_file(args.output_dir / REGISTRY_NAME),
            REGISTRY_MANIFEST_NAME: sha256_file(args.output_dir / REGISTRY_MANIFEST_NAME),
            LEXICAL_NAME: sha256_file(args.output_dir / LEXICAL_NAME),
            LEXICAL_MANIFEST_NAME: sha256_file(args.output_dir / LEXICAL_MANIFEST_NAME),
        },
    }
    summary_path = args.output_dir / "registry_build.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
