#!/usr/bin/env python3
"""Create a deterministic, validated prebuilt vector ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from rag_hpo.artifacts import (
    MANIFEST_NAME,
    META_NAME,
    VECTOR_NAME,
    load_artifacts,
    sha256_file,
)


def build(vector_dir: Path, output: Path) -> None:
    _, _, manifest = load_artifacts(vector_dir)
    canonical_manifest = manifest.model_copy(update={"created_at": "1970-01-01T00:00:00+00:00"})
    canonical_manifest_bytes = (canonical_manifest.model_dump_json(indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in (META_NAME, VECTOR_NAME, MANIFEST_NAME):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            content = (
                canonical_manifest_bytes
                if name == MANIFEST_NAME
                else (vector_dir / name).read_bytes()
            )
            archive.writestr(info, content, compresslevel=9)
    temporary.replace(output)
    metadata = {
        "schema_version": "1.0",
        "bundle": output.name,
        "bundle_sha256": sha256_file(output),
        "bundle_bytes": output.stat().st_size,
        "artifact_schema_version": manifest.schema_version,
        "artifact_manifest_sha256": hashlib.sha256(canonical_manifest_bytes).hexdigest(),
        "artifact_metadata_sha256": manifest.metadata_sha256,
        "artifact_vectors_sha256": manifest.vectors_sha256,
        "embedding_backend": manifest.embedding_backend,
        "embedding_model": manifest.embedding_model,
        "embedding_revision": manifest.embedding_revision,
        "vector_count": manifest.vector_count,
        "dimension": manifest.dimension,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vector_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.vector_dir, args.output)
    print(f"Created {args.output} (SHA-256 {sha256_file(args.output)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
