from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rag_hpo.privacy import ensure_private_directory

SCHEMA_VERSION = "2.0"
META_NAME = "hpo_meta.json"
VECTOR_NAME = "hpo_embedded.npz"
MANIFEST_NAME = "hpo_manifest.json"


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hp_id: str
    phrase: str
    term: str
    definition: str = ""
    source: str


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    created_at: str
    hpo_source: str
    hpo_sha256: str
    addons_sha256: str | None
    parser: str
    parser_version: str
    embedding_backend: str
    embedding_model: str
    embedding_revision: str
    normalized: bool = True
    dtype: str
    dimension: int = Field(gt=0)
    metadata_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    metadata_sha256: str
    vectors_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes, mode: int = 0o600) -> None:
    ensure_private_directory(path.parent)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_artifacts(
    output_dir: Path,
    *,
    entries: list[ArtifactEntry],
    vectors: np.ndarray,
    hpo_source: str,
    hpo_sha256: str,
    addons_sha256: str | None,
    parser: str,
    parser_version: str,
    embedding_backend: str,
    embedding_model: str,
    embedding_revision: str,
) -> ArtifactManifest:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if len(entries) != matrix.shape[0]:
        raise ValueError("metadata and vector counts do not match")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain non-finite values")

    ensure_private_directory(output_dir)
    metadata_path = output_dir / META_NAME
    vectors_path = output_dir / VECTOR_NAME
    manifest_path = output_dir / MANIFEST_NAME

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }
    _atomic_write_bytes(
        metadata_path,
        (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    array_buffer = io.BytesIO()
    np.save(array_buffer, matrix, allow_pickle=False)
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        member = zipfile.ZipInfo("emb.npy", date_time=(1980, 1, 1, 0, 0, 0))
        member.compress_type = zipfile.ZIP_DEFLATED
        member.external_attr = 0o600 << 16
        archive.writestr(member, array_buffer.getvalue(), compresslevel=9)
    _atomic_write_bytes(vectors_path, archive_buffer.getvalue())

    manifest = ArtifactManifest(
        created_at=datetime.now(UTC).isoformat(),
        hpo_source=hpo_source,
        hpo_sha256=hpo_sha256,
        addons_sha256=addons_sha256,
        parser=parser,
        parser_version=parser_version,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_revision=embedding_revision,
        dtype=str(matrix.dtype),
        dimension=matrix.shape[1],
        metadata_count=len(entries),
        vector_count=matrix.shape[0],
        metadata_sha256=sha256_file(metadata_path),
        vectors_sha256=sha256_file(vectors_path),
    )
    _atomic_write_bytes(
        manifest_path,
        (manifest.model_dump_json(indent=2) + "\n").encode("utf-8"),
    )
    return manifest


def load_artifacts(vector_dir: Path) -> tuple[list[ArtifactEntry], np.ndarray, ArtifactManifest]:
    metadata_path = vector_dir / META_NAME
    vectors_path = vector_dir / VECTOR_NAME
    manifest_path = vector_dir / MANIFEST_NAME
    for path in (metadata_path, vectors_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required vector artifact is missing: {path}")

    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported artifact schema: {manifest.schema_version}")
    if sha256_file(metadata_path) != manifest.metadata_sha256:
        raise ValueError("metadata hash does not match the manifest")
    if sha256_file(vectors_path) != manifest.vectors_sha256:
        raise ValueError("vector hash does not match the manifest")

    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("metadata schema does not match the manifest")
    entries = [ArtifactEntry.model_validate(value) for value in metadata.get("entries", [])]
    with np.load(vectors_path) as archive:
        matrix = np.asarray(archive["emb"], dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[0] != len(entries):
        raise ValueError("metadata and vector counts do not match")
    if manifest.metadata_count != len(entries) or manifest.vector_count != matrix.shape[0]:
        raise ValueError("manifest counts do not match the artifacts")
    if matrix.shape[1] != manifest.dimension or str(matrix.dtype) != manifest.dtype:
        raise ValueError("manifest vector shape or dtype does not match")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain non-finite values")
    return entries, matrix, manifest
