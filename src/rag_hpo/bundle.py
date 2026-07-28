from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import httpx

from rag_hpo.artifacts import MANIFEST_NAME, META_NAME, VECTOR_NAME, load_artifacts
from rag_hpo.privacy import ensure_private_directory

BUNDLE_VERSION = "v0.2.0-sapbert"
BUNDLE_URL_ENV = "RAG_HPO_VECTOR_BUNDLE_URL"
BUNDLE_SHA_ENV = "RAG_HPO_VECTOR_BUNDLE_SHA256"
BUNDLE_DOWNLOAD_BYTES = 139_934_193
MINIMUM_FREE_BYTES = 500 * 1024 * 1024


def default_cache_root(
    *,
    platform_name: str | None = None,
    environment: dict[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    environment = environment or dict(os.environ)
    platform_name = platform_name or os.name
    home = home or Path.home()
    if platform_name == "nt":
        base = Path(environment.get("LOCALAPPDATA", home / "AppData" / "Local"))
    else:
        base = Path(environment.get("XDG_CACHE_HOME", home / ".cache"))
    return base / "rag-hpo" / BUNDLE_VERSION


def _safe_extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError("vector bundle contains an unsafe path")
        archive.extractall(destination)  # nosec B202


def download_vector_bundle(
    *,
    url: str,
    expected_sha256: str,
    cache_dir: Path,
    client: httpx.Client | None = None,
) -> Path:
    expected_sha256 = expected_sha256.strip().lower()
    if len(expected_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in expected_sha256
    ):
        raise ValueError("vector bundle SHA-256 must be 64 hexadecimal characters")

    ensure_private_directory(cache_dir.parent)
    available = shutil.disk_usage(cache_dir.parent).free
    if available < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            "Insufficient free space for the vector bundle. Free at least 500 MB "
            f"in {cache_dir.parent} and retry."
        )
    temporary_root = Path(tempfile.mkdtemp(prefix=".bundle-", dir=cache_dir.parent))
    archive_path = temporary_root / "bundle.zip"
    extraction_dir = temporary_root / "vectors"
    owns_client = client is None
    download_client = client or httpx.Client(follow_redirects=True, timeout=120.0)
    try:
        digest = hashlib.sha256()
        with download_client.stream("GET", url) as response:
            response.raise_for_status()
            with archive_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    digest.update(chunk)
                    handle.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("downloaded vector bundle SHA-256 does not match")
        extraction_dir.mkdir()
        _safe_extract(archive_path, extraction_dir)
        for required_name in (META_NAME, VECTOR_NAME, MANIFEST_NAME):
            if not (extraction_dir / required_name).is_file():
                raise ValueError(f"vector bundle is missing {required_name}")
        load_artifacts(extraction_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        extraction_dir.replace(cache_dir)
        ensure_private_directory(cache_dir)
        return cache_dir
    finally:
        if owns_client:
            download_client.close()
        shutil.rmtree(temporary_root, ignore_errors=True)


def resolve_vector_dir(vector_dir: Path | None) -> Path:
    if vector_dir is not None:
        load_artifacts(vector_dir)
        return vector_dir

    cache_dir = default_cache_root()
    try:
        load_artifacts(cache_dir)
        return cache_dir
    except (FileNotFoundError, ValueError):
        pass

    url = os.environ.get(BUNDLE_URL_ENV)
    expected_sha256 = os.environ.get(BUNDLE_SHA_ENV)
    if not url or not expected_sha256:
        raise RuntimeError(
            "No validated vector bundle is cached. Pass --vector-dir PATH, or set "
            f"{BUNDLE_URL_ENV} and {BUNDLE_SHA_ENV} to the lab-approved v0.2.0 "
            "release asset. To build locally, run: rag-hpo vectorize --output-dir "
            "vectors --hpo-addons HPO_addons.csv"
        )
    return download_vector_bundle(
        url=url,
        expected_sha256=expected_sha256,
        cache_dir=cache_dir,
    )
