from __future__ import annotations

import csv
import importlib.metadata
import os
import tempfile
from pathlib import Path

import httpx

from rag_hpo.artifacts import ArtifactEntry, ArtifactManifest, sha256_file, write_artifacts
from rag_hpo.embeddings import create_backend
from rag_hpo.privacy import ensure_private_directory

DEFAULT_HPO_URL = "https://purl.obolibrary.org/obo/hp.obo"


def acquire_ontology(
    *,
    output_dir: Path,
    obo_file: Path | None,
    obo_url: str,
    refresh: bool,
    offline: bool,
) -> tuple[Path, str]:
    if obo_file is not None:
        if not obo_file.is_file():
            raise FileNotFoundError(f"ontology file does not exist: {obo_file}")
        return obo_file, str(obo_file.resolve())

    cached = output_dir / "hp.obo"
    if cached.is_file() and not refresh:
        return cached, obo_url
    if offline:
        raise FileNotFoundError(f"offline mode requires a cached ontology at {cached}")

    ensure_private_directory(output_dir)
    with httpx.stream("GET", obo_url, timeout=120.0, follow_redirects=True) as response:
        response.raise_for_status()
        fd, temporary = tempfile.mkstemp(dir=output_dir, prefix=".hp.obo.")
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, cached)
        finally:
            temporary_path.unlink(missing_ok=True)
    return cached, obo_url


def build_entries(
    obo_path: Path,
    *,
    addons_path: Path | None,
    limit: int | None,
) -> list[ArtifactEntry]:
    try:
        import pronto
    except ImportError as exc:
        raise RuntimeError("install rag-hpo[vectorize] to build HPO artifacts") from exc

    ontology = pronto.Ontology(obo_path)  # type: ignore[attr-defined]
    terms = sorted(ontology.terms(), key=lambda term: str(term.id))
    if limit is not None:
        terms = terms[:limit]

    entries: list[ArtifactEntry] = []
    known_ids: set[str] = set()
    seen: set[tuple[str, str]] = set()
    labels: dict[str, str] = {}
    definitions: dict[str, str] = {}

    def add(hp_id: str, phrase: str, term: str, definition: str, source: str) -> None:
        phrase = phrase.strip()
        if not phrase:
            return
        identity = (hp_id, phrase.casefold())
        if identity in seen:
            return
        seen.add(identity)
        entries.append(
            ArtifactEntry(
                hp_id=hp_id,
                phrase=phrase,
                term=term,
                definition=definition,
                source=source,
            )
        )

    for term in terms:
        hp_id = str(term.id)
        label = str(term.name or "").strip()
        definition = str(term.definition or "")
        known_ids.add(hp_id)
        labels[hp_id] = label
        definitions[hp_id] = definition
        add(hp_id, label, label, definition, "hpo-label")
        for synonym in sorted(term.synonyms, key=lambda item: item.description.casefold()):
            add(hp_id, synonym.description, label, definition, "hpo-synonym")

    if addons_path is not None:
        if not addons_path.is_file():
            raise FileNotFoundError(f"HPO add-on file does not exist: {addons_path}")
        with addons_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                hp_id = (row.get("HP_ID") or "").strip()
                phrase = (row.get("info") or "").strip()
                if hp_id in known_ids:
                    add(
                        hp_id,
                        phrase,
                        labels[hp_id],
                        definitions[hp_id],
                        "hpo-addon",
                    )
    return entries


def vectorize(
    *,
    output_dir: Path,
    obo_file: Path | None = None,
    obo_url: str = DEFAULT_HPO_URL,
    addons_path: Path | None = None,
    backend_name: str = "sapbert",
    limit: int | None = None,
    refresh: bool = False,
    offline: bool = False,
) -> ArtifactManifest:
    ontology_path, source = acquire_ontology(
        output_dir=output_dir,
        obo_file=obo_file,
        obo_url=obo_url,
        refresh=refresh,
        offline=offline,
    )
    entries = build_entries(ontology_path, addons_path=addons_path, limit=limit)
    if not entries:
        raise ValueError("ontology extraction produced no entries")
    backend = create_backend(backend_name)
    vectors = backend.encode([entry.phrase for entry in entries])
    return write_artifacts(
        output_dir,
        entries=entries,
        vectors=vectors,
        hpo_source=source,
        hpo_sha256=sha256_file(ontology_path),
        addons_sha256=sha256_file(addons_path) if addons_path else None,
        parser="pronto",
        parser_version=importlib.metadata.version("pronto"),
        embedding_backend=backend.name,
        embedding_model=backend.model_id,
        embedding_revision=backend.revision,
    )
