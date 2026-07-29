from __future__ import annotations

import contextlib
import csv
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.resources
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.registry import (
    REGISTRY_MANIFEST_NAME,
    HPORegistry,
    load_registry_bundle,
    normalize_phrase,
    registry_aliases,
    registry_labels,
    registry_phrase_index,
    validate_text,
)

FASTHPOCR_VERSION = "0.1.4"
INDEX_NAME = "hp.index"
INDEX_MANIFEST_NAME = "fasthpocr_index_manifest.json"
INDEX_LOG_NAME = "fasthpocr_index_build.log"


@dataclass(frozen=True)
class FastHPOAnnotation:
    phrase: str
    hpo_id: str
    hpo_term: str
    start_offset: int
    end_offset: int
    candidate_hpo_ids: tuple[str, ...] = ()
    resolution: str = "upstream"


def _write_private(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    restrict_owner(path)


def _resource_details(relative_path: str) -> dict[str, int | str]:
    resource = importlib.resources.files("FastHPOCR").joinpath(relative_path)
    with importlib.resources.as_file(resource) as path:
        return {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }


def _write_external_synonyms(addons_path: Path, output_path: Path) -> int:
    rows: list[tuple[str, str]] = []
    with addons_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"HP_ID", "info"} <= set(reader.fieldnames):
            raise ValueError("HPO add-on CSV requires HP_ID and info columns")
        for row in reader:
            hpo_id = str(row.get("HP_ID") or "").strip()
            phrase = str(row.get("info") or "").strip().replace("\n", " ")
            if hpo_id.startswith("HP:") and phrase:
                rows.append((hpo_id, phrase))
    _write_private(
        output_path,
        "".join(f"{hpo_id}={phrase}\n" for hpo_id, phrase in rows),
    )
    return len(rows)


def _write_registry_extensions(registry: HPORegistry, output_path: Path) -> int:
    rows = sorted(
        (
            extension.hp_id,
            extension.phrase.replace("\n", " "),
            extension.record_key,
        )
        for extension in registry.extensions
    )
    _write_private(
        output_path,
        "".join(f"{hp_id}={phrase}\n" for hp_id, phrase, _ in rows),
    )
    return len(rows)


def _expected_manifest(
    ontology_path: Path,
    addons_path: Path | None,
    config: dict[str, Any],
    registry_dir: Path | None,
) -> dict[str, Any]:
    expected = {
        "schema_version": "1.0",
        "fast_hpo_cr_version": importlib.metadata.version("FastHPOCR"),
        "ontology_sha256": sha256_file(ontology_path),
        "addons_sha256": sha256_file(addons_path) if addons_path else None,
        "config": dict(sorted(config.items())),
        "resources": {
            "license.txt": _resource_details("license.txt"),
            "resources/base-synonyms": _resource_details("resources/base-synonyms"),
            "resources/vocab.clusters.list": _resource_details("resources/vocab.clusters.list"),
        },
    }
    if registry_dir is not None:
        registry, registry_manifest, _ = load_registry_bundle(registry_dir)
        del registry
        expected["registry"] = {
            "schema_version": registry_manifest.schema_version,
            "registry_sha256": registry_manifest.registry_sha256,
            "manifest_sha256": sha256_file(registry_dir / REGISTRY_MANIFEST_NAME),
        }
    return expected


def _semantic_index_fingerprint(index_path: Path) -> dict[str, int | str]:
    text = index_path.read_text(encoding="utf-8")
    validate_text(text, context=str(index_path))
    data: Any = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("termData"), list):
        canonical = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "schema_version": "1.0",
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "term_count": 0,
            "label_count": 0,
        }

    clusters = {
        str(cluster_id): tuple(sorted(str(value) for value in values))
        for cluster_id, values in dict(data.get("clusters") or {}).items()
    }
    term_digests: list[str] = []
    label_count = 0
    for term in data["termData"]:
        if not isinstance(term, dict):
            raise ValueError("FastHPOCR termData contains a non-object value")
        uri = str(term.get("uri") or "")
        validate_text(uri, context="FastHPOCR term URI")
        labels: list[dict[str, Any]] = []
        for label in term.get("labels") or []:
            if not isinstance(label, dict):
                raise ValueError("FastHPOCR label contains a non-object value")
            original = str(label.get("originalLabel") or "")
            validate_text(original, context=f"FastHPOCR {uri} label")
            tokens = [
                list(clusters.get(str(token), (str(token),))) for token in label.get("tokens") or []
            ]
            token_set = sorted(
                [
                    list(clusters.get(str(token), (str(token),)))
                    for token in label.get("tokenSet") or []
                ]
            )
            labels.append(
                {
                    "native": bool(label.get("native")),
                    "original_label": original,
                    "tokens": tokens,
                    "token_set": token_set,
                }
            )
            label_count += 1
        record = {
            "uri": uri,
            "categories": sorted(str(value) for value in term.get("categories") or []),
            "labels": sorted(
                labels,
                key=lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        }
        canonical = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        term_digests.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
    digest = hashlib.sha256()
    for value in sorted(term_digests):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return {
        "schema_version": "1.0",
        "sha256": digest.hexdigest(),
        "term_count": len(data["termData"]),
        "label_count": label_count,
    }


@functools.lru_cache(maxsize=2)
def _load_morphological_index(
    index_path_value: str,
    index_sha256: str,
) -> tuple[dict[str, str], dict[tuple[str, ...], tuple[str, ...]]]:
    del index_sha256
    index_path = Path(index_path_value)
    text = index_path.read_text(encoding="utf-8")
    validate_text(text, context=str(index_path))
    data: Any = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("FastHPOCR index must be a JSON object")
    token_clusters: dict[str, str] = {}
    for cluster_id, values in dict(data.get("clusters") or {}).items():
        for value in values:
            normalized = normalize_phrase(str(value))
            if normalized and " " not in normalized:
                previous = token_clusters.setdefault(normalized, str(cluster_id))
                if previous != str(cluster_id):
                    raise ValueError(f"FastHPOCR token {normalized!r} belongs to multiple clusters")
    signatures: dict[tuple[str, ...], set[str]] = {}
    for term in data.get("termData") or []:
        if not isinstance(term, dict):
            continue
        uri = str(term.get("uri") or "")
        for label in term.get("labels") or []:
            if not isinstance(label, dict):
                continue
            signature = tuple(sorted(str(value) for value in label.get("tokenSet") or []))
            if signature:
                signatures.setdefault(signature, set()).add(uri)
    return token_clusters, {
        signature: tuple(sorted(values)) for signature, values in signatures.items()
    }


def _probe_phrases(registry: HPORegistry) -> list[str]:
    phrases = {
        phrase.phrase
        for concept in registry.concepts
        for phrase in concept.phrases
        if not concept.obsolete
    }
    phrases.update(extension.phrase for extension in registry.extensions)
    ambiguity_phrases = {group.normalized_phrase for group in registry.ambiguity_groups}
    representatives = sorted(phrases, key=lambda value: (normalize_phrase(value), value))[:256]
    return sorted(
        ambiguity_phrases | set(representatives),
        key=lambda value: (normalize_phrase(value), value),
    )


def _recognizer_output_fingerprint(
    index_path: Path,
    registry: HPORegistry,
) -> dict[str, int | str]:
    raw_recognizer = FastHPORecognizer(index_path, longest_match=True)
    normalized_recognizer = FastHPORecognizer(
        index_path,
        longest_match=True,
        registry=registry,
    )
    raw_digest = hashlib.sha256()
    normalized_digest = hashlib.sha256()
    probes = _probe_phrases(registry)
    for phrase in probes:
        raw = [
            annotation.__dict__
            for annotation in sorted(
                raw_recognizer.annotate(phrase),
                key=lambda item: (
                    item.start_offset,
                    item.end_offset,
                    item.hpo_id,
                    item.phrase,
                ),
            )
        ]
        normalized = [annotation.__dict__ for annotation in normalized_recognizer.annotate(phrase)]
        for digest, values in ((raw_digest, raw), (normalized_digest, normalized)):
            payload = json.dumps(
                [phrase, values],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update(payload.encode("utf-8"))
            digest.update(b"\n")
    return {
        "schema_version": "1.0",
        "probe_count": len(probes),
        "raw_sha256": raw_digest.hexdigest(),
        "normalized_sha256": normalized_digest.hexdigest(),
    }


def build_fasthpocr_index(
    *,
    ontology_path: Path,
    index_dir: Path,
    addons_path: Path | None = None,
    registry_dir: Path | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    """Build or validate a private FastHPOCR index from one pinned ontology."""
    if importlib.metadata.version("FastHPOCR") != FASTHPOCR_VERSION:
        raise RuntimeError(f"FastHPOCR {FASTHPOCR_VERSION} is required for this benchmark")
    if not ontology_path.is_file():
        raise FileNotFoundError(f"ontology file does not exist: {ontology_path}")
    if addons_path is not None and not addons_path.is_file():
        raise FileNotFoundError(f"add-on file does not exist: {addons_path}")

    config: dict[str, Any] = {
        "allow3LetterAcronyms": False,
        "allowDuplicateEntries": False,
        "includeTopLevelCategory": True,
        "external_synonyms": addons_path is not None,
    }
    registry: HPORegistry | None = None
    if registry_dir is not None:
        registry, registry_manifest, _ = load_registry_bundle(registry_dir)
        if registry_manifest.hpo_sha256 != sha256_file(ontology_path):
            raise ValueError("FastHPOCR ontology differs from the canonical registry")
        expected_addons = sha256_file(addons_path) if addons_path else None
        if registry_manifest.addons_sha256 != expected_addons:
            raise ValueError("FastHPOCR add-ons differ from the canonical registry")
    expected = _expected_manifest(ontology_path, addons_path, config, registry_dir)
    ensure_private_directory(index_dir)
    index_path = index_dir / INDEX_NAME
    manifest_path = index_dir / INDEX_MANIFEST_NAME
    if index_path.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ValueError(
                "existing FastHPOCR index provenance differs; use a separate index directory"
            )
        if existing.get("index_sha256") != sha256_file(index_path):
            raise ValueError("FastHPOCR index hash does not match its manifest")
        semantic = _semantic_index_fingerprint(index_path)
        if existing.get("semantic_index_fingerprint") != semantic:
            raise ValueError("FastHPOCR semantic index fingerprint does not match")
        if registry is not None:
            normalized_output = _recognizer_output_fingerprint(index_path, registry)
            recorded_output = existing.get("recognizer_output_fingerprint") or {}
            if recorded_output.get("normalized_sha256") != normalized_output["normalized_sha256"]:
                raise ValueError("FastHPOCR normalized output fingerprint does not match")
        return index_path, existing, True
    if index_path.exists() or manifest_path.exists():
        raise ValueError("FastHPOCR index directory contains an incomplete artifact")

    ensure_private_directory(index_dir.parent)
    with tempfile.TemporaryDirectory(dir=index_dir.parent, prefix=".fasthpocr-build-") as temporary:
        temporary_dir = Path(temporary)
        external_path: Path | None = None
        external_count = 0
        upstream_config: dict[str, Any] = {
            "allow3LetterAcronyms": False,
            "allowDuplicateEntries": False,
            "includeTopLevelCategory": True,
        }
        if addons_path is not None:
            external_path = temporary_dir / "external-synonyms.txt"
            external_count = (
                _write_registry_extensions(registry, external_path)
                if registry is not None
                else _write_external_synonyms(addons_path, external_path)
            )
            upstream_config["externalSynFile"] = str(external_path)

        module = importlib.import_module("FastHPOCR.IndexHPO")
        index_class = module.IndexHPO
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            index_class(
                str(ontology_path),
                str(temporary_dir),
                indexConfig=upstream_config,
            ).index()
        built_index = temporary_dir / INDEX_NAME
        if not built_index.is_file():
            raise RuntimeError("FastHPOCR did not produce hp.index")
        restrict_owner(built_index)
        os.replace(built_index, index_path)

        manifest = {
            **expected,
            "external_synonym_count": external_count,
            "index_sha256": sha256_file(index_path),
            "semantic_index_fingerprint": _semantic_index_fingerprint(index_path),
        }
        if registry is not None:
            manifest["recognizer_output_fingerprint"] = _recognizer_output_fingerprint(
                index_path, registry
            )
        _write_private(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _write_private(index_dir / INDEX_LOG_NAME, output.getvalue())
    return index_path, manifest, False


class FastHPORecognizer:
    def __init__(
        self,
        index_path: Path,
        *,
        longest_match: bool = False,
        registry: HPORegistry | None = None,
    ) -> None:
        if not index_path.is_file():
            raise FileNotFoundError(f"FastHPOCR index does not exist: {index_path}")
        module = importlib.import_module("FastHPOCR.HPOAnnotator")
        annotator_class = module.HPOAnnotator
        self._annotator = annotator_class(str(index_path))
        self.longest_match = longest_match
        self._registry = registry
        self._phrase_index = registry_phrase_index(registry) if registry else {}
        self._aliases = registry_aliases(registry) if registry else {}
        self._labels = registry_labels(registry) if registry else {}
        if registry:
            self._token_clusters, self._signature_ids = _load_morphological_index(
                str(index_path.resolve()),
                sha256_file(index_path),
            )
        else:
            self._token_clusters = {}
            self._signature_ids = {}

    def _morphological_candidates(self, phrase: str) -> tuple[str, ...]:
        tokens = normalize_phrase(phrase).split()
        cluster_ids = [self._token_clusters.get(token) for token in tokens]
        if not cluster_ids or any(value is None for value in cluster_ids):
            return ()
        signature = tuple(sorted({str(value) for value in cluster_ids}))
        canonical = {
            self._aliases.get(hp_id, hp_id) for hp_id in self._signature_ids.get(signature, ())
        }
        return tuple(sorted(hp_id for hp_id in canonical if hp_id in self._labels))

    def annotate(self, text: str) -> list[FastHPOAnnotation]:
        values = self._annotator.annotate(text, longestMatch=self.longest_match)
        annotations: list[FastHPOAnnotation] = []
        for value in values:
            phrase = str(value.getTextSpan())
            upstream_id = str(value.getHPOUri())
            if self._registry is None:
                annotation = FastHPOAnnotation(
                    phrase=phrase,
                    hpo_id=upstream_id,
                    hpo_term=str(value.getHPOLabel()),
                    start_offset=int(value.getStartOffset()),
                    end_offset=int(value.getEndOffset()),
                )
            else:
                explicit_candidates = self._phrase_index.get(
                    normalize_phrase(phrase),
                    (),
                )
                candidates = (
                    explicit_candidates
                    if explicit_candidates
                    else self._morphological_candidates(phrase)
                )
                if len(candidates) > 1:
                    annotation = FastHPOAnnotation(
                        phrase=phrase,
                        hpo_id="",
                        hpo_term="",
                        start_offset=int(value.getStartOffset()),
                        end_offset=int(value.getEndOffset()),
                        candidate_hpo_ids=candidates,
                        resolution="ambiguous",
                    )
                else:
                    canonical_id = (
                        candidates[0] if candidates else self._aliases.get(upstream_id, upstream_id)
                    )
                    if canonical_id not in self._labels:
                        raise ValueError(f"FastHPOCR returned unknown registry ID {upstream_id}")
                    annotation = FastHPOAnnotation(
                        phrase=phrase,
                        hpo_id=canonical_id,
                        hpo_term=self._labels[canonical_id],
                        start_offset=int(value.getStartOffset()),
                        end_offset=int(value.getEndOffset()),
                        candidate_hpo_ids=(canonical_id,),
                        resolution="registry",
                    )
            annotations.append(annotation)
        return sorted(
            {
                (
                    item.start_offset,
                    item.end_offset,
                    item.hpo_id,
                    item.candidate_hpo_ids,
                ): item
                for item in annotations
            }.values(),
            key=lambda item: (
                item.start_offset,
                item.end_offset,
                item.hpo_id,
                item.candidate_hpo_ids,
                item.phrase,
            ),
        )
