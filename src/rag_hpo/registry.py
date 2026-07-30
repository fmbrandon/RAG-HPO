from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import unicodedata
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_hpo.artifacts import ArtifactEntry, sha256_file
from rag_hpo.privacy import ensure_private_directory, restrict_owner

REGISTRY_SCHEMA_VERSION = "1.0"
LEXICAL_SCHEMA_VERSION = "1.0"
REGISTRY_NAME = "hpo_registry.json"
REGISTRY_MANIFEST_NAME = "hpo_registry_manifest.json"
LEXICAL_NAME = "hpo_lexical.json"
LEXICAL_MANIFEST_NAME = "hpo_lexical_manifest.json"

_MOJIBAKE_MARKERS = (
    "â€™",
    "â€œ",
    "â€\x9d",
    "â€“",
    "â€”",
    "Ã©",
    "Ã¨",
    "Ã±",
)


class RegistryPhrase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_key: str
    phrase: str
    normalized_phrase: str
    source: Literal["hpo-label", "hpo-synonym", "rag-hpo-addon"]
    scope: str


class RegistryConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hp_id: str
    label: str
    definition: str = ""
    alternate_ids: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    obsolete: bool = False
    phrases: list[RegistryPhrase] = Field(default_factory=list)


class RegistryExtension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_key: str
    hp_id: str
    phrase: str
    normalized_phrase: str
    source: Literal["rag-hpo-addon"] = "rag-hpo-addon"
    scope: str = "RELATED"


class AmbiguityGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_phrase: str
    hp_ids: list[str]
    record_keys: list[str]


class HPORegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = REGISTRY_SCHEMA_VERSION
    data_version: str | None
    normalization: str = "Unicode NFKC, casefold, non-alphanumeric to spaces"
    concepts: list[RegistryConcept]
    extensions: list[RegistryExtension] = Field(default_factory=list)
    ambiguity_groups: list[AmbiguityGroup] = Field(default_factory=list)


class RegistryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = REGISTRY_SCHEMA_VERSION
    hpo_source: str
    hpo_sha256: str
    addons_sha256: str | None
    parser: str = "pronto"
    parser_version: str
    registry_sha256: str
    concept_count: int = Field(ge=0)
    official_phrase_count: int = Field(ge=0)
    extension_phrase_count: int = Field(ge=0)
    ambiguity_group_count: int = Field(ge=0)
    utf8_validated: bool = True


class LexicalManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LEXICAL_SCHEMA_VERSION
    registry_schema_version: str = REGISTRY_SCHEMA_VERSION
    registry_sha256: str
    lexical_sha256: str
    normalized_phrase_count: int = Field(ge=0)
    ambiguity_group_count: int = Field(ge=0)


def normalize_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = [character if character.isalnum() else " " for character in normalized]
    return " ".join("".join(characters).split())


def validate_text(value: str, *, context: str) -> None:
    if "\ufffd" in value:
        raise ValueError(f"{context} contains a Unicode replacement character")
    if "\x00" in value:
        raise ValueError(f"{context} contains a NUL character")
    for marker in _MOJIBAKE_MARKERS:
        if marker in value:
            raise ValueError(f"{context} contains likely mojibake: {marker!r}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{context} is not valid Unicode") from exc


def validate_utf8_file(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8") from exc
    validate_text(value, context=str(path))
    return value


def _record_key(
    *,
    hp_id: str,
    phrase: str,
    normalized_phrase: str,
    source: str,
    scope: str,
) -> str:
    payload = json.dumps(
        [hp_id, phrase, normalized_phrase, source, scope],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    ensure_private_directory(path.parent)
    content = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        restrict_owner(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _phrase(
    *,
    hp_id: str,
    phrase: str,
    source: Literal["hpo-label", "hpo-synonym", "rag-hpo-addon"],
    scope: str,
) -> RegistryPhrase:
    phrase = phrase.strip()
    validate_text(phrase, context=f"{hp_id} phrase")
    normalized = normalize_phrase(phrase)
    if not normalized:
        raise ValueError(f"{hp_id} contains an empty normalized phrase")
    return RegistryPhrase(
        record_key=_record_key(
            hp_id=hp_id,
            phrase=phrase,
            normalized_phrase=normalized,
            source=source,
            scope=scope,
        ),
        phrase=phrase,
        normalized_phrase=normalized,
        source=source,
        scope=scope,
    )


def build_registry(
    obo_path: Path,
    *,
    addons_path: Path | None,
    limit: int | None,
) -> HPORegistry:
    validate_utf8_file(obo_path)
    try:
        import pronto
    except ImportError as exc:
        raise RuntimeError("install rag-hpo[vectorize] to build the HPO registry") from exc

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="unsound encoding, assuming ISO-8859-1.*",
            category=UnicodeWarning,
        )
        ontology = pronto.Ontology(obo_path)  # type: ignore[attr-defined]

    all_terms = sorted(ontology.terms(), key=lambda term: str(term.id))
    all_canonical_ids = {str(term.id) for term in all_terms}
    all_aliases = {
        str(alternate_id): str(term.id) for term in all_terms for alternate_id in term.alternate_ids
    }
    terms = all_terms
    if limit is not None:
        terms = terms[:limit]

    concepts: list[RegistryConcept] = []
    canonical_ids: set[str] = set()
    aliases: dict[str, str] = {}
    seen_phrases: set[tuple[str, str]] = set()
    phrase_records: list[tuple[str, RegistryPhrase]] = []
    for term in terms:
        hp_id = str(term.id)
        label = str(term.name or "").strip()
        definition = str(term.definition or "").strip()
        validate_text(label, context=f"{hp_id} label")
        validate_text(definition, context=f"{hp_id} definition")
        if not label:
            raise ValueError(f"{hp_id} has no label")
        canonical_ids.add(hp_id)
        alternate_ids = sorted(str(value) for value in term.alternate_ids)
        for alternate_id in alternate_ids:
            previous = aliases.setdefault(alternate_id, hp_id)
            if previous != hp_id:
                raise ValueError(f"alternate ID {alternate_id} maps to multiple concepts")
        phrases = [_phrase(hp_id=hp_id, phrase=label, source="hpo-label", scope="LABEL")]
        seen_phrases.add((hp_id, phrases[0].phrase.casefold()))
        synonyms = sorted(
            term.synonyms,
            key=lambda item: (
                normalize_phrase(str(item.description)),
                str(item.scope or ""),
                str(item.description),
            ),
        )
        for synonym in synonyms:
            candidate = _phrase(
                hp_id=hp_id,
                phrase=str(synonym.description),
                source="hpo-synonym",
                scope=str(synonym.scope or "RELATED"),
            )
            identity = (hp_id, candidate.phrase.casefold())
            if identity in seen_phrases:
                continue
            seen_phrases.add(identity)
            phrases.append(candidate)
        for item in phrases:
            phrase_records.append((hp_id, item))
        parents = sorted(
            str(parent.id) for parent in term.superclasses(distance=1, with_self=False)
        )
        concepts.append(
            RegistryConcept(
                hp_id=hp_id,
                label=label,
                definition=definition,
                alternate_ids=alternate_ids,
                parents=parents,
                obsolete=bool(term.obsolete),
                phrases=phrases,
            )
        )

    extensions: list[RegistryExtension] = []
    if addons_path is not None:
        validate_utf8_file(addons_path)
        with addons_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"HP_ID", "info"} <= set(reader.fieldnames):
                raise ValueError("HPO add-on CSV requires HP_ID and info columns")
            for row_number, row in enumerate(reader, start=2):
                source_id = str(row.get("HP_ID") or "").strip()
                hp_id = all_aliases.get(source_id, source_id)
                phrase = str(row.get("info") or "").strip()
                if not source_id or not phrase:
                    raise ValueError(f"HPO add-on row {row_number} is incomplete")
                if hp_id not in all_canonical_ids:
                    raise ValueError(
                        f"HPO add-on row {row_number} references unknown ID {source_id}"
                    )
                if hp_id not in canonical_ids:
                    continue
                item = _phrase(
                    hp_id=hp_id,
                    phrase=phrase,
                    source="rag-hpo-addon",
                    scope="RELATED",
                )
                identity = (hp_id, item.phrase.casefold())
                if identity in seen_phrases:
                    continue
                seen_phrases.add(identity)
                extension = RegistryExtension(hp_id=hp_id, **item.model_dump())
                extensions.append(extension)
                phrase_records.append((hp_id, item))

    by_phrase: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for hp_id, item in phrase_records:
        by_phrase[item.normalized_phrase].append((hp_id, item.record_key))
    ambiguity_groups = [
        AmbiguityGroup(
            normalized_phrase=normalized,
            hp_ids=sorted({hp_id for hp_id, _ in values}),
            record_keys=sorted({record_key for _, record_key in values}),
        )
        for normalized, values in sorted(by_phrase.items())
        if len({hp_id for hp_id, _ in values}) > 1
    ]
    return HPORegistry(
        data_version=str(ontology.metadata.data_version or "") or None,
        concepts=concepts,
        extensions=extensions,
        ambiguity_groups=ambiguity_groups,
    )


def registry_to_artifact_entries(registry: HPORegistry) -> list[ArtifactEntry]:
    concepts = {concept.hp_id: concept for concept in registry.concepts}
    entries: list[ArtifactEntry] = []
    for concept in registry.concepts:
        if concept.obsolete:
            continue
        for phrase in concept.phrases:
            entries.append(
                ArtifactEntry(
                    hp_id=concept.hp_id,
                    phrase=phrase.phrase,
                    term=concept.label,
                    definition=concept.definition,
                    source=phrase.source,
                )
            )
    for extension in registry.extensions:
        concept = concepts[extension.hp_id]
        if concept.obsolete:
            continue
        entries.append(
            ArtifactEntry(
                hp_id=extension.hp_id,
                phrase=extension.phrase,
                term=concept.label,
                definition=concept.definition,
                source="hpo-addon",
            )
        )
    return entries


def _lexical_payload(registry: HPORegistry) -> dict[str, Any]:
    values: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"hp_ids": set(), "record_keys": set()}
    )
    for concept in registry.concepts:
        if concept.obsolete:
            continue
        for phrase in concept.phrases:
            values[phrase.normalized_phrase]["hp_ids"].add(concept.hp_id)
            values[phrase.normalized_phrase]["record_keys"].add(phrase.record_key)
    for extension in registry.extensions:
        values[extension.normalized_phrase]["hp_ids"].add(extension.hp_id)
        values[extension.normalized_phrase]["record_keys"].add(extension.record_key)
    return {
        "schema_version": LEXICAL_SCHEMA_VERSION,
        "normalization": registry.normalization,
        "entries": {
            normalized: {
                "hp_ids": sorted(value["hp_ids"]),
                "record_keys": sorted(value["record_keys"]),
            }
            for normalized, value in sorted(values.items())
        },
    }


def write_registry_bundle(
    output_dir: Path,
    *,
    registry: HPORegistry,
    hpo_source: str,
    hpo_sha256: str,
    addons_sha256: str | None,
) -> tuple[RegistryManifest, LexicalManifest]:
    ensure_private_directory(output_dir)
    registry_path = output_dir / REGISTRY_NAME
    _atomic_write_json(registry_path, registry.model_dump(mode="json"))
    registry_sha256 = sha256_file(registry_path)

    lexical_path = output_dir / LEXICAL_NAME
    lexical_payload = _lexical_payload(registry)
    _atomic_write_json(lexical_path, lexical_payload)
    lexical_sha256 = sha256_file(lexical_path)

    official_phrase_count = sum(len(concept.phrases) for concept in registry.concepts)
    registry_manifest = RegistryManifest(
        hpo_source=hpo_source,
        hpo_sha256=hpo_sha256,
        addons_sha256=addons_sha256,
        parser_version=importlib.metadata.version("pronto"),
        registry_sha256=registry_sha256,
        concept_count=len(registry.concepts),
        official_phrase_count=official_phrase_count,
        extension_phrase_count=len(registry.extensions),
        ambiguity_group_count=len(registry.ambiguity_groups),
    )
    _atomic_write_json(
        output_dir / REGISTRY_MANIFEST_NAME,
        registry_manifest.model_dump(mode="json"),
    )
    lexical_manifest = LexicalManifest(
        registry_sha256=registry_sha256,
        lexical_sha256=lexical_sha256,
        normalized_phrase_count=len(lexical_payload["entries"]),
        ambiguity_group_count=len(registry.ambiguity_groups),
    )
    _atomic_write_json(
        output_dir / LEXICAL_MANIFEST_NAME,
        lexical_manifest.model_dump(mode="json"),
    )
    return registry_manifest, lexical_manifest


def load_registry_bundle(
    output_dir: Path,
) -> tuple[HPORegistry, RegistryManifest, LexicalManifest]:
    registry_path = output_dir / REGISTRY_NAME
    registry_manifest_path = output_dir / REGISTRY_MANIFEST_NAME
    lexical_path = output_dir / LEXICAL_NAME
    lexical_manifest_path = output_dir / LEXICAL_MANIFEST_NAME
    for path in (
        registry_path,
        registry_manifest_path,
        lexical_path,
        lexical_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required registry artifact is missing: {path}")

    registry_manifest = RegistryManifest.model_validate_json(
        registry_manifest_path.read_text(encoding="utf-8")
    )
    lexical_manifest = LexicalManifest.model_validate_json(
        lexical_manifest_path.read_text(encoding="utf-8")
    )
    if sha256_file(registry_path) != registry_manifest.registry_sha256:
        raise ValueError("registry hash does not match its manifest")
    if sha256_file(lexical_path) != lexical_manifest.lexical_sha256:
        raise ValueError("lexical index hash does not match its manifest")
    if lexical_manifest.registry_sha256 != registry_manifest.registry_sha256:
        raise ValueError("lexical index was derived from a different registry")

    registry = HPORegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))
    if registry.schema_version != REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"unsupported registry schema: {registry.schema_version}")
    lexical: dict[str, Any] = json.loads(lexical_path.read_text(encoding="utf-8"))
    if lexical.get("schema_version") != LEXICAL_SCHEMA_VERSION:
        raise ValueError("unsupported lexical index schema")
    expected = _lexical_payload(registry)
    if lexical != expected:
        raise ValueError("lexical index content does not match the registry")
    return registry, registry_manifest, lexical_manifest


def registry_phrase_index(registry: HPORegistry) -> dict[str, tuple[str, ...]]:
    lexical = _lexical_payload(registry)["entries"]
    return {normalized: tuple(value["hp_ids"]) for normalized, value in lexical.items()}


def registry_aliases(registry: HPORegistry) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for concept in registry.concepts:
        aliases[concept.hp_id] = concept.hp_id
        aliases.update({alternate_id: concept.hp_id for alternate_id in concept.alternate_ids})
    return aliases


def registry_labels(registry: HPORegistry) -> dict[str, str]:
    return {concept.hp_id: concept.label for concept in registry.concepts}


class HPOTermRegistry:
    def __init__(
        self,
        raw_ontology_terms: dict[str, dict[str, Any]] | None = None,
        acronym_dictionary_path: str | Path | None = None,
    ) -> None:
        """raw_ontology_terms: Parsed HPO obo/json dictionary structure."""
        self.raw_terms: dict[str, dict[str, Any]] = raw_ontology_terms or {}
        self.active_terms: dict[str, dict[str, Any]] = {}
        self.obsolete_map: dict[str, str] = {}
        self.acronym_dict: dict[str, str] = {}

        if acronym_dictionary_path:
            self.load_acronym_dictionary(acronym_dictionary_path)

        if raw_ontology_terms:
            self._build_resolved_registry()

    def load_acronym_dictionary(self, path: str | Path) -> None:
        """Loads external UMLS/ADAM acronym dictionary JSON."""
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as f:
            external_data = json.load(f)
            self.acronym_dict.update(external_data)

    def _build_resolved_registry(self) -> None:
        for term_id, term_data in self.raw_terms.items():
            if term_data.get("is_obsolete", False) or term_data.get("obsolete", False):
                replacement = term_data.get("replaced_by")
                if not replacement and term_data.get("consider"):
                    consider_list = term_data.get("consider")
                    if isinstance(consider_list, list) and consider_list:
                        replacement = str(consider_list[0])

                if replacement:
                    self.obsolete_map[term_id] = str(replacement)
            else:
                self.active_terms[term_id] = term_data

                synonyms = term_data.get("synonyms", [])
                primary_name = str(term_data.get("name") or term_data.get("label", ""))

                for syn in synonyms:
                    if isinstance(syn, dict):
                        syn_str = str(syn.get("phrase", ""))
                    else:
                        syn_str = str(syn)
                    if re.match(r"^[A-Z0-9]{2,8}$", syn_str):
                        if syn_str not in self.acronym_dict and primary_name:
                            self.acronym_dict[syn_str] = primary_name

    def resolve_term_id(self, hpo_id: str) -> str:
        """Recursively resolves deprecated HPO IDs to active terms with loop protection."""
        visited: set[str] = set()
        current_id = hpo_id

        while current_id in self.obsolete_map:
            if current_id in visited:
                break
            visited.add(current_id)
            current_id = self.obsolete_map[current_id]

        return current_id

    def expand_acronyms(self, text: str) -> str:
        """Applies word-boundary expansion across all registered acronyms."""
        if not self.acronym_dict:
            return text

        def replace_match(match: re.Match[str]) -> str:
            word = str(match.group(0))
            expansion = self.acronym_dict.get(word)
            if expansion:
                return f"{word} ({expansion})"
            return word

        pattern = re.compile(r"\b[A-Z0-9]{2,8}\b")
        return pattern.sub(replace_match, text)

    @classmethod
    def from_hpo_registry(
        cls,
        registry: HPORegistry,
        acronym_dictionary_path: str | Path | None = None,
    ) -> HPOTermRegistry:
        """Constructs an HPOTermRegistry directly from an HPORegistry instance."""
        terms: dict[str, dict[str, Any]] = {}
        for concept in registry.concepts:
            syn_phrases = [p.phrase for p in concept.phrases if p.source == "hpo-synonym"]
            terms[concept.hp_id] = {
                "name": concept.label,
                "label": concept.label,
                "is_obsolete": concept.obsolete,
                "obsolete": concept.obsolete,
                "synonyms": syn_phrases,
            }
            for alt_id in concept.alternate_ids:
                terms[alt_id] = {
                    "is_obsolete": True,
                    "replaced_by": concept.hp_id,
                }
        return cls(raw_ontology_terms=terms, acronym_dictionary_path=acronym_dictionary_path)
