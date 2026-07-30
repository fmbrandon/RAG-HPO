from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel
from rapidfuzz import fuzz

from rag_hpo import __version__
from rag_hpo.artifacts import (
    MANIFEST_NAME,
    ArtifactManifest,
    load_artifacts,
    sha256_file,
)
from rag_hpo.artifacts import SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION
from rag_hpo.assertion import AssertionDecision, analyze_assertion
from rag_hpo.calibration import (
    STAGED_PIPELINE_SCHEMA_VERSION,
    ConfidenceCalibration,
    apply_calibration,
)
from rag_hpo.embeddings import EmbeddingBackend, create_backend
from rag_hpo.export import export_results
from rag_hpo.fasthpocr import FastHPORecognizer
from rag_hpo.lexical import NativeLexicalRecognizer, SINGLE_TOKEN_MODIFIER_BLOCKLIST
from rag_hpo.lexical_rescue import LexicalRescueEngine
from rag_hpo.registry import HPORegistry, HPOTermRegistry, load_registry_bundle
from rag_hpo.models import (
    AnnotationInput,
    AnnotationResult,
    Candidate,
    Category,
    FinalCategoryDecisionBatch,
    MappingSetDecisionBatch,
    PhenotypeSpanExtraction,
)
from rag_hpo.pipeline import _short_error, hash_inputs
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.prompts import load_prompts
from rag_hpo.provider import ProviderError
from rag_hpo.registry import load_registry_bundle, normalize_phrase
from rag_hpo.retrieval import RETRIEVAL_POLICY_VERSION, HybridCandidateRetriever
from rag_hpo.state import PipelineState


class AnnotationMode(StrEnum):
    MODEL = "model"
    BALANCED = "balanced"
    HIGH_RECALL = "high-recall"
    NATIVE = "native"
    FASTHPOCR = "fasthpocr"


class MappingPromptMode(StrEnum):
    ZERO_SHOT = "zero-shot"
    ONE_SHOT = "one-shot"


class StagedProvider(Protocol):
    def request(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[BaseModel],
        temperature: float = 0.2,
    ) -> tuple[Any, str]: ...


@dataclass(frozen=True)
class SentenceSpan:
    sentence_id: str
    start: int
    end: int
    text: str


@dataclass
class Mention:
    phrase: str
    start: int
    end: int
    category: Category | None = None
    methods: set[str] = field(default_factory=set)
    recognizer_ids: set[str] = field(default_factory=set)
    modifier_ids: set[str] = field(default_factory=set)
    evidence_segments: list[tuple[int, int]] = field(default_factory=list)
    phrase_variants: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ContextPacket:
    section: str | None
    subject: str
    assertion: str
    sentence: str
    preceding_sentence: str | None
    following_sentence: str | None
    extended_context_reason: str | None


@dataclass(frozen=True)
class NoteContext:
    """Per-note parsing shared by extraction, retrieval, and mapping stages."""

    text: str
    sentences: tuple[SentenceSpan, ...]

    @classmethod
    def build(cls, text: str) -> NoteContext:
        return cls(text=text, sentences=tuple(sentence_spans(text)))

    def sentence_index(self, start: int, end: int) -> int | None:
        return next(
            (
                index
                for index, sentence in enumerate(self.sentences)
                if _overlaps(sentence.start, sentence.end, start, end)
            ),
            None,
        )


_SENTENCE = re.compile(r"[^\n.!?]+(?:[.!?]+|(?=\n)|$)", re.UNICODE)
_LIST_CUE = re.compile(r"[,;]|\b(?:and|with|including)\b", re.I)
_MEASUREMENT_CUE = re.compile(
    r"\b(?:high|low|raised|reduced|decreased|increased|elevated|abnormal)\b|"
    r"\d+(?:\.\d+)?\s*(?:mg|g|mmol|µmol|cm|mm|kg|%|bpm|mmhg)\b",
    re.I,
)
_STRUCTURAL_CUE = re.compile(
    r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+|"
    r"\b(?:exam(?:ination)?|findings?|features?|phenotypes?|imaging|mri|ct|"
    r"ultrasound|laboratory|labs?)\s*:|"
    r"\b(?:showed|shows|revealed|demonstrated|noted|observed|presented with)\b",
    re.I,
)
MAPPING_BATCH_SIZE = 8
FINAL_CATEGORY_BATCH_SIZE = 16
_CONTEXT_CUE = re.compile(
    r"\b(?:it|this|these|those|they|former|latter|respectively|"
    r"however|therefore|subsequently|also|both)\b",
    re.I,
)
_MAX_ADJACENT_CONTEXT_CHARS = 280
_TOKEN = re.compile(r"\w+(?:[-'\u2019]\w+)*", re.UNICODE)
_COORDINATED_MAX_GAP_CHARS = 120
_CLINICAL_MODIFIER_ROOT = "HP:0012823"


def sentence_spans(text: str) -> list[SentenceSpan]:
    values: list[SentenceSpan] = []
    for index, match in enumerate(_SENTENCE.finditer(text)):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start = match.start() + left
        end = match.start() + right
        if end > start:
            values.append(
                SentenceSpan(
                    sentence_id=f"s{index:04d}",
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
    if not values and text:
        values.append(SentenceSpan("s0000", 0, len(text), text))
    return values


def _overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and end > other_start


def _mention_segments(mention: Mention) -> list[tuple[int, int]]:
    return mention.evidence_segments or [(mention.start, mention.end)]


def _coordinated_alignment(
    phrase: str,
    note: str,
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Align an ordered, non-contiguous model phrase without inventing offsets."""

    phrase_tokens = [normalize_phrase(match.group(0)) for match in _TOKEN.finditer(phrase)]
    if len(phrase_tokens) < 2:
        return None
    for range_start, range_end in ranges:
        source_tokens = [
            (
                normalize_phrase(match.group(0)),
                range_start + match.start(),
                range_start + match.end(),
            )
            for match in _TOKEN.finditer(note[range_start:range_end])
        ]
        for source_index, (source_token, start, end) in enumerate(source_tokens):
            if source_token != phrase_tokens[0]:
                continue
            matched = [(start, end)]
            cursor = source_index + 1
            for phrase_token in phrase_tokens[1:]:
                found: tuple[int, int, int] | None = None
                while cursor < len(source_tokens):
                    token, token_start, token_end = source_tokens[cursor]
                    if token_start - matched[-1][1] > _COORDINATED_MAX_GAP_CHARS:
                        break
                    if token == phrase_token:
                        found = (cursor, token_start, token_end)
                        break
                    cursor += 1
                if found is None:
                    break
                cursor, token_start, token_end = found
                matched.append((token_start, token_end))
                cursor += 1
            if len(matched) != len(phrase_tokens):
                continue
            between = note[matched[0][0] : matched[-1][1]]
            if re.search(r"[\n.!?]", between):
                continue
            segments: list[tuple[int, int]] = []
            for token_start, token_end in matched:
                if segments and not note[segments[-1][1] : token_start].strip(" \t-/"):
                    segments[-1] = (segments[-1][0], token_end)
                else:
                    segments.append((token_start, token_end))
            if len(segments) > 1:
                return segments
    return None


def build_context_packet(
    note: str,
    mention: Mention,
    assertion: AssertionDecision,
    candidates: list[Candidate],
    *,
    sentences: Sequence[SentenceSpan] | None = None,
) -> ContextPacket:
    """Build bounded, deterministic context without another model call."""

    parsed_sentences = sentences if sentences is not None else sentence_spans(note)
    sentence_index = next(
        (
            index
            for index, sentence in enumerate(parsed_sentences)
            if _overlaps(sentence.start, sentence.end, mention.start, mention.end)
        ),
        None,
    )
    reason: str | None = None
    close_candidates = False
    if len(candidates) >= 2:
        first, second = candidates[:2]
        close_candidates = (
            first.lexical_score is not None
            and second.lexical_score is not None
            and abs(first.lexical_score - second.lexical_score) <= 3.0
        ) or (
            first.dense_score is not None
            and second.dense_score is not None
            and abs(first.dense_score - second.dense_score) <= 0.02
        )
    if _CONTEXT_CUE.search(assertion.sentence):
        reason = "context-cue"
    elif len(normalize_phrase(mention.phrase).split()) <= 2 and len(candidates) > 1:
        reason = "short-ambiguous-phrase"
    elif close_candidates:
        reason = "close-candidate-scores"
    elif assertion.status != "affirmed":
        reason = "assertion-disambiguation"

    preceding: str | None = None
    following: str | None = None
    if reason is not None and sentence_index is not None:
        if sentence_index > 0:
            preceding = parsed_sentences[sentence_index - 1].text[-_MAX_ADJACENT_CONTEXT_CHARS:]
        if sentence_index + 1 < len(parsed_sentences):
            following = parsed_sentences[sentence_index + 1].text[:_MAX_ADJACENT_CONTEXT_CHARS]

    return ContextPacket(
        section=assertion.section,
        subject=("relative" if assertion.status == "family_history" else "patient"),
        assertion=assertion.status,
        sentence=assertion.sentence,
        preceding_sentence=preceding,
        following_sentence=following,
        extended_context_reason=reason,
    )


class StagedAnnotationPipeline:
    """Recall-expanded annotation with provider-neutral batched mapping."""

    @classmethod
    def for_final_category_repair(
        cls,
        *,
        provider: StagedProvider,
        output_dir: Path,
        keep_raw_responses: bool = False,
    ) -> StagedAnnotationPipeline:
        """Create a lightweight final-stage repair worker without loading vectors."""

        instance = cls.__new__(cls)
        instance.provider = provider
        instance.output_dir = output_dir
        instance.keep_raw_responses = keep_raw_responses
        instance.prompts = load_prompts()
        instance._active_provider_cache = {}
        instance._cache_stats = {
            "provider_response_hits": 0,
            "provider_response_misses": 0,
        }
        return instance

    def __init__(
        self,
        *,
        provider: StagedProvider | None,
        vector_dir: Path,
        output_dir: Path,
        mode: AnnotationMode,
        recognizers: set[str],
        fasthpocr_index: Path | None,
        resume: bool,
        keep_state: bool,
        keep_raw_responses: bool,
        include_evidence_text: bool,
        offline: bool,
        confidence_calibration: Path | None = None,
        backend: EmbeddingBackend | None = None,
        mapping_prompt: MappingPromptMode = MappingPromptMode.ZERO_SHOT,
    ) -> None:
        if mode is AnnotationMode.MODEL:
            raise ValueError("model mode uses the compatibility pipeline")
        self.provider = provider
        self.vector_dir = vector_dir
        self.output_dir = output_dir
        self.mode = mode
        self.resume = resume
        self.keep_state = keep_state
        self.keep_raw_responses = keep_raw_responses
        self.include_evidence_text = include_evidence_text
        self.mapping_prompt = mapping_prompt
        self.model_enabled = provider is not None and mode in {
            AnnotationMode.BALANCED,
            AnnotationMode.HIGH_RECALL,
        }
        if self.model_enabled:
            self.entries, matrix, self.manifest = load_artifacts(vector_dir)
        else:
            manifest_path = vector_dir / MANIFEST_NAME
            self.manifest = ArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if self.manifest.schema_version != ARTIFACT_SCHEMA_VERSION:
                raise ValueError(f"unsupported artifact schema: {self.manifest.schema_version}")
            self.entries = []
            matrix = None
        self.registry, self.registry_manifest, _ = load_registry_bundle(vector_dir)
        if (
            self.manifest.registry_sha256
            and self.manifest.registry_sha256 != self.registry_manifest.registry_sha256
        ):
            raise ValueError("registry identity differs from the vector manifest")
        self.backend: EmbeddingBackend | None = None
        self.retriever: HybridCandidateRetriever | None = None
        if self.model_enabled:
            if matrix is None:
                raise RuntimeError("model mode requires vector artifacts")
            self.backend = backend or create_backend(
                self.manifest.embedding_backend,
                offline=offline,
            )
            self._validate_backend(self.manifest, self.backend)
            self.retriever = HybridCandidateRetriever(
                registry=self.registry,
                entries=self.entries,
                matrix=matrix,
                backend=self.backend,
            )
        self.native = NativeLexicalRecognizer(self.registry)
        use_fast = "fasthpocr" in recognizers or mode is AnnotationMode.FASTHPOCR
        self.fast = (
            FastHPORecognizer(
                fasthpocr_index or vector_dir / "hp.index",
                longest_match=True,
                registry=self.registry,
            )
            if use_fast
            else None
        )
        self.prompts = load_prompts()
        self.calibration = (
            ConfidenceCalibration.load(
                confidence_calibration,
                prompt_bundle_sha256=sha256_file(
                    Path(__file__).parent / "data" / "system_prompts.json"
                ),
                artifact_manifest_sha256=sha256_file(self.vector_dir / "hpo_manifest.json"),
            )
            if confidence_calibration is not None
            else None
        )
        self._concepts = {
            concept.hp_id: concept for concept in self.registry.concepts if not concept.obsolete
        }
        self.term_registry = HPOTermRegistry.from_hpo_registry(self.registry)
        self.lexical_rescue = LexicalRescueEngine()
        self._modifier_ids = self._descendants_of(_CLINICAL_MODIFIER_ROOT)
        self.distinct_limit = 32 if mode is AnnotationMode.HIGH_RECALL else 16
        self._active_provider_cache: dict[str, tuple[Any, str]] | None = None
        self._cache_stats = {
            "provider_response_hits": 0,
            "provider_response_misses": 0,
        }

    def _descendants_of(self, root_id: str) -> set[str]:
        children: dict[str, set[str]] = {}
        for concept in self._concepts.values():
            for parent_id in concept.parents:
                children.setdefault(parent_id, set()).add(concept.hp_id)
        descendants = {root_id} if root_id in self._concepts else set()
        pending = list(descendants)
        while pending:
            parent_id = pending.pop()
            for child_id in children.get(parent_id, set()):
                if child_id not in descendants:
                    descendants.add(child_id)
                    pending.append(child_id)
        return descendants

    def run(
        self,
        rows: list[AnnotationInput],
        *,
        initial_errors: list[AnnotationResult] | None = None,
        max_attempts: int = 1,
    ) -> list[AnnotationResult]:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        ensure_private_directory(self.output_dir)
        state_path = self.output_dir / ".rag-hpo-state.sqlite3"
        config_hash = hashlib.sha256(
            json.dumps(
                {
                    "manifest": sha256_file(self.vector_dir / "hpo_manifest.json"),
                    "pipeline_schema_version": STAGED_PIPELINE_SCHEMA_VERSION,
                    "mode": self.mode.value,
                    "distinct_limit": self.distinct_limit,
                    "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
                    "model_enabled": self.model_enabled,
                    "mapping_prompt": self.mapping_prompt.value,
                    "prompts": sha256_file(Path(__file__).parent / "data" / "system_prompts.json"),
                    "calibration": (
                        self.calibration.identity if self.calibration is not None else None
                    ),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        all_results: list[AnnotationResult] = []
        retry_caches: dict[int, dict[str, tuple[Any, str]]] = {}
        completed_successfully = False
        try:
            with PipelineState(
                state_path,
                input_sha256=hash_inputs(rows),
                artifact_sha256=config_hash,
                pipeline_version=__version__,
                resume=self.resume,
            ) as state:
                for attempt_index in range(max_attempts):
                    started = time.perf_counter()
                    usage_before = self._provider_usage()
                    all_results = list(initial_errors or [])
                    row_failures = 0
                    for row_index, row in enumerate(rows):
                        note_hash = hashlib.sha256(row.clinical_note.encode("utf-8")).hexdigest()
                        cached = (
                            state.completed(row_index, note_hash)
                            if self.resume or attempt_index > 0
                            else None
                        )
                        if cached is not None:
                            all_results.extend(cached)
                            retry_caches.pop(row_index, None)
                            continue
                        self._active_provider_cache = retry_caches.setdefault(
                            row_index,
                            {},
                        )
                        try:
                            results = self._annotate_row(row, row_index)
                            state.save_complete(
                                row_index,
                                row.patient_id,
                                note_hash,
                                results,
                            )
                            all_results.extend(results)
                            retry_caches.pop(row_index, None)
                        except (ProviderError, ValueError, RuntimeError) as exc:
                            row_failures += 1
                            code = exc.code if isinstance(exc, ProviderError) else "row_failure"
                            message = _short_error(exc)
                            state.save_error(
                                row_index,
                                row.patient_id,
                                note_hash,
                                code,
                                message,
                            )
                            all_results.append(
                                AnnotationResult(
                                    patient_id=row.patient_id,
                                    phrase="",
                                    category=None,
                                    mapping_status="error",
                                    error_code=code,
                                    error_message=message,
                                    review_status="rejected",
                                )
                            )
                        finally:
                            self._active_provider_cache = None
                    export_results(all_results, self.output_dir)
                    elapsed_seconds = time.perf_counter() - started
                    usage_after = self._provider_usage()
                    attempt_usage = (
                        usage_after
                        if attempt_index == 0
                        else self._usage_delta(usage_after, usage_before)
                    )
                    self._write_run_manifest(
                        rows=rows,
                        results=all_results,
                        input_sha256=hash_inputs(rows),
                        config_sha256=config_hash,
                        elapsed_seconds=elapsed_seconds,
                        provider_usage=attempt_usage,
                        append_existing=self.resume or attempt_index > 0,
                    )
                    if row_failures == 0:
                        break
                completed_successfully = not any(
                    result.mapping_status == "error" for result in all_results
                )
        finally:
            self._active_provider_cache = None
            retry_caches.clear()
            if completed_successfully and not self.keep_state:
                state_path.unlink(missing_ok=True)
                Path(f"{state_path}-wal").unlink(missing_ok=True)
                Path(f"{state_path}-shm").unlink(missing_ok=True)
        return all_results

    def _write_run_manifest(
        self,
        *,
        rows: list[AnnotationInput],
        results: list[AnnotationResult],
        input_sha256: str,
        config_sha256: str,
        elapsed_seconds: float,
        provider_usage: dict[str, int],
        append_existing: bool,
    ) -> None:
        provider_config = getattr(self.provider, "config", None)
        path = self.output_dir / "rag_hpo_run_manifest.json"
        prior_attempts: list[dict[str, Any]] = []
        if append_existing and path.exists():
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior = {}
            if (
                prior.get("input_sha256") == input_sha256
                and prior.get("config_sha256") == config_sha256
            ):
                recorded = prior.get("attempts")
                if isinstance(recorded, list):
                    prior_attempts = [value for value in recorded if isinstance(value, dict)]
                elif isinstance(prior.get("provider_usage"), dict):
                    prior_attempts = [
                        {
                            "elapsed_seconds": prior.get("elapsed_seconds", 0),
                            "provider_usage": prior["provider_usage"],
                            "error_rows_after_attempt": prior.get("error_rows", 0),
                        }
                    ]
        current_attempt = {
            "elapsed_seconds": round(elapsed_seconds, 6),
            "provider_usage": provider_usage,
            "error_rows_after_attempt": sum(value.mapping_status == "error" for value in results),
        }
        attempts = [*prior_attempts, current_attempt]
        cumulative_usage = {
            key: sum(int(attempt.get("provider_usage", {}).get(key, 0)) for attempt in attempts)
            for key in ("requests", "input_tokens", "output_tokens", "total_tokens")
        }
        manifest = {
            "schema_version": "1.0",
            "rag_hpo_version": __version__,
            "pipeline_schema_version": STAGED_PIPELINE_SCHEMA_VERSION,
            "mode": self.mode.value,
            "mapping_prompt": self.mapping_prompt.value,
            "distinct_candidate_limit": self.distinct_limit,
            "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
            "input_rows": len(rows),
            "input_sha256": input_sha256,
            "config_sha256": config_sha256,
            "artifact_manifest_sha256": sha256_file(self.vector_dir / "hpo_manifest.json"),
            "registry_sha256": self.registry_manifest.registry_sha256,
            "prompt_bundle_sha256": sha256_file(
                Path(__file__).parent / "data" / "system_prompts.json"
            ),
            "confidence_calibration": (
                {
                    "identity": self.calibration.identity,
                    "minimum_precision": self.calibration.minimum_precision,
                }
                if self.calibration is not None
                else None
            ),
            "provider": (
                provider_config.redacted()
                if provider_config is not None
                else {"model": "none", "base_url": "none"}
            ),
            "runtime": {
                "python": sys.version.split()[0],
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "inference_parameters": {
                "coverage_extraction_temperature": 0.2,
                "coverage_audit_temperature": 0.0,
                "mapping_temperature": 0.0,
                "mapping_batch_size": MAPPING_BATCH_SIZE,
                "mapping_prompt": self.mapping_prompt.value,
                "final_categorization_temperature": 0.0,
                "final_categorization_batch_size": FINAL_CATEGORY_BATCH_SIZE,
                "final_category_values": [value.value for value in Category],
                "maximum_mapping_alternatives": 3,
                "deterministic_candidate_order": True,
                "provider_seed": "not-configured",
                "quantization": "not-recorded",
                "embedding_backend": self.manifest.embedding_backend,
                "embedding_model": self.manifest.embedding_model,
                "embedding_revision": self.manifest.embedding_revision,
            },
            "cache": {
                **self._cache_stats,
                **(
                    self.retriever.cache_info()
                    if self.retriever is not None
                    else {
                        "embedding_hits": 0,
                        "embedding_misses": 0,
                        "embedding_entries": 0,
                        "candidate_hits": 0,
                        "candidate_misses": 0,
                        "candidate_entries": 0,
                    }
                ),
            },
            "provider_usage": cumulative_usage,
            "latest_attempt_provider_usage": provider_usage,
            "elapsed_seconds": round(
                sum(float(attempt.get("elapsed_seconds", 0)) for attempt in attempts),
                6,
            ),
            "latest_attempt_elapsed_seconds": round(elapsed_seconds, 6),
            "run_attempt_count": len(attempts),
            "attempts": attempts,
            "result_rows": len(results),
            "accepted_rows": sum(
                value.mapping_status == "mapped" and value.review_status == "accepted"
                for value in results
            ),
            "review_rows": sum(value.review_status == "review" for value in results),
            "rejected_rows": sum(value.review_status == "rejected" for value in results),
            "error_rows": sum(value.mapping_status == "error" for value in results),
        }
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        restrict_owner(path)

    def _annotate_row(
        self,
        row: AnnotationInput,
        row_index: int,
    ) -> list[AnnotationResult]:
        note = row.clinical_note
        note_context = NoteContext.build(note)
        mentions = self._recognizer_mentions(note)
        if self.model_enabled:
            first_mentions = self._extract_with_strict_chunk_fallback(
                note=note,
                row_index=row_index,
                system_message=self.prompts["coverage_extraction"],
                stage="coverage-extract",
                method="model-pass-1",
                temperature=0.2,
                sentences=note_context.sentences,
            )
            mentions.extend(first_mentions)
            targets = self._coverage_targets(
                note,
                mentions,
                first_mentions,
                sentences=note_context.sentences,
            )
            if targets:
                payload = json.dumps(
                    {
                        "source_length": len(note),
                        "sentences": [
                            {
                                "sentence_id": value.sentence_id,
                                "start_offset": value.start,
                                "text": value.text,
                            }
                            for value in targets
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                second_mentions = self._extract_with_strict_chunk_fallback(
                    note=note,
                    row_index=row_index,
                    system_message=self.prompts["coverage_audit"],
                    stage="coverage-audit",
                    method="model-pass-2",
                    temperature=0.0,
                    primary_payload=payload,
                    allowed_ranges=[(value.start, value.end) for value in targets],
                    sentences=note_context.sentences,
                )
                mentions.extend(second_mentions)
        merged = self._merge_mentions(mentions)
        return apply_calibration(
            self._map_mentions(row, row_index, merged, note_context),
            self.calibration,
        )

    def _extract_with_strict_chunk_fallback(
        self,
        *,
        note: str,
        row_index: int,
        system_message: str,
        stage: str,
        method: str,
        temperature: float,
        primary_payload: str | None = None,
        allowed_ranges: list[tuple[int, int]] | None = None,
        sentences: Sequence[SentenceSpan] | None = None,
    ) -> list[Mention]:
        try:
            extraction, raw = self.provider_request(
                system_message=system_message,
                user_message=primary_payload or note,
                response_model=PhenotypeSpanExtraction,
                temperature=temperature,
            )
            self._write_raw(row_index, stage, raw)
            return self._validated_extraction(
                note,
                extraction,
                method,
                allowed_ranges=allowed_ranges,
            )
        except ProviderError as exc:
            if exc.status_code != 400:
                raise

        fallback_ranges = allowed_ranges or [
            (value.start, value.end)
            for value in (sentences if sentences is not None else sentence_spans(note))
        ]
        values: list[Mention] = []
        for chunk_index, (start, end) in enumerate(fallback_ranges):
            extraction, raw = self.provider_request(
                system_message=system_message,
                user_message=note[start:end],
                response_model=PhenotypeSpanExtraction,
                temperature=temperature,
            )
            self._write_raw(row_index, f"{stage}-chunk-{chunk_index:03d}", raw)
            values.extend(
                self._validated_extraction(
                    note,
                    extraction,
                    method,
                    allowed_ranges=[(start, end)],
                )
            )
        return values

    def _recognizer_mentions(self, note: str) -> list[Mention]:
        mentions = [
            Mention(
                phrase=value.phrase,
                start=value.start_offset,
                end=value.end_offset,
                methods={"native"},
                recognizer_ids=set(value.candidate_hpo_ids),
                evidence_segments=[(value.start_offset, value.end_offset)],
                phrase_variants={value.phrase},
            )
            for value in self.native.recognize(note)
        ]
        if self.fast is not None:
            mentions.extend(
                Mention(
                    phrase=value.phrase,
                    start=value.start_offset,
                    end=value.end_offset,
                    methods={"fasthpocr"},
                    recognizer_ids=set(
                        value.candidate_hpo_ids or ((value.hpo_id,) if value.hpo_id else ())
                    ),
                    evidence_segments=[(value.start_offset, value.end_offset)],
                    phrase_variants={value.phrase},
                )
                for value in self.fast.annotate(note)
            )
        return mentions

    @staticmethod
    def _validated_extraction(
        note: str,
        extraction: PhenotypeSpanExtraction,
        method: str,
        *,
        allowed_ranges: list[tuple[int, int]] | None = None,
    ) -> list[Mention]:
        values: list[Mention] = []
        used: set[tuple[int, int]] = set()
        for phenotype in extraction.phenotypes:
            matches = [
                match
                for match in re.finditer(
                    re.escape(phenotype.phrase),
                    note,
                    flags=re.IGNORECASE,
                )
                if allowed_ranges is None
                or any(
                    match.start() >= range_start and match.end() <= range_end
                    for range_start, range_end in allowed_ranges
                )
            ]
            if not matches:
                search_ranges = allowed_ranges or [(0, len(note))]
                segments = _coordinated_alignment(
                    phenotype.phrase,
                    note,
                    search_ranges,
                )
                if segments is not None:
                    start = segments[0][0]
                    end = segments[-1][1]
                    values.append(
                        Mention(
                            phrase=phenotype.phrase,
                            start=start,
                            end=end,
                            methods={method},
                            evidence_segments=segments,
                            phrase_variants={phenotype.phrase},
                        )
                    )
                    continue
                alignments = []
                for range_start, range_end in search_ranges:
                    candidate_alignment = fuzz.partial_ratio_alignment(
                        phenotype.phrase,
                        note[range_start:range_end],
                    )
                    if candidate_alignment is not None:
                        alignments.append((candidate_alignment, range_start))
                if not alignments:
                    continue
                alignment, base = max(
                    alignments,
                    key=lambda value: value[0].score,
                )
                if alignment.score < 90:
                    continue
                start = base + alignment.dest_start
                end = base + alignment.dest_end
                used.add((start, end))
                values.append(
                    Mention(
                        phrase=note[start:end],
                        start=start,
                        end=end,
                        methods={method},
                        evidence_segments=[(start, end)],
                        phrase_variants={phenotype.phrase, note[start:end]},
                    )
                )
                continue
            match = next(
                (item for item in matches if (item.start(), item.end()) not in used),
                matches[0],
            )
            start, end = match.start(), match.end()
            used.add((start, end))
            values.append(
                Mention(
                    phrase=note[start:end],
                    start=start,
                    end=end,
                    methods={method},
                    evidence_segments=[(start, end)],
                    phrase_variants={phenotype.phrase, note[start:end]},
                )
            )
        return values

    @staticmethod
    def _coverage_targets(
        note: str,
        all_mentions: list[Mention],
        first_mentions: list[Mention],
        *,
        sentences: Sequence[SentenceSpan] | None = None,
    ) -> list[SentenceSpan]:
        targets: list[SentenceSpan] = []
        for sentence in sentences if sentences is not None else sentence_spans(note):
            sentence_mentions = [
                value
                for value in all_mentions
                if _overlaps(sentence.start, sentence.end, value.start, value.end)
            ]
            first_in_sentence = [
                value
                for value in first_mentions
                if _overlaps(sentence.start, sentence.end, value.start, value.end)
            ]
            unmatched_recognizer = any(
                value.methods <= {"native", "fasthpocr"}
                and not any(
                    _overlaps(
                        value.start,
                        value.end,
                        first.start,
                        first.end,
                    )
                    for first in first_mentions
                )
                for value in sentence_mentions
            )
            list_gap = bool(first_in_sentence) and bool(_LIST_CUE.search(sentence.text))
            measurement_gap = bool(_MEASUREMENT_CUE.search(sentence.text)) and not bool(
                first_in_sentence
            )
            structural_gap = bool(_STRUCTURAL_CUE.search(sentence.text)) and not bool(
                first_in_sentence
            )
            if unmatched_recognizer or list_gap or measurement_gap or structural_gap:
                targets.append(sentence)
        return targets

    @staticmethod
    def _merge_mentions(mentions: list[Mention]) -> list[Mention]:
        merged: dict[tuple[int, int, str], Mention] = {}
        for mention in mentions:
            norm = normalize_phrase(mention.phrase)
            if norm in SINGLE_TOKEN_MODIFIER_BLOCKLIST:
                continue
            key = (
                mention.start,
                mention.end,
                norm,
            )
            existing = merged.get(key)
            if existing is None:
                if not mention.evidence_segments:
                    mention.evidence_segments = [(mention.start, mention.end)]
                if not mention.phrase_variants:
                    mention.phrase_variants = {mention.phrase}
                merged[key] = mention
            else:
                existing.methods.update(mention.methods)
                existing.recognizer_ids.update(mention.recognizer_ids)
                existing.modifier_ids.update(mention.modifier_ids)
                existing.phrase_variants.update(mention.phrase_variants or {mention.phrase})
                for segment in _mention_segments(mention):
                    if segment not in existing.evidence_segments:
                        existing.evidence_segments.append(segment)
        values = list(merged.values())
        for m in values:
            words = m.phrase.split()
            if len(words) > 1 and normalize_phrase(words[0]) in SINGLE_TOKEN_MODIFIER_BLOCKLIST:
                clean_words = [w for w in words if normalize_phrase(w) not in SINGLE_TOKEN_MODIFIER_BLOCKLIST]
                if clean_words:
                    head = " ".join(clean_words).strip()
                    if head and head != m.phrase and len(head) >= 3 and normalize_phrase(head) not in SINGLE_TOKEN_MODIFIER_BLOCKLIST:
                        m.phrase_variants.add(head)

        suppressed: set[int] = set()
        for broad_index, broad in enumerate(values):
            nested = [
                (narrow_index, narrow)
                for narrow_index, narrow in enumerate(values)
                if narrow_index != broad_index
                and broad.start <= narrow.start
                and broad.end >= narrow.end
                and (broad.start, broad.end) != (narrow.start, narrow.end)
                and normalize_phrase(narrow.phrase) in normalize_phrase(broad.phrase)
            ]
            # One nested phenotype is the same mention with added context
            # ("possible scoliosis" versus "scoliosis"). Multiple nested spans
            # can be a coordinated list and must remain separate.
            if len(nested) == 1:
                narrow_index, narrow = nested[0]
                broad.methods.update(narrow.methods)
                broad.recognizer_ids.update(narrow.recognizer_ids)
                broad.modifier_ids.update(narrow.modifier_ids)
                broad.phrase_variants.update(narrow.phrase_variants or {narrow.phrase})
                for segment in _mention_segments(narrow):
                    if segment not in broad.evidence_segments:
                        broad.evidence_segments.append(segment)
                suppressed.add(narrow_index)
        return sorted(
            (value for index, value in enumerate(values) if index not in suppressed),
            key=lambda value: (
                value.start,
                value.end,
                normalize_phrase(value.phrase),
            ),
        )

    def _map_mentions(
        self,
        row: AnnotationInput,
        row_index: int,
        mentions: list[Mention],
        note_context: NoteContext,
    ) -> list[AnnotationResult]:
        results: list[AnnotationResult] = []
        abnormal: list[tuple[str, Mention, AssertionDecision, list[Candidate]]] = []
        pending: list[tuple[str, Mention, AssertionDecision]] = []
        for index, mention in enumerate(mentions):
            assertion = analyze_assertion(
                row.clinical_note,
                mention.start,
                mention.end,
            )
            routing_category = self._routing_category(mention, assertion)
            if routing_category is not Category.ABNORMAL:
                mention.category = routing_category
                results.append(
                    self._result(
                        row,
                        mention,
                        assertion,
                        mapping_status="not_mapped_category",
                        review_status="accepted",
                        category_confidence="high",
                    )
                )
                continue
            pending.append((f"m{index:04d}", mention, assertion))

        retrieved_batches = (
            self.retriever.retrieve_many(
                [mention.phrase for _mention_id, mention, _assertion in pending],
                distinct_limit=self.distinct_limit,
                contexts=[
                    self._retrieval_context(note_context, mention)
                    for _mention_id, mention, _assertion in pending
                ],
            )
            if self.retriever is not None
            else [[] for _value in pending]
        )
        for (
            (mention_id, mention, assertion),
            retrieved,
        ) in zip(pending, retrieved_batches, strict=True):
            recognized_modifiers = mention.recognizer_ids & self._modifier_ids
            mention.modifier_ids.update(recognized_modifiers)
            mention.recognizer_ids.difference_update(recognized_modifiers)
            retrieved = [
                candidate for candidate in retrieved if candidate.hpo_id not in self._modifier_ids
            ]
            candidates = self._inject_recognizer_candidates(
                retrieved,
                mention.recognizer_ids,
            )
            if self.model_enabled and not candidates:
                results.append(
                    self._result(
                        row,
                        mention,
                        assertion,
                        candidate_ids=[],
                        retrieval_candidate_ids=[],
                        mapping_status="no_candidate_fit",
                        confidence="low",
                        review_status="review",
                    )
                )
                continue
            abnormal.append((mention_id, mention, assertion, candidates))

        if not abnormal:
            return (
                self._finalize_categories(row, row_index, self._deduplicate(results))
                if self.model_enabled
                else self._deduplicate(results)
            )
        if not self.model_enabled:
            for _mention_id, mention, assertion, candidates in abnormal:
                mention.category = Category.ABNORMAL
                candidate_ids = sorted(mention.recognizer_ids)[:3]
                selected = (
                    next(
                        (value for value in candidates if value.hpo_id == candidate_ids[0]),
                        None,
                    )
                    if len(candidate_ids) == 1
                    else None
                )
                results.append(
                    self._result(
                        row,
                        mention,
                        assertion,
                        candidate_ids=candidate_ids,
                        retrieval_candidate_ids=[value.hpo_id for value in candidates],
                        selected=selected,
                        mapping_status="mapped" if candidate_ids else "no_candidate_fit",
                        mapping_verdict=(
                            "supported"
                            if len(candidate_ids) == 1
                            else ("ambiguous" if candidate_ids else "unsupported")
                        ),
                        confidence="high" if selected else "low",
                        category_confidence="high",
                        review_status=(
                            "accepted" if selected and assertion.status == "affirmed" else "review"
                        ),
                    )
                )
            return self._deduplicate(results)

        batches = [
            (batch_index, abnormal[start : start + MAPPING_BATCH_SIZE])
            for batch_index, start in enumerate(range(0, len(abnormal), MAPPING_BATCH_SIZE))
        ]
        decisions: dict[str, Any] = {}
        if len(batches) == 1:
            batch_index, items = batches[0]
            decisions = self._request_mapping_items(
                items,
                note=row.clinical_note,
                sentences=note_context.sentences,
                row_index=row_index,
                stage=f"batch-map-{batch_index:03d}",
            )
        elif batches:
            with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
                future_to_batch = {
                    executor.submit(
                        self._request_mapping_items,
                        items,
                        note=row.clinical_note,
                        sentences=note_context.sentences,
                        row_index=row_index,
                        stage=f"batch-map-{batch_index:03d}",
                    ): batch_index
                    for batch_index, items in batches
                }
                for future in as_completed(future_to_batch):
                    batch_decisions = future.result()
                    if set(decisions) & set(batch_decisions):
                        raise ValueError("batched mapper returned duplicate decisions")
                    decisions.update(batch_decisions)
        for mention_id, mention, assertion, candidates in abnormal:
            decision = decisions.get(mention_id)
            if decision is None:
                results.append(
                    self._result(
                        row,
                        mention,
                        assertion,
                        candidate_ids=[],
                        retrieval_candidate_ids=[value.hpo_id for value in candidates],
                        mapping_status="no_candidate_fit",
                        confidence="low",
                        mapping_verdict="unsupported",
                        review_status="review",
                        error_code="incomplete_mapping_decision",
                        error_message=(
                            "The mapper omitted or duplicated this finding; no HPO ID was accepted."
                        ),
                    )
                )
                continue
            by_id = {value.hpo_id: value for value in candidates}
            invalid_ids = [hpo_id for hpo_id in decision.candidate_hpo_ids if hpo_id not in by_id]
            candidate_ids = [hpo_id for hpo_id in decision.candidate_hpo_ids if hpo_id in by_id]
            if invalid_ids:
                mapping_verdict = (
                    "supported"
                    if len(candidate_ids) == 1
                    else ("ambiguous" if candidate_ids else "unsupported")
                )
                selected = by_id[candidate_ids[0]] if len(candidate_ids) == 1 else None
                results.append(
                    self._result(
                        row,
                        mention,
                        assertion,
                        candidate_ids=candidate_ids,
                        retrieval_candidate_ids=[value.hpo_id for value in candidates],
                        selected=selected,
                        mapping_status="mapped" if candidate_ids else "no_candidate_fit",
                        confidence="low",
                        mapping_verdict=mapping_verdict,
                        review_status="review",
                        error_code="mapping_candidate_out_of_scope",
                        error_message=(
                            "The mapper returned an HPO ID outside the supplied candidate "
                            "set; out-of-scope IDs were discarded."
                        ),
                    )
                )
                continue
            selected = by_id[candidate_ids[0]] if len(candidate_ids) == 1 else None
            consensus = bool(
                selected
                and selected.hpo_id in mention.recognizer_ids
                and decision.verdict == "supported"
            )
            base_finding = "model-pass-1" in mention.methods
            if consensus:
                review_status = "accepted"
                confidence = "high"
            elif (
                base_finding
                and candidate_ids
                and decision.verdict in {"supported", "ambiguous"}
                and decision.confidence in {"high", "medium"}
            ):
                review_status = "accepted"
                confidence = decision.confidence
            elif (
                decision.verdict == "supported"
                and decision.confidence in {"high", "medium"}
                and candidate_ids
            ):
                review_status = "accepted"
                confidence = decision.confidence
            elif decision.verdict == "unsupported" and decision.confidence == "high":
                review_status = "rejected"
                confidence = "high"
                selected = None
                candidate_ids = []
            else:
                review_status = "review"
                confidence = decision.confidence
            if assertion.status != "affirmed" and review_status == "accepted":
                # Assertion rules triage but never delete a finding by themselves.
                review_status = "review"
            results.append(
                self._result(
                    row,
                    mention,
                    assertion,
                    candidate_ids=candidate_ids,
                    retrieval_candidate_ids=[value.hpo_id for value in candidates],
                    selected=selected,
                    mapping_status="mapped" if candidate_ids else "no_candidate_fit",
                    mapping_verdict=decision.verdict,
                    confidence=confidence,
                    review_status=review_status,
                )
            )
        return self._finalize_categories(
            row,
            row_index,
            self._deduplicate(results),
        )

    @staticmethod
    def _retrieval_context(
        note_context: NoteContext,
        mention: Mention,
    ) -> str | None:
        """Add bounded local anatomy/context to short phrases without replacing them."""

        if len(normalize_phrase(mention.phrase).split()) > 3:
            return None
        sentences = note_context.sentences
        sentence_index = note_context.sentence_index(mention.start, mention.end)
        if sentence_index is None:
            return None
        sentence = sentences[sentence_index].text
        preceding = (
            sentences[sentence_index - 1].text[-_MAX_ADJACENT_CONTEXT_CHARS:]
            if sentence_index > 0
            else ""
        )
        return " ".join(
            value.strip() for value in (mention.phrase, preceding, sentence) if value.strip()
        )[: 2 * _MAX_ADJACENT_CONTEXT_CHARS]

    @staticmethod
    def _routing_category(
        mention: Mention,
        assertion: AssertionDecision,
    ) -> Category:
        if assertion.status == "family_history":
            return Category.FAMILY_HISTORY
        if assertion.status in {"normal", "negated"}:
            return Category.NORMAL
        if re.search(
            r"\b(?:normal|unremarkable|within normal limits|intact)\b",
            mention.phrase,
            re.I,
        ):
            return Category.NORMAL
        return Category.ABNORMAL

    def _finalize_categories(
        self,
        row: AnnotationInput,
        row_index: int,
        results: list[AnnotationResult],
    ) -> list[AnnotationResult]:
        targets = [
            (f"c{index:04d}", index, result)
            for index, result in enumerate(results)
            if result.category is None and result.phrase
        ]
        if not targets:
            return results
        decisions: dict[str, Any] = {}
        for batch_index, start in enumerate(range(0, len(targets), FINAL_CATEGORY_BATCH_SIZE)):
            items = targets[start : start + FINAL_CATEGORY_BATCH_SIZE]
            batch_decisions = self._request_final_category_items(
                items,
                note=row.clinical_note,
                row_index=row_index,
                stage=f"final-categorization-{batch_index:03d}",
            )
            if set(decisions) & set(batch_decisions):
                raise ValueError("final categorizer returned duplicate decisions")
            decisions.update(batch_decisions)
        updated = list(results)
        for mention_id, position, result in targets:
            decision = decisions.get(mention_id)
            if decision is None:
                updated[position] = result.model_copy(
                    update={
                        "category": Category.ABNORMAL,
                        "category_confidence": "low",
                        "review_status": "review",
                        "error_code": "incomplete_final_category",
                        "error_message": (
                            "The final categorizer omitted or duplicated this finding; "
                            "it was retained for review."
                        ),
                    }
                )
                continue
            changes: dict[str, object] = {
                "category": decision.category,
                "category_confidence": decision.confidence,
            }
            if decision.category is not Category.ABNORMAL:
                changes.update(
                    {
                        "hpo_id": None,
                        "hpo_term": None,
                        "vector_score": None,
                        "mapping_status": "not_mapped_category",
                        "assertion_status": (
                            "normal" if decision.category is Category.NORMAL else "family_history"
                        ),
                        "review_status": (
                            "accepted" if decision.confidence in {"high", "medium"} else "review"
                        ),
                    }
                )
            elif decision.confidence == "low" and result.review_status == "accepted":
                changes["review_status"] = "review"
            updated[position] = result.model_copy(update=changes)
        return updated

    def repair_final_categories(
        self,
        row: AnnotationInput,
        row_index: int,
        results: list[AnnotationResult],
    ) -> list[AnnotationResult]:
        """Recalculate only failed final-category decisions for a completed row."""

        repaired_inputs: list[AnnotationResult] = []
        for result in results:
            if result.error_code != "incomplete_final_category":
                repaired_inputs.append(result)
                continue
            repaired_inputs.append(
                result.model_copy(
                    update={
                        "category": None,
                        "category_confidence": None,
                        "review_status": self._pre_final_review_status(result),
                        "error_code": None,
                        "error_message": None,
                    }
                )
            )
        return self._finalize_categories(row, row_index, repaired_inputs)

    @staticmethod
    def _pre_final_review_status(
        result: AnnotationResult,
    ) -> str:
        """Reconstruct the review state immediately before final categorization."""

        if result.assertion_status != "affirmed":
            return "review"
        candidate_ids = result.candidate_hpo_ids or []
        if (
            result.mapping_status == "mapped"
            and candidate_ids
            and result.mapping_verdict in {"supported", "ambiguous"}
            and result.confidence in {"high", "medium"}
        ):
            return "accepted"
        if result.mapping_verdict == "unsupported" and result.confidence == "high":
            return "rejected"
        return "review"

    def _request_final_category_items(
        self,
        items: list[tuple[str, int, AnnotationResult]],
        *,
        note: str,
        row_index: int,
        stage: str,
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "note": note,
                "items": [
                    {
                        "mention_id": mention_id,
                        "phrase": result.phrase,
                        "start_offset": result.evidence_start,
                        "end_offset": result.evidence_end,
                        "assertion_hint": result.assertion_status,
                        "candidate_hpo_ids": result.candidate_hpo_ids or [],
                    }
                    for mention_id, _position, result in items
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            batch, raw = self.provider_request(
                system_message=self.prompts["final_categorization"],
                user_message=payload,
                response_model=FinalCategoryDecisionBatch,
                temperature=0.0,
            )
        except ProviderError as exc:
            if exc.status_code != 400:
                raise
            return self._split_final_category_items(
                items,
                note=note,
                row_index=row_index,
                stage=stage,
            )
        self._write_raw(row_index, stage, raw)
        expected = {mention_id for mention_id, _position, _result in items}
        counts: dict[str, int] = {}
        for decision in batch.decisions:
            counts[decision.mention_id] = counts.get(decision.mention_id, 0) + 1
        decisions = {
            decision.mention_id: decision
            for decision in batch.decisions
            if decision.mention_id in expected and counts[decision.mention_id] == 1
        }
        missing = [item for item in items if item[0] not in decisions]
        if missing:
            recovered = (
                self._split_final_category_items(
                    missing,
                    note=note,
                    row_index=row_index,
                    stage=f"{stage}-missing",
                )
                if len(missing) == len(items)
                else self._request_final_category_items(
                    missing,
                    note=note,
                    row_index=row_index,
                    stage=f"{stage}-missing",
                )
            )
            decisions.update(recovered)
        return decisions

    def _split_final_category_items(
        self,
        items: list[tuple[str, int, AnnotationResult]],
        *,
        note: str,
        row_index: int,
        stage: str,
    ) -> dict[str, Any]:
        if len(items) <= 1:
            return {}
        midpoint = len(items) // 2
        left = self._request_final_category_items(
            items[:midpoint],
            note=note,
            row_index=row_index,
            stage=f"{stage}-a",
        )
        right = self._request_final_category_items(
            items[midpoint:],
            note=note,
            row_index=row_index,
            stage=f"{stage}-b",
        )
        if set(left) & set(right):
            raise ValueError("final categorizer returned duplicate decisions")
        return {**left, **right}

    def _request_mapping_items(
        self,
        items: list[tuple[str, Mention, AssertionDecision, list[Candidate]]],
        *,
        note: str,
        sentences: Sequence[SentenceSpan],
        row_index: int,
        stage: str,
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "items": [
                    {
                        "mention_id": mention_id,
                        "phrase": mention.phrase,
                        "context": {
                            key: value
                            for key, value in build_context_packet(
                                note,
                                mention,
                                assertion,
                                candidates,
                                sentences=sentences,
                            ).__dict__.items()
                            if value is not None
                        },
                        "source_methods": sorted(mention.methods),
                        "recognizer_hpo_ids": sorted(mention.recognizer_ids),
                        "candidates": [
                            self._candidate_payload(candidate, rank=index + 1)
                            for index, candidate in enumerate(candidates)
                        ],
                    }
                    for mention_id, mention, assertion, candidates in items
                ]
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            batch, raw = self.provider_request(
                system_message=self.prompts[
                    (
                        "context_mapping_one_shot"
                        if self.mapping_prompt is MappingPromptMode.ONE_SHOT
                        else "context_mapping_zero_shot"
                    )
                ],
                user_message=payload,
                response_model=MappingSetDecisionBatch,
                temperature=0.0,
            )
        except ProviderError as exc:
            if exc.status_code != 400:
                raise
            if len(items) == 1:
                return {}
            midpoint = len(items) // 2
            left = self._request_mapping_items(
                items[:midpoint],
                note=note,
                sentences=sentences,
                row_index=row_index,
                stage=f"{stage}-a",
            )
            right = self._request_mapping_items(
                items[midpoint:],
                note=note,
                sentences=sentences,
                row_index=row_index,
                stage=f"{stage}-b",
            )
            if set(left) & set(right):
                raise ValueError("batched mapper returned duplicate decisions") from exc
            return {**left, **right}
        self._write_raw(row_index, stage, raw)
        expected = {value[0] for value in items}
        counts: dict[str, int] = {}
        for value in batch.decisions:
            counts[value.mention_id] = counts.get(value.mention_id, 0) + 1
        return {
            value.mention_id: value
            for value in batch.decisions
            if value.mention_id in expected and counts[value.mention_id] == 1
        }

    @staticmethod
    def _candidate_payload(
        candidate: Candidate,
        *,
        rank: int,
    ) -> dict[str, object]:
        """Keep batched mapping bounded while preserving auditable rank evidence."""

        return {
            "hpo_id": candidate.hpo_id,
            "term": candidate.term,
            "definition": ((candidate.definition or "")[:160] if rank <= 4 else ""),
            "synonyms": (candidate.synonyms or [])[:3] if rank <= 4 else [],
            "parents": (candidate.parents or [])[:2] if rank <= 4 else [],
            "dense_rank": candidate.dense_rank,
            "dense_score": candidate.dense_score,
            "lexical_rank": candidate.lexical_rank,
            "lexical_score": candidate.lexical_score,
            "context_dense_rank": candidate.context_dense_rank,
            "context_dense_score": candidate.context_dense_score,
            "source_methods": candidate.source_methods or [],
        }

    def _inject_recognizer_candidates(
        self,
        candidates: list[Candidate],
        recognizer_ids: set[str],
    ) -> list[Candidate]:
        by_id = {value.hpo_id: value for value in candidates}
        injected: list[Candidate] = []
        for hp_id in sorted(recognizer_ids):
            if hp_id in self._modifier_ids:
                continue
            concept = self._concepts.get(hp_id)
            if concept is None or hp_id in by_id:
                continue
            injected.append(
                Candidate(
                    hpo_id=hp_id,
                    term=concept.label,
                    score=0.0,
                    definition=concept.definition[:500] or None,
                    synonyms=sorted({value.phrase for value in concept.phrases})[:12],
                    parents=concept.parents,
                    source_methods=["recognizer"],
                )
            )
        return (injected + candidates)[: self.distinct_limit]

    def _result(
        self,
        row: AnnotationInput,
        mention: Mention,
        assertion: AssertionDecision,
        *,
        mapping_status: str,
        review_status: str,
        candidate_ids: list[str] | None = None,
        retrieval_candidate_ids: list[str] | None = None,
        selected: Candidate | None = None,
        confidence: str | None = None,
        category_confidence: str | None = None,
        mapping_verdict: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> AnnotationResult:
        return AnnotationResult.model_validate(
            {
                "patient_id": row.patient_id,
                "phrase": mention.phrase,
                "category": mention.category,
                "hpo_id": selected.hpo_id if selected else None,
                "hpo_term": selected.term if selected else None,
                "vector_score": selected.score if selected else None,
                "mapping_status": mapping_status,
                "error_code": error_code,
                "error_message": error_message,
                "evidence_start": mention.start,
                "evidence_end": mention.end,
                "assertion_status": assertion.status,
                "confidence": confidence,
                "category_confidence": category_confidence,
                "review_status": review_status,
                "source_methods": sorted(mention.methods),
                "candidate_hpo_ids": candidate_ids,
                "retrieval_candidate_hpo_ids": retrieval_candidate_ids,
                "modifier_hpo_ids": sorted(mention.modifier_ids),
                "evidence_segments": _mention_segments(mention),
                "mapping_verdict": mapping_verdict,
                "confidence_basis": "uncalibrated-evidence",
                "evidence_text": assertion.sentence if self.include_evidence_text else None,
            }
        )

    @staticmethod
    def _deduplicate(results: list[AnnotationResult]) -> list[AnnotationResult]:
        output: dict[tuple[str | None, int | None, int | None, str], AnnotationResult] = {}
        priority = {"accepted": 0, "review": 1, "rejected": 2, None: 3}
        for result in results:
            key = (
                result.hpo_id,
                result.evidence_start,
                result.evidence_end,
                result.phrase.casefold(),
            )
            existing = output.get(key)
            if (
                existing is None
                or priority[result.review_status] < priority[existing.review_status]
            ):
                output[key] = result
        return sorted(
            output.values(),
            key=lambda value: (
                value.evidence_start if value.evidence_start is not None else 10**9,
                value.evidence_end if value.evidence_end is not None else 10**9,
                value.hpo_id or "",
            ),
        )

    def provider_request(self, **kwargs: Any) -> tuple[Any, str]:
        if self.provider is None:
            raise RuntimeError("this annotation mode requires a model provider")
        cache = self._active_provider_cache
        if cache is None:
            return self.provider.request(**kwargs)
        response_model = kwargs.get("response_model")
        key_payload = {
            "system_message": kwargs.get("system_message"),
            "user_message": kwargs.get("user_message"),
            "temperature": kwargs.get("temperature", 0.2),
            "response_model": (
                f"{response_model.__module__}.{response_model.__qualname__}"
                if isinstance(response_model, type)
                else str(response_model)
            ),
        }
        cache_key = hashlib.sha256(
            json.dumps(
                key_payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cached = cache.get(cache_key)
        if cached is not None:
            self._cache_stats["provider_response_hits"] += 1
            value, raw = cached
            return (
                value.model_copy(deep=True) if isinstance(value, BaseModel) else value,
                raw,
            )
        self._cache_stats["provider_response_misses"] += 1
        value, raw = self.provider.request(**kwargs)
        cache[cache_key] = (
            value.model_copy(deep=True) if isinstance(value, BaseModel) else value,
            raw,
        )
        return value, raw

    def _provider_usage(self) -> dict[str, int]:
        usage = getattr(self.provider, "usage", None) or {}
        return {
            key: int(usage.get(key, 0))
            for key in ("requests", "input_tokens", "output_tokens", "total_tokens")
        }

    @staticmethod
    def _usage_delta(
        after: dict[str, int],
        before: dict[str, int],
    ) -> dict[str, int]:
        return {
            key: max(0, after.get(key, 0) - before.get(key, 0))
            for key in ("requests", "input_tokens", "output_tokens", "total_tokens")
        }

    def _write_raw(self, row_index: int, stage: str, content: str) -> None:
        if not self.keep_raw_responses:
            return
        directory = self.output_dir / "raw_responses"
        ensure_private_directory(directory)
        path = directory / f"row-{row_index:05d}-{stage}.json"
        path.write_text(content, encoding="utf-8")
        restrict_owner(path)

    @staticmethod
    def _validate_backend(
        manifest: ArtifactManifest,
        backend: EmbeddingBackend,
    ) -> None:
        if backend.name != manifest.embedding_backend:
            raise ValueError("embedding backend does not match the vector manifest")
        if backend.model_id != manifest.embedding_model:
            raise ValueError("embedding model does not match the vector manifest")
        if backend.revision != manifest.embedding_revision:
            raise ValueError("embedding revision does not match the vector manifest")
