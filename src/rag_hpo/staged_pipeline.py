from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
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
from rag_hpo.embeddings import EmbeddingBackend, create_backend
from rag_hpo.export import export_results
from rag_hpo.fasthpocr import FastHPORecognizer
from rag_hpo.lexical import NativeLexicalRecognizer
from rag_hpo.models import (
    AnnotationInput,
    AnnotationResult,
    Candidate,
    Category,
    MappingDecisionBatch,
    PhenotypeExtraction,
    SpanPhenotype,
    SpanPhenotypeExtraction,
)
from rag_hpo.pipeline import _short_error, hash_inputs
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.prompts import load_prompts
from rag_hpo.provider import ProviderError
from rag_hpo.registry import load_registry_bundle, normalize_phrase
from rag_hpo.retrieval import HybridCandidateRetriever
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
    category: Category
    methods: set[str] = field(default_factory=set)
    recognizer_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ContextPacket:
    section: str | None
    subject: str
    assertion: str
    sentence: str
    preceding_sentence: str | None
    following_sentence: str | None
    extended_context_reason: str | None


_SENTENCE = re.compile(r"[^\n.!?]+(?:[.!?]+|(?=\n)|$)", re.UNICODE)
_LIST_CUE = re.compile(r"[,;]|\b(?:and|with|including)\b", re.I)
_MEASUREMENT_CUE = re.compile(
    r"\b(?:high|low|raised|reduced|decreased|increased|elevated|abnormal)\b|"
    r"\d+(?:\.\d+)?\s*(?:mg|g|mmol|µmol|cm|mm|kg|%|bpm|mmhg)\b",
    re.I,
)
MAPPING_BATCH_SIZE = 8
_CONTEXT_CUE = re.compile(
    r"\b(?:it|this|these|those|they|former|latter|respectively|"
    r"however|therefore|subsequently|also|both)\b",
    re.I,
)
_MAX_ADJACENT_CONTEXT_CHARS = 280


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


def build_context_packet(
    note: str,
    mention: Mention,
    assertion: AssertionDecision,
    candidates: list[Candidate],
) -> ContextPacket:
    """Build bounded, deterministic context without another model call."""

    sentences = sentence_spans(note)
    sentence_index = next(
        (
            index
            for index, sentence in enumerate(sentences)
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
            preceding = sentences[sentence_index - 1].text[-_MAX_ADJACENT_CONTEXT_CHARS:]
        if sentence_index + 1 < len(sentences):
            following = sentences[sentence_index + 1].text[:_MAX_ADJACENT_CONTEXT_CHARS]

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
        self._concepts = {
            concept.hp_id: concept for concept in self.registry.concepts if not concept.obsolete
        }
        self.distinct_limit = 32 if mode is AnnotationMode.HIGH_RECALL else 16

    def run(
        self,
        rows: list[AnnotationInput],
        *,
        initial_errors: list[AnnotationResult] | None = None,
    ) -> list[AnnotationResult]:
        started = time.perf_counter()
        ensure_private_directory(self.output_dir)
        state_path = self.output_dir / ".rag-hpo-state.sqlite3"
        config_hash = hashlib.sha256(
            json.dumps(
                {
                    "manifest": sha256_file(self.vector_dir / "hpo_manifest.json"),
                    "mode": self.mode.value,
                    "distinct_limit": self.distinct_limit,
                    "model_enabled": self.model_enabled,
                    "mapping_prompt": self.mapping_prompt.value,
                    "prompts": sha256_file(Path(__file__).parent / "data" / "system_prompts.json"),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        all_results = list(initial_errors or [])
        completed_successfully = False
        try:
            with PipelineState(
                state_path,
                input_sha256=hash_inputs(rows),
                artifact_sha256=config_hash,
                pipeline_version=__version__,
                resume=self.resume,
            ) as state:
                for row_index, row in enumerate(rows):
                    note_hash = hashlib.sha256(row.clinical_note.encode("utf-8")).hexdigest()
                    cached = state.completed(row_index, note_hash) if self.resume else None
                    if cached is not None:
                        all_results.extend(cached)
                        continue
                    try:
                        results = self._annotate_row(row, row_index)
                        state.save_complete(
                            row_index,
                            row.patient_id,
                            note_hash,
                            results,
                        )
                        all_results.extend(results)
                    except (ProviderError, ValueError, RuntimeError) as exc:
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
                                category=Category.OTHER,
                                mapping_status="error",
                                error_code=code,
                                error_message=message,
                                review_status="rejected",
                            )
                        )
                export_results(all_results, self.output_dir)
                self._write_run_manifest(
                    rows=rows,
                    results=all_results,
                    input_sha256=hash_inputs(rows),
                    config_sha256=config_hash,
                    elapsed_seconds=time.perf_counter() - started,
                )
                completed_successfully = not any(
                    result.mapping_status == "error" for result in all_results
                )
        finally:
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
    ) -> None:
        provider_usage = getattr(self.provider, "usage", None) or {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        provider_config = getattr(self.provider, "config", None)
        path = self.output_dir / "rag_hpo_run_manifest.json"
        prior_attempts: list[dict[str, Any]] = []
        if self.resume and path.exists():
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
            "mode": self.mode.value,
            "mapping_prompt": self.mapping_prompt.value,
            "distinct_candidate_limit": self.distinct_limit,
            "input_rows": len(rows),
            "input_sha256": input_sha256,
            "config_sha256": config_sha256,
            "artifact_manifest_sha256": sha256_file(self.vector_dir / "hpo_manifest.json"),
            "registry_sha256": self.registry_manifest.registry_sha256,
            "prompt_bundle_sha256": sha256_file(
                Path(__file__).parent / "data" / "system_prompts.json"
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
                "deterministic_candidate_order": True,
                "provider_seed": "not-configured",
                "quantization": "not-recorded",
                "embedding_backend": self.manifest.embedding_backend,
                "embedding_model": self.manifest.embedding_model,
                "embedding_revision": self.manifest.embedding_revision,
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
        mentions = self._recognizer_mentions(note)
        if self.model_enabled:
            first_mentions = self._extract_with_strict_chunk_fallback(
                note=note,
                row_index=row_index,
                system_message=self.prompts["coverage_extraction"],
                stage="coverage-extract",
                method="model-pass-1",
                temperature=0.2,
            )
            mentions.extend(first_mentions)
            targets = self._coverage_targets(note, mentions, first_mentions)
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
                )
                mentions.extend(second_mentions)
        merged = self._merge_mentions(mentions)
        return self._map_mentions(row, row_index, merged)

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
    ) -> list[Mention]:
        try:
            extraction, raw = self.provider_request(
                system_message=system_message,
                user_message=primary_payload or note,
                response_model=PhenotypeExtraction,
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
            (value.start, value.end) for value in sentence_spans(note)
        ]
        values: list[Mention] = []
        for chunk_index, (start, end) in enumerate(fallback_ranges):
            extraction, raw = self.provider_request(
                system_message=system_message,
                user_message=note[start:end],
                response_model=PhenotypeExtraction,
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
                category=Category.ABNORMAL,
                methods={"native"},
                recognizer_ids=set(value.candidate_hpo_ids),
            )
            for value in self.native.recognize(note)
        ]
        if self.fast is not None:
            mentions.extend(
                Mention(
                    phrase=value.phrase,
                    start=value.start_offset,
                    end=value.end_offset,
                    category=Category.ABNORMAL,
                    methods={"fasthpocr"},
                    recognizer_ids=set(
                        value.candidate_hpo_ids or ((value.hpo_id,) if value.hpo_id else ())
                    ),
                )
                for value in self.fast.annotate(note)
            )
        return mentions

    @staticmethod
    def _validated_extraction(
        note: str,
        extraction: PhenotypeExtraction | SpanPhenotypeExtraction,
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
                        category=phenotype.category,
                        methods={method},
                    )
                )
                continue
            if isinstance(phenotype, SpanPhenotype):
                match = min(
                    matches,
                    key=lambda item: abs(item.start() - phenotype.start_offset),
                )
            else:
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
                    category=phenotype.category,
                    methods={method},
                )
            )
        return values

    @staticmethod
    def _coverage_targets(
        note: str,
        all_mentions: list[Mention],
        first_mentions: list[Mention],
    ) -> list[SentenceSpan]:
        targets: list[SentenceSpan] = []
        for sentence in sentence_spans(note):
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
            if unmatched_recognizer or list_gap or measurement_gap:
                targets.append(sentence)
        return targets

    @staticmethod
    def _merge_mentions(mentions: list[Mention]) -> list[Mention]:
        merged: dict[tuple[int, int, str], Mention] = {}
        for mention in mentions:
            key = (
                mention.start,
                mention.end,
                normalize_phrase(mention.phrase),
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = mention
            else:
                if any(method.startswith("model-pass") for method in mention.methods) and not any(
                    method.startswith("model-pass") for method in existing.methods
                ):
                    existing.category = mention.category
                existing.methods.update(mention.methods)
                existing.recognizer_ids.update(mention.recognizer_ids)
        return sorted(
            merged.values(),
            key=lambda value: (
                value.start,
                value.end,
                value.category.value,
                normalize_phrase(value.phrase),
            ),
        )

    def _map_mentions(
        self,
        row: AnnotationInput,
        row_index: int,
        mentions: list[Mention],
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
            if mention.category is not Category.ABNORMAL:
                results.append(
                    self._result(
                        row,
                        mention,
                        assertion,
                        mapping_status="not_mapped_category",
                        review_status="review",
                    )
                )
                continue
            pending.append((f"m{index:04d}", mention, assertion))

        retrieved_batches = (
            self.retriever.retrieve_many(
                [mention.phrase for _mention_id, mention, _assertion in pending],
                distinct_limit=self.distinct_limit,
            )
            if self.retriever is not None
            else [[] for _value in pending]
        )
        for (
            (mention_id, mention, assertion),
            retrieved,
        ) in zip(pending, retrieved_batches, strict=True):
            candidates = self._inject_recognizer_candidates(
                retrieved,
                mention.recognizer_ids,
            )
            lower_confidence_addition = (
                "model-pass-1" not in mention.methods
                and not {"native", "fasthpocr"} <= mention.methods
            )
            if (
                self.model_enabled
                and self.mode is AnnotationMode.BALANCED
                and lower_confidence_addition
            ):
                results.append(
                    self._result(
                        row,
                        mention,
                        assertion,
                        candidate_ids=[value.hpo_id for value in candidates],
                        mapping_status="no_candidate_fit",
                        confidence="low",
                        review_status="review",
                    )
                )
                continue
            abnormal.append((mention_id, mention, assertion, candidates))

        if not abnormal:
            return results
        if not self.model_enabled:
            for _mention_id, mention, assertion, candidates in abnormal:
                selected = (
                    next(
                        (value for value in candidates if value.hpo_id in mention.recognizer_ids),
                        None,
                    )
                    if len(mention.recognizer_ids) == 1
                    else None
                )
                results.append(
                    self._result(
                        row,
                        mention,
                        assertion,
                        candidate_ids=[value.hpo_id for value in candidates],
                        selected=selected,
                        mapping_status="mapped" if selected else "no_candidate_fit",
                        confidence="high" if selected else "low",
                        review_status=(
                            "accepted" if selected and assertion.status == "affirmed" else "review"
                        ),
                    )
                )
            return self._deduplicate(results)

        expected = {value[0] for value in abnormal}
        decisions: dict[str, Any] = {}
        for batch_index, start in enumerate(range(0, len(abnormal), MAPPING_BATCH_SIZE)):
            items = abnormal[start : start + MAPPING_BATCH_SIZE]
            batch_decisions = self._request_mapping_items(
                items,
                note=row.clinical_note,
                row_index=row_index,
                stage=f"batch-map-{batch_index:03d}",
            )
            if set(decisions) & set(batch_decisions):
                raise ValueError("batched mapper returned duplicate decisions")
            decisions.update(batch_decisions)
        if set(decisions) != expected:
            raise ValueError("batched mapper returned incomplete or duplicate decisions")

        for mention_id, mention, assertion, candidates in abnormal:
            decision = decisions[mention_id]
            by_id = {value.hpo_id: value for value in candidates}
            selected = by_id.get(decision.hpo_id or "")
            consensus = bool(selected and selected.hpo_id in mention.recognizer_ids)
            base_finding = "model-pass-1" in mention.methods
            if consensus:
                review_status = "accepted"
                confidence = "high"
            elif base_finding and selected is not None:
                review_status = "accepted"
                confidence = decision.confidence
            elif (
                decision.verdict == "supported"
                and decision.confidence in {"high", "medium"}
                and selected is not None
            ):
                review_status = "accepted"
                confidence = decision.confidence
            elif decision.verdict == "unsupported" and decision.confidence == "high":
                review_status = "rejected"
                confidence = "high"
                selected = None
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
                    candidate_ids=[value.hpo_id for value in candidates],
                    selected=selected,
                    mapping_status="mapped" if selected else "no_candidate_fit",
                    confidence=confidence,
                    review_status=review_status,
                )
            )
        return self._deduplicate(results)

    def _request_mapping_items(
        self,
        items: list[tuple[str, Mention, AssertionDecision, list[Candidate]]],
        *,
        note: str,
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
                response_model=MappingDecisionBatch,
                temperature=0.0,
            )
        except ProviderError as exc:
            if exc.status_code != 400 or len(items) == 1:
                raise
            midpoint = len(items) // 2
            left = self._request_mapping_items(
                items[:midpoint],
                note=note,
                row_index=row_index,
                stage=f"{stage}-a",
            )
            right = self._request_mapping_items(
                items[midpoint:],
                note=note,
                row_index=row_index,
                stage=f"{stage}-b",
            )
            if set(left) & set(right):
                raise ValueError("batched mapper returned duplicate decisions") from exc
            return {**left, **right}
        self._write_raw(row_index, stage, raw)
        decisions = {value.mention_id: value for value in batch.decisions}
        expected = {value[0] for value in items}
        if len(decisions) != len(batch.decisions) or set(decisions) != expected:
            if len(items) > 1:
                midpoint = len(items) // 2
                left = self._request_mapping_items(
                    items[:midpoint],
                    note=note,
                    row_index=row_index,
                    stage=f"{stage}-incomplete-a",
                )
                right = self._request_mapping_items(
                    items[midpoint:],
                    note=note,
                    row_index=row_index,
                    stage=f"{stage}-incomplete-b",
                )
                if set(left) & set(right):
                    raise ValueError("batched mapper returned duplicate decisions")
                return {**left, **right}
            raise ValueError("batched mapper returned incomplete or duplicate decisions")
        return decisions

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
        selected: Candidate | None = None,
        confidence: str | None = None,
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
                "evidence_start": mention.start,
                "evidence_end": mention.end,
                "assertion_status": assertion.status,
                "confidence": confidence,
                "review_status": review_status,
                "source_methods": sorted(mention.methods),
                "candidate_hpo_ids": candidate_ids,
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
        return self.provider.request(**kwargs)

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
