#!/usr/bin/env python3
"""Trace CSC Cases 1 and 99 through every RAG-HPO pipeline boundary.

Raw notes and provider payloads are written only to the explicitly supplied
private audit directory. Gold annotations are loaded only after all inference,
including the forced-deferral counterfactual, has completed.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404
import time
import unicodedata
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import httpx
import numpy as np
import tiktoken
from pydantic import BaseModel, ValidationError

from rag_hpo.artifacts import sha256_file
from rag_hpo.assertion import analyze_assertion
from rag_hpo.benchmark import (
    load_hpo_aliases,
    load_prediction_groups,
    load_prediction_sets,
    load_reference_groups,
    score_prediction_groups,
    score_reference_groups,
    summarize,
)
from rag_hpo.config import ProviderConfig
from rag_hpo.diagnostics import parse_obo
from rag_hpo.fasthpocr import FastHPORecognizer
from rag_hpo.hierarchy_scoring import score_layered_case
from rag_hpo.models import (
    AnnotationInput,
    AnnotationResult,
    Candidate,
    PhenotypeSpanExtraction,
)
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.provider import RETRYABLE_STATUSES, OpenAICompatibleProvider, ProviderError
from rag_hpo.registry import normalize_phrase
from rag_hpo.retrieval import HybridCandidateRetriever
from rag_hpo.staged_pipeline import (
    AnnotationMode,
    MappingPromptMode,
    Mention,
    SentenceSpan,
    StagedAnnotationPipeline,
)

T = TypeVar("T", bound=BaseModel)
CASE_IDS = ("1", "99")
INPUT_TOKEN_CEILING = 96_000
INPUT_TOKEN_WARNING = 76_800
TOTAL_TOKEN_CEILING = 300_000
SAFE_RESPONSE_HEADERS = {
    "content-type",
    "date",
    "retry-after",
    "x-request-id",
    "x-groq-region",
}


def canonicalize_note(value: str) -> str:
    """Return the single text representation used for hashes and offsets."""

    normalized_newlines = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_newlines).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mention_dict(mention: Mention) -> dict[str, Any]:
    return {
        "phrase": mention.phrase,
        "start_char": mention.start,
        "end_char": mention.end,
        "length": mention.end - mention.start,
        "methods": sorted(mention.methods),
        "recognizer_hpo_ids": sorted(mention.recognizer_ids),
    }


class PrivateWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        ensure_private_directory(root)
        os.chmod(root, 0o700)

    def directory(self, relative: str | Path) -> Path:
        path = self.root / relative
        ensure_private_directory(path)
        os.chmod(path, 0o700)
        return path

    def write_text(self, relative: str | Path, value: str) -> Path:
        path = self.root / relative
        ensure_private_directory(path.parent)
        path.write_text(value, encoding="utf-8")
        restrict_owner(path)
        return path

    def write_json(self, relative: str | Path, value: Any) -> Path:
        return self.write_text(
            relative,
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )


class TracingProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider that retains the pre-parse response privately."""

    def __init__(self, config: ProviderConfig, *, writer: PrivateWriter) -> None:
        super().__init__(config)
        self.writer = writer
        existing_summaries = (
            json.loads((writer.root / "provider-summary.json").read_text(encoding="utf-8"))
            if (writer.root / "provider-summary.json").is_file()
            else []
        )
        existing_call_numbers = [
            int(match.group(1))
            for path in (writer.root / "provider").glob("call-*")
            if (match := re.match(r"call-(\d+)-", path.name))
        ]
        self.logical_call = max(existing_call_numbers, default=0)
        self.usage = {
            "requests": len(existing_call_numbers),
            "input_tokens": sum(
                int(value.get("prompt_tokens") or 0) for value in existing_summaries
            ),
            "output_tokens": sum(
                int(value.get("completion_tokens") or 0) for value in existing_summaries
            ),
            "total_tokens": sum(
                int(value.get("total_tokens") or 0) for value in existing_summaries
            ),
        }
        self.stage = "unlabeled"
        self.patient_id = "unknown"
        self.call_summaries: list[dict[str, Any]] = []
        try:
            self.encoding = tiktoken.get_encoding("o200k_base")
        except ValueError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    @contextmanager
    def context(self, *, stage: str, patient_id: str) -> Iterator[None]:
        previous = (self.stage, self.patient_id)
        self.stage, self.patient_id = stage, patient_id
        try:
            yield
        finally:
            self.stage, self.patient_id = previous

    def _estimate_tokens(self, payload: dict[str, Any]) -> int:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return len(self.encoding.encode(serialized))

    @staticmethod
    def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
        return {
            key.lower(): value
            for key, value in headers.items()
            if key.lower() in SAFE_RESPONSE_HEADERS or key.lower().startswith("x-ratelimit-")
        }

    def request(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[T],
        temperature: float = 0.2,
    ) -> tuple[T, str]:
        self.logical_call += 1
        self.usage["requests"] += 1
        call_name = (
            f"call-{self.logical_call:04d}-case-{self.patient_id}-"
            f"{re.sub(r'[^a-zA-Z0-9_.-]+', '-', self.stage)}"
        )
        call_dir = self.writer.directory(Path("provider") / call_name)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
        }
        response_format = self._response_format(response_model)
        if response_format is not None:
            payload["response_format"] = response_format
        estimated_tokens = self._estimate_tokens(payload)
        warning = estimated_tokens >= INPUT_TOKEN_WARNING
        if estimated_tokens > INPUT_TOKEN_CEILING:
            raise ProviderError(
                "audit_token_ceiling",
                f"estimated provider input exceeds {INPUT_TOKEN_CEILING} tokens",
            )
        if self.usage["total_tokens"] + estimated_tokens > TOTAL_TOKEN_CEILING:
            raise ProviderError(
                "audit_total_token_ceiling",
                f"audit would exceed {TOTAL_TOKEN_CEILING} total tokens",
            )
        request_record = {
            "patient_id": self.patient_id,
            "stage": self.stage,
            "response_model": response_model.__name__,
            "estimated_input_tokens": estimated_tokens,
            "input_token_warning": warning,
            "payload": payload,
        }
        path = call_dir / "request.json"
        path.write_text(
            json.dumps(request_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        restrict_owner(path)

        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        response: httpx.Response | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            started = time.perf_counter()
            attempt_record: dict[str, Any] = {"attempt": attempt}
            try:
                response = self.client.post(
                    str(self.config.base_url),
                    headers=headers,
                    json=payload,
                )
                attempt_record.update(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 6),
                        "status_code": response.status_code,
                        "headers": self._safe_headers(response.headers),
                        "body": response.text,
                    }
                )
            except httpx.TimeoutException:
                attempt_record.update(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 6),
                        "error": "timeout",
                    }
                )
            except httpx.HTTPError as exc:
                attempt_record.update(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 6),
                        "error": type(exc).__name__,
                    }
                )
            attempt_path = call_dir / f"attempt-{attempt:02d}.json"
            attempt_path.write_text(
                json.dumps(attempt_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            restrict_owner(attempt_path)

            if response is not None and response.is_success:
                break
            if response is not None and response.status_code not in RETRYABLE_STATUSES:
                raise ProviderError(
                    "provider_rejected",
                    f"provider rejected the request with HTTP {response.status_code}",
                    response.status_code,
                )
            if attempt == self.config.max_attempts:
                code = "timeout" if response is None else "retry_exhausted"
                raise ProviderError(code, f"provider failed after {attempt} attempts")
            retry_after = response.headers.get("Retry-After") if response is not None else None
            self._sleep(self._delay(attempt, retry_after))
            response = None
        if response is None:
            raise AssertionError("provider loop completed without a response")

        validation_error: str | None = None
        envelope: dict[str, Any] | None = None
        content = ""
        try:
            envelope = response.json()
            envelope_path = call_dir / "response-envelope.json"
            envelope_path.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            restrict_owner(envelope_path)
            usage = envelope.get("usage", {})
            for source, target in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                value = usage.get(source)
                if isinstance(value, int):
                    self.usage[target] += value
            content = envelope["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("completion content is not text")
            raw_path = call_dir / "raw-content.txt"
            raw_path.write_text(content, encoding="utf-8")
            restrict_owner(raw_path)
            parsed = response_model.model_validate_json(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            error_path = call_dir / "validation-error.txt"
            error_path.write_text(validation_error, encoding="utf-8")
            restrict_owner(error_path)
            raise ProviderError(
                "invalid_response",
                "provider returned an invalid response",
            ) from exc
        finally:
            usage = (envelope or {}).get("usage", {})
            choice = ((envelope or {}).get("choices") or [{}])[0]
            summary = {
                "call": self.logical_call,
                "patient_id": self.patient_id,
                "stage": self.stage,
                "temperature": temperature,
                "estimated_input_tokens": estimated_tokens,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "system_fingerprint": (envelope or {}).get("system_fingerprint"),
                "response_id": (envelope or {}).get("id"),
                "input_token_warning": warning,
                "validation_error": validation_error,
            }
            self.call_summaries.append(summary)
            summary_path = call_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            restrict_owner(summary_path)
        return parsed, content


class TracingRetriever:
    """Execute retrieval once while retaining all pre-bounding lanes."""

    def __init__(self, base: HybridCandidateRetriever) -> None:
        self.base = base
        self.current_patient = "unknown"
        self.calls: list[dict[str, Any]] = []
        self.returned: dict[tuple[str, str], list[list[Candidate]]] = {}

    def retrieve_many(
        self,
        phrases: list[str],
        *,
        distinct_limit: int,
        contexts: list[str | None] | None = None,
    ) -> list[list[Candidate]]:
        if not phrases:
            return []
        context_rows = [
            (index, value)
            for index, value in enumerate(contexts or [])
            if value and normalize_phrase(value) != normalize_phrase(phrases[index])
        ]
        queries = self.base.backend.encode([*phrases, *(value for _index, value in context_rows)])
        raw_limit = min(self.base.raw_limit, self.base._dense_index.ntotal)
        scores, indices = self.base._dense_index.search(
            np.asarray(queries[: len(phrases)], dtype=np.float32),
            raw_limit,
        )
        context_by_phrase: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if context_rows:
            context_scores, context_indices = self.base._dense_index.search(
                np.asarray(queries[len(phrases) :], dtype=np.float32),
                raw_limit,
            )
            context_by_phrase = {
                phrase_index: (context_scores[index], context_indices[index])
                for index, (phrase_index, _value) in enumerate(context_rows)
            }
        sparse_limit = min(self.base.raw_limit, len(self.base._lexical_phrases))
        sparse_queries = self.base._sparse_vectorizer.transform(phrases)
        sparse_distances, sparse_indices = self.base._sparse_index.kneighbors(
            sparse_queries,
            n_neighbors=sparse_limit,
        )
        output: list[list[Candidate]] = []
        per_phrase: dict[str, list[Candidate]] = {}
        for index, phrase in enumerate(phrases):
            dense = self.base._dense_from_search(scores[index], indices[index])
            lexical = self.base._lexical(
                phrase,
                sparse_distances=sparse_distances[index],
                sparse_indices=sparse_indices[index],
            )
            context_dense = (
                self.base._dense_from_search(*context_by_phrase[index])
                if index in context_by_phrase
                else {}
            )
            fused = self.base._fuse(dense, lexical, context_dense)
            candidates = self.base._candidates(
                dense,
                lexical,
                context_dense,
                distinct_limit=distinct_limit,
            )
            output.append(candidates)
            per_phrase.setdefault(phrase, candidates)
            self.calls.append(
                {
                    "patient_id": self.current_patient,
                    "phrase": phrase,
                    "dense_top64": [
                        {"hpo_id": hp_id, "rank": rank, "score": score}
                        for hp_id, (rank, score) in sorted(
                            dense.items(), key=lambda value: value[1][0]
                        )
                    ],
                    "lexical_top64": [
                        {"hpo_id": hp_id, "rank": rank, "score": score}
                        for hp_id, (rank, score) in sorted(
                            lexical.items(), key=lambda value: value[1][0]
                        )
                    ],
                    "fused_top64": [
                        {
                            "hpo_id": value.hpo_id,
                            "rank": rank,
                            "score": value.combined_score,
                            "dense_rank": value.dense_rank,
                            "lexical_rank": value.lexical_rank,
                        }
                        for rank, value in enumerate(fused[:64], start=1)
                    ],
                    "bounded": [
                        {
                            "hpo_id": value.hpo_id,
                            "rank": rank,
                            "score": value.score,
                            "dense_rank": value.dense_rank,
                            "dense_score": value.dense_score,
                            "lexical_rank": value.lexical_rank,
                            "lexical_score": value.lexical_score,
                        }
                        for rank, value in enumerate(candidates, start=1)
                    ],
                }
            )
        self.returned.setdefault((self.current_patient, str(len(self.calls))), []).extend(output)
        self.latest_by_patient = getattr(self, "latest_by_patient", {})
        self.latest_by_patient[self.current_patient] = per_phrase
        return output


class TracePipeline(StagedAnnotationPipeline):
    def __init__(self, *args: Any, writer: PrivateWriter, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.events: list[dict[str, Any]] = []
        self.current_patient = "unknown"
        self.current_extraction_stage = "unknown"
        self.merged_mentions: dict[str, list[Mention]] = {}
        self.mapper_payloads: list[dict[str, Any]] = []
        if self.retriever is not None:
            self.trace_retriever = TracingRetriever(self.retriever)
            self.retriever = self.trace_retriever  # type: ignore[assignment]
        else:
            self.trace_retriever = None

    def _event(self, stage: str, **values: Any) -> None:
        self.events.append(
            {
                "patient_id": self.current_patient,
                "stage": stage,
                **values,
            }
        )

    def _annotate_row(
        self,
        row: AnnotationInput,
        row_index: int,
    ) -> list[AnnotationResult]:
        self.current_patient = row.patient_id
        return super()._annotate_row(row, row_index)

    def _recognizer_mentions(self, note: str) -> list[Mention]:
        mentions = super()._recognizer_mentions(note)
        self._event(
            "recognizer-mentions",
            mentions=[mention_dict(value) for value in mentions],
        )
        return mentions

    def _extract_with_strict_chunk_fallback(self, **kwargs: Any) -> list[Mention]:
        self.current_extraction_stage = str(kwargs["stage"])
        provider = self.provider
        context = (
            provider.context(
                stage=self.current_extraction_stage,
                patient_id=self.current_patient,
            )
            if isinstance(provider, TracingProvider)
            else nullcontext()
        )
        with context:
            values = super()._extract_with_strict_chunk_fallback(**kwargs)
        self._event(
            self.current_extraction_stage,
            temperature=kwargs["temperature"],
            validated_mentions=[mention_dict(value) for value in values],
        )
        return values

    def _validated_extraction(
        self,
        note: str,
        extraction: PhenotypeSpanExtraction,
        method: str,
        *,
        allowed_ranges: list[tuple[int, int]] | None = None,
    ) -> list[Mention]:
        values = StagedAnnotationPipeline._validated_extraction(
            note,
            extraction,
            method,
            allowed_ranges=allowed_ranges,
        )
        self._event(
            f"{self.current_extraction_stage}-validation",
            provider_phrases=[value.phrase for value in extraction.phenotypes],
            accepted_mentions=[mention_dict(value) for value in values],
            dropped_phrases=[
                value.phrase
                for value in extraction.phenotypes
                if not any(
                    value.phrase.casefold() == mention.phrase.casefold() for mention in values
                )
            ],
            allowed_ranges=allowed_ranges,
        )
        return values

    def _coverage_targets(
        self,
        note: str,
        all_mentions: list[Mention],
        first_mentions: list[Mention],
    ) -> list[SentenceSpan]:
        values = StagedAnnotationPipeline._coverage_targets(
            note,
            all_mentions,
            first_mentions,
        )
        self._event(
            "coverage-targets",
            targets=[asdict(value) for value in values],
        )
        return values

    def _merge_mentions(self, mentions: list[Mention]) -> list[Mention]:
        values = StagedAnnotationPipeline._merge_mentions(mentions)
        output_keys = {(value.start, value.end, value.phrase.casefold()) for value in values}
        suppressed = [
            {
                **mention_dict(value),
                "shrinkage_candidates": [
                    {
                        **mention_dict(other),
                        "fraction_reduction": round(
                            1 - ((other.end - other.start) / (value.end - value.start)),
                            6,
                        ),
                    }
                    for other in values
                    if value.start <= other.start
                    and value.end >= other.end
                    and (value.start, value.end) != (other.start, other.end)
                ],
            }
            for value in mentions
            if (value.start, value.end, value.phrase.casefold()) not in output_keys
        ]
        self._event(
            "mention-merge",
            inputs=[mention_dict(value) for value in mentions],
            outputs=[mention_dict(value) for value in values],
            suppressed=suppressed,
        )
        return values

    def _map_mentions(
        self,
        row: AnnotationInput,
        row_index: int,
        mentions: list[Mention],
    ) -> list[AnnotationResult]:
        self.merged_mentions[row.patient_id] = mentions
        if self.trace_retriever is not None:
            self.trace_retriever.current_patient = row.patient_id
        self._event(
            "mapping-input",
            mentions=[mention_dict(value) for value in mentions],
        )
        values = super()._map_mentions(row, row_index, mentions)
        self._event(
            "mapping-output",
            results=[value.model_dump(mode="json") for value in values],
        )
        return values

    def _request_mapping_items(
        self,
        items: list[tuple[str, Mention, Any, list[Candidate]]],
        *,
        note: str,
        row_index: int,
        stage: str,
    ) -> dict[str, Any]:
        self.mapper_payloads.append(
            {
                "patient_id": self.current_patient,
                "stage": stage,
                "items": [
                    {
                        "mention_id": mention_id,
                        "mention": mention_dict(mention),
                        "assertion": asdict(assertion),
                        "candidate_hpo_ids": [value.hpo_id for value in candidates],
                    }
                    for mention_id, mention, assertion, candidates in items
                ],
            }
        )
        provider = self.provider
        context = (
            provider.context(stage=stage, patient_id=self.current_patient)
            if isinstance(provider, TracingProvider)
            else nullcontext()
        )
        with context:
            return super()._request_mapping_items(
                items,
                note=note,
                row_index=row_index,
                stage=stage,
            )

    def _finalize_categories(
        self,
        row: AnnotationInput,
        row_index: int,
        results: list[AnnotationResult],
    ) -> list[AnnotationResult]:
        provider = self.provider
        context = (
            provider.context(stage="final-categorization", patient_id=row.patient_id)
            if isinstance(provider, TracingProvider)
            else nullcontext()
        )
        with context:
            values = super()._finalize_categories(row, row_index, results)
        self._event(
            "final-categorization",
            before=[value.model_dump(mode="json") for value in results],
            after=[value.model_dump(mode="json") for value in values],
        )
        return values


def load_cases(path: Path) -> tuple[list[AnnotationInput], list[dict[str, Any]]]:
    rows: list[AnnotationInput] = []
    baseline: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            patient_id = str(raw.get("Case") or raw.get("patient_id") or "").strip()
            if patient_id not in CASE_IDS:
                continue
            original = str(raw.get("clinical_note") or "")
            canonical = canonicalize_note(original)
            if "\r" in canonical or unicodedata.normalize("NFC", canonical) != canonical:
                raise ValueError(f"Case {patient_id} failed canonical text validation")
            rows.append(AnnotationInput(patient_id=patient_id, clinical_note=canonical))
            baseline.append(
                {
                    "patient_id": patient_id,
                    "original_sha256": sha256_text(original),
                    "canonical_sha256": sha256_text(canonical),
                    "original_length": len(original),
                    "canonical_length": len(canonical),
                    "changed_by_canonicalization": original != canonical,
                    "canonical_note": canonical,
                }
            )
    if {value.patient_id for value in rows} != set(CASE_IDS):
        raise ValueError("CSC input must contain exactly Cases 1 and 99")
    return rows, baseline


def run_fast_overlay(
    rows: list[AnnotationInput],
    *,
    pipeline: TracePipeline,
    index_path: Path,
) -> list[dict[str, Any]]:
    recognizer = FastHPORecognizer(
        index_path,
        longest_match=True,
        registry=pipeline.registry,
    )
    return [
        {
            "patient_id": row.patient_id,
            "mentions": [
                {
                    "phrase": value.phrase,
                    "start_char": value.start_offset,
                    "end_char": value.end_offset,
                    "hpo_id": value.hpo_id,
                    "candidate_hpo_ids": list(value.candidate_hpo_ids),
                    "resolution": value.resolution,
                }
                for value in recognizer.annotate(row.clinical_note)
            ],
        }
        for row in rows
    ]


def force_deferred(
    rows: list[AnnotationInput],
    results: list[AnnotationResult],
    *,
    pipeline: TracePipeline,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rows_by_id = {value.patient_id: value for value in rows}
    for patient_id in CASE_IDS:
        row = rows_by_id[patient_id]
        mentions = pipeline.merged_mentions[patient_id]
        retrieved_by_phrase = (
            pipeline.trace_retriever.latest_by_patient.get(patient_id, {})
            if pipeline.trace_retriever is not None
            else {}
        )
        deferred = [
            value
            for value in results
            if value.patient_id == patient_id
            and value.mapping_status == "no_candidate_fit"
            and value.candidate_hpo_ids
        ]
        items: list[tuple[str, Mention, Any, list[Candidate]]] = []
        result_by_mention: dict[str, AnnotationResult] = {}
        for index, result in enumerate(deferred):
            mention = next(
                (
                    value
                    for value in mentions
                    if value.start == result.evidence_start
                    and value.end == result.evidence_end
                    and value.phrase == result.phrase
                ),
                None,
            )
            if mention is None:
                continue
            retrieved = retrieved_by_phrase.get(mention.phrase, [])
            candidates = pipeline._inject_recognizer_candidates(
                retrieved,
                mention.recognizer_ids,
            )
            mention_id = f"fd-{patient_id}-{index:03d}"
            items.append(
                (
                    mention_id,
                    mention,
                    analyze_assertion(row.clinical_note, mention.start, mention.end),
                    candidates,
                )
            )
            result_by_mention[mention_id] = result
        if not items:
            continue
        pipeline.current_patient = patient_id
        decisions = pipeline._request_mapping_items(
            items,
            note=row.clinical_note,
            row_index=int(patient_id),
            stage="forced-deferred-map",
        )
        for mention_id, mention, assertion, candidates in items:
            decision = decisions.get(mention_id)
            output.append(
                {
                    "patient_id": patient_id,
                    "mention_id": mention_id,
                    "phrase": mention.phrase,
                    "start_char": mention.start,
                    "end_char": mention.end,
                    "original_candidate_hpo_ids": (
                        result_by_mention[mention_id].candidate_hpo_ids or []
                    ),
                    "forced_payload_hpo_ids": [value.hpo_id for value in candidates],
                    "decision": decision.model_dump(mode="json") if decision else None,
                    "assertion": asdict(assertion),
                }
            )
    return output


def force_stored_deferred(
    rows: list[AnnotationInput],
    stored_results_path: Path,
    *,
    pipeline: TracePipeline,
) -> list[dict[str, Any]]:
    """Map every stored Case 99 deferred candidate without access to gold."""

    row = next(value for value in rows if value.patient_id == "99")
    stored = [
        value
        for value in json.loads(stored_results_path.read_text(encoding="utf-8"))
        if str(value.get("patient_id")) == "99"
        and value.get("mapping_status") == "no_candidate_fit"
        and value.get("candidate_hpo_ids")
    ]
    items: list[tuple[str, Mention, Any, list[Candidate]]] = []
    source_by_id: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(stored):
        start = int(result["evidence_start"])
        end = int(result["evidence_end"])
        phrase = row.clinical_note[start:end]
        mention_id = f"stored-fd-99-{index:03d}"
        candidate_ids = list(
            dict.fromkeys(
                [
                    *result.get("candidate_hpo_ids", []),
                    *result.get("retrieval_candidate_hpo_ids", []),
                ]
            )
        )[:16]
        candidates: list[Candidate] = []
        for hp_id in candidate_ids:
            concept = pipeline._concepts.get(hp_id)
            if concept is None:
                continue
            candidates.append(
                Candidate(
                    hpo_id=hp_id,
                    term=concept.label,
                    score=0.0,
                    definition=concept.definition[:500] or None,
                    synonyms=sorted({value.phrase for value in concept.phrases})[:12],
                    parents=concept.parents,
                    source_methods=["stored-regression"],
                )
            )
        mention = Mention(
            phrase=phrase,
            start=start,
            end=end,
            methods=set(result.get("source_methods") or []),
            recognizer_ids=set(result.get("candidate_hpo_ids") or []),
        )
        assertion = analyze_assertion(row.clinical_note, start, end)
        items.append((mention_id, mention, assertion, candidates))
        source_by_id[mention_id] = result

    decisions: dict[str, Any] = {}
    pipeline.current_patient = "99"
    for batch_index, start in enumerate(range(0, len(items), 8)):
        decisions.update(
            pipeline._request_mapping_items(
                items[start : start + 8],
                note=row.clinical_note,
                row_index=99,
                stage=f"forced-stored-deferred-{batch_index:03d}",
            )
        )
    return [
        {
            "patient_id": "99",
            "mention_id": mention_id,
            "phrase": mention.phrase,
            "start_char": mention.start,
            "end_char": mention.end,
            "stored_candidate_hpo_ids": source_by_id[mention_id].get("candidate_hpo_ids", []),
            "forced_payload_hpo_ids": [value.hpo_id for value in candidates],
            "decision": (
                decisions[mention_id].model_dump(mode="json") if mention_id in decisions else None
            ),
            "assertion": asdict(assertion),
        }
        for mention_id, mention, assertion, candidates in items
    ]


def extraction_control(
    rows: list[AnnotationInput],
    *,
    pipeline: TracePipeline,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        pipeline.current_patient = row.patient_id
        mentions = pipeline._extract_with_strict_chunk_fallback(
            note=row.clinical_note,
            row_index=row_index,
            system_message=pipeline.prompts["coverage_extraction"],
            stage="control-temp0-coverage-extract",
            method="model-pass-1-temp0",
            temperature=0.0,
        )
        output.append(
            {
                "patient_id": row.patient_id,
                "mentions": [mention_dict(value) for value in mentions],
            }
        )
    return output


def config_diff(old: Any, new: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(old, dict) and isinstance(new, dict):
        output: list[dict[str, Any]] = []
        for key in sorted(set(old) | set(new)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in old:
                output.append({"path": child, "old": "<missing>", "new": new[key]})
            elif key not in new:
                output.append({"path": child, "old": old[key], "new": "<missing>"})
            else:
                output.extend(config_diff(old[key], new[key], child))
        return output
    if old != new:
        return [{"path": prefix, "old": old, "new": new}]
    return []


def score_file(
    path: Path,
    *,
    references: dict[str, list[set[str]]],
    aliases: dict[str, str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for accepted_only in (False, True):
        policy = "accepted" if accepted_only else "all-mapped"
        try:
            groups = load_prediction_groups(path, aliases, accepted_only=accepted_only)
            case_scores = score_prediction_groups(
                groups,
                references,
                patient_ids=list(CASE_IDS),
            )
            output[f"candidate-set-{policy}"] = {
                "summary": summarize(case_scores),
                "per_case": [asdict(value) for value in case_scores],
            }
        except ValueError as exc:
            output[f"candidate-set-{policy}"] = {
                "not_comparable": str(exc),
                "reason": (
                    "This artifact predates bounded mapping alternatives and stores "
                    "the broad retrieval pool in candidate_hpo_ids."
                ),
            }
        selected = load_prediction_sets(path, aliases, accepted_only=accepted_only)
        case_scores = score_reference_groups(
            selected,
            references,
            patient_ids=list(CASE_IDS),
        )
        output[f"selected-id-{policy}"] = {
            "summary": summarize(case_scores),
            "per_case": [asdict(value) for value in case_scores],
        }
    return output


def merged_mentions_from_events(
    events: list[dict[str, Any]],
) -> dict[str, list[Mention]]:
    output: dict[str, list[Mention]] = {}
    for event in events:
        if event.get("stage") != "mapping-input":
            continue
        output[str(event["patient_id"])] = [
            Mention(
                phrase=str(value["phrase"]),
                start=int(value["start_char"]),
                end=int(value["end_char"]),
                methods=set(value.get("methods", [])),
                recognizer_ids=set(value.get("recognizer_hpo_ids", [])),
            )
            for value in event.get("mentions", [])
        ]
    return output


def all_candidate_score(
    results: list[AnnotationResult],
    *,
    references: dict[str, list[set[str]]],
    aliases: dict[str, str],
    writer: PrivateWriter,
) -> dict[str, Any]:
    rows = []
    for result in results:
        value = result.model_dump(mode="json")
        if value.get("candidate_hpo_ids"):
            value["mapping_status"] = "mapped"
            rows.append(value)
    path = writer.write_json("analysis/all-candidate-counterfactual.json", rows)
    groups = load_prediction_groups(path, aliases)
    return summarize(score_prediction_groups(groups, references, patient_ids=list(CASE_IDS)))


def ids_at(record: dict[str, Any], key: str, limit: int | None = None) -> set[str]:
    values = record.get(key, [])
    if limit is not None:
        values = values[:limit]
    return {str(value["hpo_id"]) for value in values}


def build_recall_checkpoints(
    retrieval: list[dict[str, Any]],
    mapper_payloads: list[dict[str, Any]],
    references: dict[str, list[set[str]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for patient_id in CASE_IDS:
        patient_records = [value for value in retrieval if value["patient_id"] == patient_id]
        normal_payloads = [
            value
            for value in mapper_payloads
            if value["patient_id"] == patient_id and not value["stage"].startswith("forced-")
        ]
        payload_ids = {
            hp_id
            for payload in normal_payloads
            for item in payload["items"]
            for hp_id in item["candidate_hpo_ids"]
        }
        rows: list[dict[str, Any]] = []
        for group in references.get(patient_id, []):
            row: dict[str, Any] = {"reference_group": sorted(group)}
            for lane, key in (
                ("dense", "dense_top64"),
                ("lexical", "lexical_top64"),
                ("fused", "fused_top64"),
            ):
                for k in (1, 8, 16, 32, 64):
                    present = any(
                        bool(group & ids_at(record, key, k)) for record in patient_records
                    )
                    row[f"{lane}@{k}"] = present
            row["bounded16"] = any(
                bool(group & ids_at(record, "bounded", 16)) for record in patient_records
            )
            row["mapper_payload"] = bool(group & payload_ids)
            rows.append(row)
        output[patient_id] = {
            "reference_rows": rows,
            "aggregate": {
                key: sum(bool(row[key]) for row in rows) / len(rows)
                for key in rows[0]
                if key != "reference_group"
            }
            if rows
            else {},
        }
    return output


def output_group_ids(result: AnnotationResult) -> set[str]:
    return set(result.candidate_hpo_ids or ([result.hpo_id] if result.hpo_id else []))


def build_loss_ledger(
    results: list[AnnotationResult],
    *,
    references: dict[str, list[set[str]]],
    retrieval: list[dict[str, Any]],
    mapper_payloads: list[dict[str, Any]],
    forced: list[dict[str, Any]],
    merged_mentions: dict[str, list[Mention]],
    merge_events: list[dict[str, Any]],
) -> dict[str, Any]:
    ledger: dict[str, Any] = {}
    for patient_id in CASE_IDS:
        case_results = [value for value in results if value.patient_id == patient_id]
        accepted_by_identity: dict[frozenset[str], AnnotationResult] = {}
        for value in case_results:
            identity = frozenset(output_group_ids(value))
            if value.mapping_status == "mapped" and value.review_status == "accepted" and identity:
                accepted_by_identity.setdefault(identity, value)
        accepted_results = list(accepted_by_identity.values())
        accepted_groups = [output_group_ids(value) for value in accepted_results]
        patient_retrieval = [value for value in retrieval if value["patient_id"] == patient_id]
        payload_ids = {
            hp_id
            for payload in mapper_payloads
            if payload["patient_id"] == patient_id and not payload["stage"].startswith("forced-")
            for item in payload["items"]
            for hp_id in item["candidate_hpo_ids"]
        }
        patient_forced = [value for value in forced if value["patient_id"] == patient_id]
        patient_mentions = merged_mentions[patient_id]
        shrinkage = [
            item
            for event in merge_events
            if event["patient_id"] == patient_id
            for suppressed in event.get("suppressed", [])
            for item in suppressed.get("shrinkage_candidates", [])
            if item.get("fraction_reduction", 0) > 0.2
        ]
        false_negatives = []
        for group in references.get(patient_id, []):
            if any(group & prediction for prediction in accepted_groups):
                continue
            recognizer_span = any(group & mention.recognizer_ids for mention in patient_mentions)
            dense_lexical = any(
                group & (ids_at(record, "dense_top64") | ids_at(record, "lexical_top64"))
                for record in patient_retrieval
            )
            bounded = any(group & ids_at(record, "bounded", 16) for record in patient_retrieval)
            in_payload = bool(group & payload_ids)
            deferred_rows = [
                value
                for value in case_results
                if value.mapping_status == "no_candidate_fit" and group & output_group_ids(value)
            ]
            forced_rows = [
                value
                for value in patient_forced
                if group & set(value["original_candidate_hpo_ids"])
            ]
            forced_selected = any(
                value.get("decision")
                and group & set(value["decision"].get("candidate_hpo_ids", []))
                for value in forced_rows
            )
            mapped_nonaccepted = [
                value for value in case_results if group & output_group_ids(value)
            ]
            if not (recognizer_span or dense_lexical):
                cause = "[1] Extraction Miss"
            elif shrinkage and not dense_lexical:
                cause = "[2] Span Alignment / Modifier Loss"
            elif not dense_lexical:
                cause = "[3] Vector / Lexical Retrieval Miss"
            elif not bounded:
                cause = "[4] Candidate Pool Truncation"
            elif deferred_rows or (bounded and not in_payload):
                cause = (
                    "[5] Premature Deferral" if forced_selected else "[6] Mapper Reasoning Error"
                )
            elif in_payload and not mapped_nonaccepted:
                cause = "[6] Mapper Reasoning Error"
            elif any(
                value.category is not None and value.category.value != "Abnormal"
                for value in mapped_nonaccepted
            ):
                cause = "[7] Category Assignment Loss"
            elif mapped_nonaccepted:
                cause = "[8] Threshold Exclusion"
            else:
                cause = "[9] Evaluation / Manual-Standard Disagreement"
            false_negatives.append(
                {
                    "reference_group": sorted(group),
                    "primary_cause": cause,
                    "recognizer_span": recognizer_span,
                    "dense_or_lexical_top64": dense_lexical,
                    "bounded16": bounded,
                    "mapper_payload": in_payload,
                    "deferred_result_count": len(deferred_rows),
                    "forced_mapper_selected_gold": forced_selected,
                }
            )
        matched_predictions: set[int] = set()
        for group in references.get(patient_id, []):
            for index, prediction in enumerate(accepted_groups):
                if index not in matched_predictions and group & prediction:
                    matched_predictions.add(index)
                    break
        false_positives = []
        for index, result in enumerate(accepted_results):
            if index in matched_predictions:
                continue
            if result.assertion_status in {"normal", "negated", "family_history"}:
                cause = "Assertion / experiencer error"
            elif set(result.source_methods or []) <= {"native", "fasthpocr"}:
                cause = "Unsupported recognizer extraction"
            else:
                cause = "Mapping error or plausible unannotated finding"
            false_positives.append(
                {
                    "phrase": result.phrase,
                    "candidate_hpo_ids": sorted(output_group_ids(result)),
                    "primary_cause": cause,
                }
            )
        ledger[patient_id] = {
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "false_negative_cause_counts": dict(
                Counter(value["primary_cause"] for value in false_negatives)
            ),
            "false_positive_cause_counts": dict(
                Counter(value["primary_cause"] for value in false_positives)
            ),
        }
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--fasthpocr-index", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--historical-results", type=Path, required=True)
    parser.add_argument("--regression-results", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--regression-manifest", type=Path, required=True)
    parser.add_argument("--historical-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Reuse completed private traces without making provider calls.",
    )
    parser.add_argument(
        "--force-stored-deferred-only",
        action="store_true",
        help="Map only the stored regression's deferred Case 99 rows.",
    )
    args = parser.parse_args()

    writer = PrivateWriter(args.output_dir)
    for child in (
        "run-a-reproduction",
        "run-b-temp0-extraction",
        "offline-fasthpocr",
        "forced-deferred-mapping",
        "analysis",
        "reports",
    ):
        writer.directory(child)

    rows, baseline = load_cases(args.input)
    writer.write_json("input-baseline.json", baseline)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for audit provenance")
    source_commit = subprocess.run(  # noqa: S603  # nosec B603
        [git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_status = subprocess.run(  # noqa: S603  # nosec B603
        [git, "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    writer.write_json(
        "preflight.json",
        {
            "source_commit": source_commit,
            "expected_comparison_commit": "0038b64",
            "commit_matches_prefix": source_commit.startswith("0038b64"),
            "source_status": source_status,
            "case_ids": list(CASE_IDS),
            "vector_manifest_sha256": sha256_file(args.vector_dir / "hpo_manifest.json"),
            "ontology_sha256": sha256_file(args.ontology),
            "prompt_bundle_sha256": sha256_file(
                Path(__file__).parents[1] / "src" / "rag_hpo" / "data" / "system_prompts.json"
            ),
            "temperatures": {
                "run_a_extraction": 0.2,
                "run_a_other": 0.0,
                "run_b_extraction_only": 0.0,
            },
            "token_limits": {
                "input_warning": INPUT_TOKEN_WARNING,
                "input_ceiling": INPUT_TOKEN_CEILING,
                "total_audit_ceiling": TOTAL_TOKEN_CEILING,
            },
        },
    )

    if args.force_stored_deferred_only:
        provider = TracingProvider(ProviderConfig.from_env(), writer=writer)
        pipeline = TracePipeline(
            provider=None,
            vector_dir=args.vector_dir,
            output_dir=writer.directory("forced-stored-deferred"),
            mode=AnnotationMode.NATIVE,
            recognizers={"native"},
            fasthpocr_index=None,
            resume=False,
            keep_state=False,
            keep_raw_responses=False,
            include_evidence_text=False,
            offline=True,
            confidence_calibration=None,
            mapping_prompt=MappingPromptMode.ZERO_SHOT,
            writer=writer,
        )
        pipeline.provider = provider
        try:
            forced_stored = force_stored_deferred(
                rows,
                args.regression_results,
                pipeline=pipeline,
            )
        finally:
            provider.close()
        writer.write_json("forced-stored-deferred/results.json", forced_stored)
        existing = (
            json.loads((writer.root / "provider-summary.json").read_text(encoding="utf-8"))
            if (writer.root / "provider-summary.json").is_file()
            else []
        )
        writer.write_json("provider-summary.json", [*existing, *provider.call_summaries])
        print(
            json.dumps(
                {
                    "forced_rows": len(forced_stored),
                    "new_provider_calls": len(provider.call_summaries),
                    "cumulative_provider_usage": provider.usage,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.analysis_only:
        run_a_results = [
            AnnotationResult.model_validate(value)
            for value in json.loads(
                (writer.root / "run-a-reproduction" / "rag_hpo_results.json").read_text(
                    encoding="utf-8"
                )
            )
        ]
        control = json.loads(
            (writer.root / "run-b-temp0-extraction" / "results.json").read_text(encoding="utf-8")
        )
        forced = json.loads(
            (writer.root / "forced-deferred-mapping" / "results.json").read_text(encoding="utf-8")
        )
        fast_overlay = json.loads(
            (writer.root / "offline-fasthpocr" / "results.json").read_text(encoding="utf-8")
        )
        events = json.loads((writer.root / "stage-events.json").read_text(encoding="utf-8"))
        mapper_payloads = json.loads(
            (writer.root / "mapper-payloads.json").read_text(encoding="utf-8")
        )
        retrieval = json.loads(
            (writer.root / "retrieval-checkpoints.json").read_text(encoding="utf-8")
        )
        provider_summaries = json.loads(
            (writer.root / "provider-summary.json").read_text(encoding="utf-8")
        )
        provider_usage = {
            "requests": len(list((writer.root / "provider").glob("call-*"))),
            "input_tokens": sum(
                int(value.get("prompt_tokens") or 0) for value in provider_summaries
            ),
            "output_tokens": sum(
                int(value.get("completion_tokens") or 0) for value in provider_summaries
            ),
            "total_tokens": sum(
                int(value.get("total_tokens") or 0) for value in provider_summaries
            ),
        }
        merged_mentions = merged_mentions_from_events(events)
    else:
        provider = TracingProvider(ProviderConfig.from_env(), writer=writer)
        pipeline = TracePipeline(
            provider=provider,
            vector_dir=args.vector_dir,
            output_dir=writer.directory("run-a-reproduction"),
            mode=AnnotationMode.BALANCED,
            recognizers={"native"},
            fasthpocr_index=None,
            resume=False,
            keep_state=False,
            keep_raw_responses=False,
            include_evidence_text=False,
            offline=False,
            confidence_calibration=args.calibration,
            mapping_prompt=MappingPromptMode.ZERO_SHOT,
            writer=writer,
        )
        try:
            run_a_results = pipeline.run(rows)
            control = extraction_control(rows, pipeline=pipeline)
            forced = force_deferred(rows, run_a_results, pipeline=pipeline)
            fast_overlay = run_fast_overlay(
                rows,
                pipeline=pipeline,
                index_path=args.fasthpocr_index,
            )
        finally:
            provider.close()
        events = pipeline.events
        mapper_payloads = pipeline.mapper_payloads
        retrieval = pipeline.trace_retriever.calls if pipeline.trace_retriever else []
        provider_summaries = provider.call_summaries
        provider_usage = provider.usage
        merged_mentions = pipeline.merged_mentions
        writer.write_json("run-b-temp0-extraction/results.json", control)
        writer.write_json("forced-deferred-mapping/results.json", forced)
        writer.write_json("offline-fasthpocr/results.json", fast_overlay)
        writer.write_json("stage-events.json", events)
        writer.write_json("mapper-payloads.json", mapper_payloads)
        writer.write_json("provider-summary.json", provider_summaries)
        writer.write_json("retrieval-checkpoints.json", retrieval)

    # Gold-standard data enters only after every provider call has completed.
    aliases = load_hpo_aliases(args.ontology)
    references = load_reference_groups(args.references, aliases)
    references = {key: references[key] for key in CASE_IDS}
    scores = {
        "historical": score_file(
            args.historical_results,
            references=references,
            aliases=aliases,
        ),
        "stored_regression": score_file(
            args.regression_results,
            references=references,
            aliases=aliases,
        ),
        "audit_reproduction": score_file(
            writer.root / "run-a-reproduction" / "rag_hpo_results.json",
            references=references,
            aliases=aliases,
        ),
        "audit_all_candidate_counterfactual": all_candidate_score(
            run_a_results,
            references=references,
            aliases=aliases,
            writer=writer,
        ),
    }
    ontology = parse_obo(args.ontology)
    selected = load_prediction_sets(
        writer.root / "run-a-reproduction" / "rag_hpo_results.json",
        aliases,
        accepted_only=True,
    )
    scores["hierarchy_sensitivity"] = {
        patient_id: {
            f"distance_{distance}": asdict(
                score_layered_case(
                    patient_id,
                    selected.get(patient_id, set()),
                    references.get(patient_id, []),
                    ontology,
                    max_distance=distance,
                )
            )
            for distance in (0, 1, 2)
        }
        for patient_id in CASE_IDS
    }
    writer.write_json("analysis/scores.json", scores)

    checkpoints = build_recall_checkpoints(
        retrieval,
        mapper_payloads,
        references,
    )
    writer.write_json("analysis/retrieval-recall.json", checkpoints)
    merge_events = [value for value in events if value["stage"] == "mention-merge"]
    ledger = build_loss_ledger(
        run_a_results,
        references=references,
        retrieval=retrieval,
        mapper_payloads=mapper_payloads,
        forced=forced,
        merged_mentions=merged_mentions,
        merge_events=merge_events,
    )
    writer.write_json("analysis/loss-ledger.json", ledger)

    run_a_pass1 = {
        patient_id: [
            mention
            for event in events
            if event["patient_id"] == patient_id and event["stage"] == "coverage-extract"
            for mention in event["validated_mentions"]
        ]
        for patient_id in CASE_IDS
    }
    run_b_pass1 = {value["patient_id"]: value["mentions"] for value in control}
    determinism = {
        patient_id: {
            "temperature_0_2_count": len(run_a_pass1[patient_id]),
            "temperature_0_0_count": len(run_b_pass1[patient_id]),
            "temperature_0_2_spans": run_a_pass1[patient_id],
            "temperature_0_0_spans": run_b_pass1[patient_id],
            "identical_span_tuples": {
                (value["phrase"], value["start_char"], value["end_char"])
                for value in run_a_pass1[patient_id]
            }
            == {
                (value["phrase"], value["start_char"], value["end_char"])
                for value in run_b_pass1[patient_id]
            },
        }
        for patient_id in CASE_IDS
    }
    writer.write_json("analysis/determinism.json", determinism)

    old_manifest = json.loads(args.historical_manifest.read_text(encoding="utf-8"))
    new_manifest = json.loads(args.regression_manifest.read_text(encoding="utf-8"))
    writer.write_json("analysis/config-diff.json", config_diff(old_manifest, new_manifest))
    current_prompt = Path(__file__).parents[1] / "src" / "rag_hpo" / "data" / "system_prompts.json"
    prompt_diff = "".join(
        difflib.unified_diff(
            args.historical_prompt.read_text(encoding="utf-8").splitlines(keepends=True),
            current_prompt.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=str(args.historical_prompt),
            tofile=str(current_prompt),
        )
    )
    writer.write_text("analysis/prompt-schema.diff", prompt_diff)

    loss_counts = Counter(
        item["primary_cause"] for patient in ledger.values() for item in patient["false_negatives"]
    )
    report = [
        "# CSC Case 1 and Case 99 Root-Cause Audit",
        "",
        f"- Source commit: `{source_commit}`",
        f"- Provider calls: {provider_usage['requests']}",
        f"- Provider tokens: {provider_usage['total_tokens']}",
        f"- Case 1 pass-one spans: {determinism['1']['temperature_0_2_count']} at "
        f"temperature 0.2 and {determinism['1']['temperature_0_0_count']} at 0.0.",
        f"- Case 99 pass-one spans: {determinism['99']['temperature_0_2_count']} at "
        f"temperature 0.2 and {determinism['99']['temperature_0_0_count']} at 0.0.",
        "",
        "## False-negative primary causes",
        "",
    ]
    report.extend(f"- {key}: {value}" for key, value in sorted(loss_counts.items()))
    report.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Strict exact scoring remains primary.",
            "- Candidate-set and hierarchy-aware results are sensitivity analyses.",
            "- A forced mapper decision was made without access to the gold standard.",
            "- Raw notes, requests, and responses remain in this private directory.",
            "",
            "See `analysis/loss-ledger.json`, `analysis/retrieval-recall.json`, "
            "`analysis/scores.json`, and `analysis/determinism.json` for exact evidence.",
            "",
        ]
    )
    writer.write_text("reports/ROOT_CAUSE_AUDIT.md", "\n".join(report))
    print(
        json.dumps(
            {
                "output_dir": str(writer.root),
                "provider_usage": provider_usage,
                "loss_counts": dict(loss_counts),
                "determinism_counts": {
                    key: {
                        "temperature_0_2": value["temperature_0_2_count"],
                        "temperature_0_0": value["temperature_0_0_count"],
                    }
                    for key, value in determinism.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
