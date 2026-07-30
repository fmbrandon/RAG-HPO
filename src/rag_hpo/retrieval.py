from __future__ import annotations

import hashlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

from rag_hpo.artifacts import ArtifactEntry
from rag_hpo.embeddings import EmbeddingBackend
from rag_hpo.models import Candidate
from rag_hpo.registry import HPORegistry, normalize_phrase

RRF_CONSTANT = 60
DENSE_WEIGHT = 0.8
LEXICAL_WEIGHT = 0.2
# A context-only anatomy match must be able to enter the bounded set even when
# the short phrase itself strongly retrieves many wrong-organ homonyms.
CONTEXT_DENSE_WEIGHT = 0.8
SPARSE_WEIGHT = 0.8
FUZZY_WEIGHT = 0.2
RETRIEVAL_POLICY_VERSION = "2.0"


@dataclass(frozen=True)
class RankedCandidate:
    hpo_id: str
    dense_rank: int | None
    dense_score: float | None
    lexical_rank: int | None
    lexical_score: float | None
    context_dense_rank: int | None
    context_dense_score: float | None
    combined_score: float


class HybridCandidateRetriever:
    """Deterministic active-ID fusion of dense and lexical HPO retrieval."""

    def __init__(
        self,
        *,
        registry: HPORegistry,
        entries: list[ArtifactEntry],
        matrix: np.ndarray,
        backend: EmbeddingBackend,
        raw_limit: int = 128,
        cache_size: int = 4096,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.registry = registry
        self.entries = entries
        self.backend = backend
        self.raw_limit = raw_limit
        self.cache_size = cache_size
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._candidate_cache: OrderedDict[
            tuple[str, str, int],
            tuple[Candidate, ...],
        ] = OrderedDict()
        self._cache_hits = {"embeddings": 0, "candidates": 0}
        self._cache_misses = {"embeddings": 0, "candidates": 0}
        self._concepts = {
            concept.hp_id: concept for concept in registry.concepts if not concept.obsolete
        }
        self._active = set(self._concepts)
        self._aliases = {
            alternate: concept.hp_id
            for concept in registry.concepts
            for alternate in concept.alternate_ids
        }
        self._aliases.update({hp_id: hp_id for hp_id in self._active})

        dense_rows = [
            index
            for index, entry in enumerate(entries)
            if self._aliases.get(entry.hp_id, entry.hp_id) in self._active
        ]
        if not dense_rows:
            raise ValueError("vector artifacts contain no active registry IDs")
        self._dense_rows = np.asarray(dense_rows, dtype=np.int64)
        dense_matrix = np.asarray(matrix[self._dense_rows], dtype=np.float32)
        self._dense_index = faiss.IndexFlatIP(dense_matrix.shape[1])
        self._dense_index.add(dense_matrix)

        lexical_ids: dict[str, set[str]] = defaultdict(set)
        for concept in self._concepts.values():
            lexical_ids[normalize_phrase(concept.label)].add(concept.hp_id)
            for phrase in concept.phrases:
                lexical_ids[phrase.normalized_phrase].add(concept.hp_id)
        for extension in registry.extensions:
            if extension.hp_id in self._active:
                lexical_ids[extension.normalized_phrase].add(extension.hp_id)
        self._lexical_phrases = sorted(lexical_ids)
        self._lexical_ids = {
            phrase: tuple(sorted(hp_ids)) for phrase, hp_ids in lexical_ids.items()
        }
        self._sparse_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            norm="l2",
            dtype=np.float32,
        )
        sparse_matrix = self._sparse_vectorizer.fit_transform(self._lexical_phrases)
        self._sparse_index = NearestNeighbors(
            algorithm="brute",
            metric="cosine",
            n_jobs=1,
        ).fit(sparse_matrix)

    def retrieve(self, phrase: str, *, distinct_limit: int) -> list[Candidate]:
        return self.retrieve_many([phrase], distinct_limit=distinct_limit)[0]

    def retrieve_many(
        self,
        phrases: list[str],
        *,
        distinct_limit: int,
        contexts: list[str | None] | None = None,
    ) -> list[list[Candidate]]:
        if distinct_limit <= 0:
            raise ValueError("distinct candidate limit must be positive")
        if not phrases:
            return []
        if contexts is not None and len(contexts) != len(phrases):
            raise ValueError("retrieval contexts must align with phrases")
        aligned_contexts = contexts or [None] * len(phrases)
        keys = [
            self._candidate_key(phrase, context, distinct_limit)
            for phrase, context in zip(phrases, aligned_contexts, strict=True)
        ]
        output: list[list[Candidate] | None] = [None] * len(phrases)
        pending: OrderedDict[
            tuple[str, str, int],
            tuple[str, str | None],
        ] = OrderedDict()
        for index, (key, phrase, context) in enumerate(
            zip(keys, phrases, aligned_contexts, strict=True)
        ):
            cached = self._candidate_cache.get(key)
            if cached is None:
                self._cache_misses["candidates"] += 1
                pending.setdefault(key, (phrase, context))
                continue
            self._cache_hits["candidates"] += 1
            self._candidate_cache.move_to_end(key)
            output[index] = list(cached)

        if pending:
            pending_values = list(pending.values())
            computed = self._retrieve_uncached(
                [value[0] for value in pending_values],
                distinct_limit=distinct_limit,
                contexts=[value[1] for value in pending_values],
            )
            for key, candidates in zip(pending, computed, strict=True):
                self._remember(self._candidate_cache, key, tuple(candidates))

        for index, key in enumerate(keys):
            if output[index] is None:
                output[index] = list(self._candidate_cache[key])
        return [value for value in output if value is not None]

    def _retrieve_uncached(
        self,
        phrases: list[str],
        *,
        distinct_limit: int,
        contexts: list[str | None],
    ) -> list[list[Candidate]]:
        context_rows = [
            (index, value)
            for index, value in enumerate(contexts)
            if value and normalize_phrase(value) != normalize_phrase(phrases[index])
        ]
        queries = self._encode_cached([*phrases, *(value for _index, value in context_rows)])
        if queries.ndim != 2 or queries.shape[1] != self._dense_index.d:
            raise ValueError("query embedding dimension does not match vector artifacts")
        raw_limit = min(self.raw_limit, self._dense_index.ntotal)
        phrase_count = len(phrases)
        phrase_scores, phrase_indices = self._dense_index.search(
            np.asarray(queries[:phrase_count], dtype=np.float32),
            raw_limit,
        )
        context_by_phrase: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if context_rows:
            context_scores, context_indices = self._dense_index.search(
                np.asarray(queries[phrase_count:], dtype=np.float32),
                raw_limit,
            )
            context_by_phrase = {
                phrase_index: (context_scores[index], context_indices[index])
                for index, (phrase_index, _value) in enumerate(context_rows)
            }
        sparse_limit = min(self.raw_limit, len(self._lexical_phrases))
        sparse_queries = self._sparse_vectorizer.transform(phrases)
        sparse_distances, sparse_indices = self._sparse_index.kneighbors(
            sparse_queries,
            n_neighbors=sparse_limit,
        )
        return [
            self._candidates(
                self._dense_from_search(phrase_scores[index], phrase_indices[index]),
                self._lexical(
                    phrase,
                    sparse_distances=sparse_distances[index],
                    sparse_indices=sparse_indices[index],
                ),
                (
                    self._dense_from_search(*context_by_phrase[index])
                    if index in context_by_phrase
                    else {}
                ),
                distinct_limit=distinct_limit,
            )
            for index, phrase in enumerate(phrases)
        ]

    def _encode_cached(self, texts: list[str]) -> np.ndarray:
        keys = [self._text_key(value) for value in texts]
        missing: OrderedDict[str, str] = OrderedDict()
        for key, value in zip(keys, texts, strict=True):
            if key in self._embedding_cache:
                self._cache_hits["embeddings"] += 1
                self._embedding_cache.move_to_end(key)
            else:
                self._cache_misses["embeddings"] += 1
                missing.setdefault(key, value)
        if missing:
            encoded = self.backend.encode(list(missing.values()))
            if encoded.ndim != 2 or encoded.shape[0] != len(missing):
                raise ValueError("embedding backend returned an unexpected batch shape")
            for key, row in zip(missing, encoded, strict=True):
                self._remember(
                    self._embedding_cache,
                    key,
                    np.asarray(row, dtype=np.float32).copy(),
                )
        return np.stack([self._embedding_cache[key] for key in keys])

    @staticmethod
    def _text_key(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _candidate_key(
        cls,
        phrase: str,
        context: str | None,
        distinct_limit: int,
    ) -> tuple[str, str, int]:
        return (
            cls._text_key(phrase),
            cls._text_key(context) if context else "",
            distinct_limit,
        )

    def _remember(
        self,
        cache: OrderedDict[Any, Any],
        key: Any,
        value: Any,
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def cache_info(self) -> dict[str, int]:
        return {
            "embedding_hits": self._cache_hits["embeddings"],
            "embedding_misses": self._cache_misses["embeddings"],
            "embedding_entries": len(self._embedding_cache),
            "candidate_hits": self._cache_hits["candidates"],
            "candidate_misses": self._cache_misses["candidates"],
            "candidate_entries": len(self._candidate_cache),
        }

    def _candidates(
        self,
        dense: dict[str, tuple[int, float]],
        lexical: dict[str, tuple[int, float]],
        context_dense: dict[str, tuple[int, float]],
        *,
        distinct_limit: int,
    ) -> list[Candidate]:
        ranked = self._fuse(dense, lexical, context_dense)
        output: list[Candidate] = []
        for item in ranked[:distinct_limit]:
            concept = self._concepts[item.hpo_id]
            methods: list[str] = []
            if item.dense_rank is not None:
                methods.append("sapbert")
            if item.lexical_rank is not None:
                methods.append("lexical")
            if item.context_dense_rank is not None:
                methods.append("context-sapbert")
            output.append(
                Candidate(
                    hpo_id=item.hpo_id,
                    term=concept.label,
                    score=round(item.combined_score, 8),
                    definition=concept.definition[:500] or None,
                    synonyms=sorted({value.phrase for value in concept.phrases})[:12],
                    parents=concept.parents,
                    dense_rank=item.dense_rank,
                    dense_score=item.dense_score,
                    lexical_rank=item.lexical_rank,
                    lexical_score=item.lexical_score,
                    context_dense_rank=item.context_dense_rank,
                    context_dense_score=item.context_dense_score,
                    source_methods=methods,
                )
            )
        return output

    def _dense_from_search(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> dict[str, tuple[int, float]]:
        result: dict[str, tuple[int, float]] = {}
        for raw_rank, (score, dense_index) in enumerate(
            zip(scores, indices, strict=True),
            start=1,
        ):
            if dense_index < 0:
                continue
            entry = self.entries[int(self._dense_rows[int(dense_index)])]
            hp_id = self._aliases.get(entry.hp_id, entry.hp_id)
            if hp_id in self._active and hp_id not in result:
                result[hp_id] = (raw_rank, float(score))
        return result

    def _lexical(
        self,
        phrase: str,
        *,
        sparse_distances: np.ndarray,
        sparse_indices: np.ndarray,
    ) -> dict[str, tuple[int, float]]:
        normalized = normalize_phrase(phrase)
        sparse: dict[str, tuple[int, float]] = {}
        sparse_rank = 0
        for distance, sparse_index in zip(
            sparse_distances,
            sparse_indices,
            strict=True,
        ):
            matched_phrase = self._lexical_phrases[int(sparse_index)]
            for hp_id in self._lexical_ids[matched_phrase]:
                if hp_id in sparse:
                    continue
                sparse_rank += 1
                sparse[hp_id] = (sparse_rank, float(1.0 - distance))

        fuzzy_candidates = list(
            dict.fromkeys(self._lexical_phrases[int(idx)] for idx in sparse_indices)
        )
        matches = process.extract(
            normalized,
            fuzzy_candidates,
            scorer=fuzz.WRatio,
            limit=min(self.raw_limit, len(fuzzy_candidates)),
        )
        fuzzy: dict[str, tuple[int, float]] = {}
        rank = 0
        for matched_phrase, score, _index in matches:
            for hp_id in self._lexical_ids[str(matched_phrase)]:
                if hp_id in fuzzy:
                    continue
                rank += 1
                fuzzy[hp_id] = (rank, float(score))

        exact_ids = self._lexical_ids.get(normalized, ())
        scored: list[tuple[str, float, float]] = []
        for hp_id in sorted(set(sparse) | set(fuzzy)):
            combined = 0.0
            if hp_id in sparse:
                combined += SPARSE_WEIGHT / (RRF_CONSTANT + sparse[hp_id][0])
            if hp_id in fuzzy:
                combined += FUZZY_WEIGHT / (RRF_CONSTANT + fuzzy[hp_id][0])
            lexical_score = max(
                sparse.get(hp_id, (0, 0.0))[1] * 100.0,
                fuzzy.get(hp_id, (0, 0.0))[1],
            )
            scored.append((hp_id, combined, lexical_score))
        scored.sort(
            key=lambda value: (
                -int(value[0] in exact_ids),
                -value[1],
                -value[2],
                value[0],
            )
        )
        result: dict[str, tuple[int, float]] = {}
        for lexical_rank, (hp_id, _combined, lexical_score) in enumerate(
            scored[: self.raw_limit],
            start=1,
        ):
            result[hp_id] = (lexical_rank, lexical_score)
        return result

    @staticmethod
    def _fuse(
        dense: dict[str, tuple[int, float]],
        lexical: dict[str, tuple[int, float]],
        context_dense: dict[str, tuple[int, float]],
    ) -> list[RankedCandidate]:
        output: list[RankedCandidate] = []
        for hp_id in sorted(set(dense) | set(lexical) | set(context_dense)):
            dense_value = dense.get(hp_id)
            lexical_value = lexical.get(hp_id)
            context_value = context_dense.get(hp_id)
            combined = 0.0
            if dense_value is not None:
                combined += DENSE_WEIGHT / (RRF_CONSTANT + dense_value[0])
            if lexical_value is not None:
                combined += LEXICAL_WEIGHT / (RRF_CONSTANT + lexical_value[0])
            if context_value is not None:
                combined += CONTEXT_DENSE_WEIGHT / (RRF_CONSTANT + context_value[0])
            output.append(
                RankedCandidate(
                    hpo_id=hp_id,
                    dense_rank=dense_value[0] if dense_value else None,
                    dense_score=dense_value[1] if dense_value else None,
                    lexical_rank=lexical_value[0] if lexical_value else None,
                    lexical_score=lexical_value[1] if lexical_value else None,
                    context_dense_rank=context_value[0] if context_value else None,
                    context_dense_score=context_value[1] if context_value else None,
                    combined_score=combined,
                )
            )
        return sorted(
            output,
            key=lambda item: (
                -item.combined_score,
                item.dense_rank or 10**9,
                item.lexical_rank or 10**9,
                item.context_dense_rank or 10**9,
                item.hpo_id,
            ),
        )
