from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

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
SPARSE_WEIGHT = 0.8
FUZZY_WEIGHT = 0.2


@dataclass(frozen=True)
class RankedCandidate:
    hpo_id: str
    dense_rank: int | None
    dense_score: float | None
    lexical_rank: int | None
    lexical_score: float | None
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
        raw_limit: int = 64,
    ) -> None:
        self.registry = registry
        self.entries = entries
        self.backend = backend
        self.raw_limit = raw_limit
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
    ) -> list[list[Candidate]]:
        if distinct_limit <= 0:
            raise ValueError("distinct candidate limit must be positive")
        if not phrases:
            return []
        queries = self.backend.encode(phrases)
        if queries.ndim != 2 or queries.shape[1] != self._dense_index.d:
            raise ValueError("query embedding dimension does not match vector artifacts")
        raw_limit = min(self.raw_limit, self._dense_index.ntotal)
        scores, indices = self._dense_index.search(
            np.asarray(queries, dtype=np.float32),
            raw_limit,
        )
        sparse_limit = min(self.raw_limit, len(self._lexical_phrases))
        sparse_queries = self._sparse_vectorizer.transform(phrases)
        sparse_distances, sparse_indices = self._sparse_index.kneighbors(
            sparse_queries,
            n_neighbors=sparse_limit,
        )
        return [
            self._candidates(
                self._dense_from_search(scores[index], indices[index]),
                self._lexical(
                    phrase,
                    sparse_distances=sparse_distances[index],
                    sparse_indices=sparse_indices[index],
                ),
                distinct_limit=distinct_limit,
            )
            for index, phrase in enumerate(phrases)
        ]

    def _candidates(
        self,
        dense: dict[str, tuple[int, float]],
        lexical: dict[str, tuple[int, float]],
        *,
        distinct_limit: int,
    ) -> list[Candidate]:
        ranked = self._fuse(dense, lexical)
        output: list[Candidate] = []
        for item in ranked[:distinct_limit]:
            concept = self._concepts[item.hpo_id]
            methods: list[str] = []
            if item.dense_rank is not None:
                methods.append("sapbert")
            if item.lexical_rank is not None:
                methods.append("lexical")
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

        matches = process.extract(
            normalized,
            self._lexical_phrases,
            scorer=fuzz.WRatio,
            limit=min(self.raw_limit, len(self._lexical_phrases)),
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
    ) -> list[RankedCandidate]:
        output: list[RankedCandidate] = []
        for hp_id in sorted(set(dense) | set(lexical)):
            dense_value = dense.get(hp_id)
            lexical_value = lexical.get(hp_id)
            combined = 0.0
            if dense_value is not None:
                combined += DENSE_WEIGHT / (RRF_CONSTANT + dense_value[0])
            if lexical_value is not None:
                combined += LEXICAL_WEIGHT / (RRF_CONSTANT + lexical_value[0])
            output.append(
                RankedCandidate(
                    hpo_id=hp_id,
                    dense_rank=dense_value[0] if dense_value else None,
                    dense_score=dense_value[1] if dense_value else None,
                    lexical_rank=lexical_value[0] if lexical_value else None,
                    lexical_score=lexical_value[1] if lexical_value else None,
                    combined_score=combined,
                )
            )
        return sorted(
            output,
            key=lambda item: (
                -item.combined_score,
                item.dense_rank or 10**9,
                item.lexical_rank or 10**9,
                item.hpo_id,
            ),
        )
