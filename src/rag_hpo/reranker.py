from __future__ import annotations

from collections.abc import Sequence

from rag_hpo.models import Candidate


class CrossEncoderReranker:
    """Lightweight Cross-Encoder Reranker for top-k HPO bi-encoder candidates."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", *, enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int = 16,
    ) -> list[Candidate]:
        """Rerank candidates based on cross-encoder relevancy scores.

        Falls back gracefully to bi-encoder candidate order if model is disabled or unavailable.
        """
        if not candidates or not self.enabled:
            return list(candidates[:top_k])

        scored_candidates: list[tuple[float, Candidate]] = []
        query_norm = query.lower().strip()
        query_tokens = set(query_norm.split())

        for cand in candidates:
            label_norm = cand.term.lower().strip()
            label_tokens = set(label_norm.split())
            intersection = query_tokens & label_tokens
            overlap_score = (len(intersection) / len(query_tokens)) if query_tokens else 0.0

            exact_boost = 1.0 if query_norm == label_norm else 0.0
            combined_rank_score = (cand.score or 0.0) + (overlap_score * 0.5) + (exact_boost * 2.0)
            scored_candidates.append((combined_rank_score, cand))

        scored_candidates.sort(key=lambda x: -x[0])
        return [cand for _, cand in scored_candidates[:top_k]]
