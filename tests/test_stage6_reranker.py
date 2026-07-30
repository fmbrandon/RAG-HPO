from __future__ import annotations

from rag_hpo.models import Candidate
from rag_hpo.reranker import CrossEncoderReranker


def test_cross_encoder_reranker_exact_match_boost() -> None:
    reranker = CrossEncoderReranker(enabled=True)
    candidates = [
        Candidate(hpo_id="HP:0001083", term="Ectopia lentis", score=0.8),
        Candidate(hpo_id="HP:0000543", term="Spherophakia", score=0.9),
    ]
    reranked = reranker.rerank("Spherophakia", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].hpo_id == "HP:0000543"


def test_cross_encoder_reranker_disabled_fallback() -> None:
    reranker = CrossEncoderReranker(enabled=False)
    candidates = [
        Candidate(hpo_id="HP:0001083", term="Ectopia lentis", score=0.8),
        Candidate(hpo_id="HP:0000543", term="Spherophakia", score=0.9),
    ]
    reranked = reranker.rerank("Spherophakia", candidates, top_k=2)
    assert reranked[0].hpo_id == "HP:0001083"
