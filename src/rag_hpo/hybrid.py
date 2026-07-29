from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from rag_hpo.assertion import AssertionDecision, analyze_assertion
from rag_hpo.prompts import load_prompts
from rag_hpo.registry import HPORegistry


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionReview(_StrictModel):
    mention_id: str
    verdict: Literal["supported", "unsupported", "ambiguous"]
    confidence: Literal["high", "medium", "low"]


class PredictionReviewBatch(_StrictModel):
    decisions: list[PredictionReview]


class ReviewProvider(Protocol):
    def request(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[BaseModel],
        temperature: float = 0.2,
    ) -> tuple[Any, str]: ...


@dataclass(frozen=True)
class ModelFinding:
    phrase: str
    hpo_id: str
    vector_score: float | None = None


@dataclass(frozen=True)
class HybridDecision:
    mention_id: str
    phrase: str
    hpo_id: str
    status: Literal["accepted", "rejected", "retained_ambiguous"]
    reason: str
    confidence: str
    assertion: AssertionDecision | None


@dataclass(frozen=True)
class HybridResult:
    decisions: tuple[HybridDecision, ...]
    verification_call_count: int
    reviewed_count: int
    approximate_input_tokens: int


def _locate(text: str, phrase: str) -> tuple[int, int] | None:
    match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
    return (match.start(), match.end()) if match else None


class HybridPredictionVerifier:
    """Precision filter for model predictions with recall-preserving abstention."""

    def __init__(
        self,
        registry: HPORegistry,
        *,
        provider: ReviewProvider | None,
    ) -> None:
        self.provider = provider
        self._concepts = {
            concept.hp_id: concept for concept in registry.concepts if not concept.obsolete
        }

    def verify(
        self,
        note: str,
        findings: list[ModelFinding],
        *,
        high_confidence_ids: set[str],
    ) -> HybridResult:
        unique: dict[str, ModelFinding] = {}
        for finding in findings:
            if finding.hpo_id not in self._concepts:
                raise ValueError(f"model finding uses unknown active ID {finding.hpo_id}")
            unique.setdefault(finding.hpo_id, finding)

        decisions: list[HybridDecision] = []
        review_items: list[dict[str, Any]] = []
        review_findings: dict[str, tuple[ModelFinding, AssertionDecision | None]] = {}
        for index, finding in enumerate(unique.values()):
            mention_id = f"m{index:04d}"
            located = _locate(note, finding.phrase)
            assertion = analyze_assertion(note, *located) if located is not None else None
            if finding.hpo_id in high_confidence_ids and (assertion is None or assertion.accepted):
                decisions.append(
                    HybridDecision(
                        mention_id=mention_id,
                        phrase=finding.phrase,
                        hpo_id=finding.hpo_id,
                        status="accepted",
                        reason="model_fasthpocr_agreement",
                        confidence="high",
                        assertion=assertion,
                    )
                )
                continue
            if assertion is not None and not assertion.accepted:
                decisions.append(
                    HybridDecision(
                        mention_id=mention_id,
                        phrase=finding.phrase,
                        hpo_id=finding.hpo_id,
                        status="retained_ambiguous",
                        reason=f"assertion_{assertion.status}_retained",
                        confidence="low",
                        assertion=assertion,
                    )
                )
                continue
            if located is None:
                decisions.append(
                    HybridDecision(
                        mention_id=mention_id,
                        phrase=finding.phrase,
                        hpo_id=finding.hpo_id,
                        status="retained_ambiguous",
                        reason="phrase_not_located",
                        confidence="low",
                        assertion=None,
                    )
                )
                continue
            concept = self._concepts[finding.hpo_id]
            review_items.append(
                {
                    "mention_id": mention_id,
                    "phrase": finding.phrase,
                    "sentence_context": (assertion.sentence if assertion is not None else ""),
                    "selected_hpo": {
                        "hpo_id": finding.hpo_id,
                        "label": concept.label,
                        "definition": concept.definition[:500],
                        "parents": concept.parents,
                    },
                    "vector_score": finding.vector_score,
                }
            )
            review_findings[mention_id] = (finding, assertion)

        if not review_items or self.provider is None:
            for mention_id, (finding, assertion) in review_findings.items():
                decisions.append(
                    HybridDecision(
                        mention_id=mention_id,
                        phrase=finding.phrase,
                        hpo_id=finding.hpo_id,
                        status="retained_ambiguous",
                        reason="verification_unavailable",
                        confidence="low",
                        assertion=assertion,
                    )
                )
            return HybridResult(
                decisions=tuple(sorted(decisions, key=lambda item: item.mention_id)),
                verification_call_count=0,
                reviewed_count=0,
                approximate_input_tokens=0,
            )

        payload = json.dumps(
            {"items": review_items},
            ensure_ascii=False,
            sort_keys=True,
        )
        response, _ = self.provider.request(
            system_message=load_prompts()["prediction_verification"],
            user_message=payload,
            response_model=PredictionReviewBatch,
            temperature=0.0,
        )
        if not isinstance(response, PredictionReviewBatch):
            raise ValueError("prediction verifier returned an unexpected response type")
        received = {decision.mention_id: decision for decision in response.decisions}
        if len(received) != len(response.decisions) or set(received) != set(review_findings):
            raise ValueError("prediction verifier returned incomplete or duplicate decisions")
        for mention_id, (finding, assertion) in review_findings.items():
            review = received[mention_id]
            reject = review.verdict == "unsupported" and review.confidence == "high"
            decisions.append(
                HybridDecision(
                    mention_id=mention_id,
                    phrase=finding.phrase,
                    hpo_id=finding.hpo_id,
                    status=(
                        "rejected"
                        if reject
                        else ("accepted" if review.verdict == "supported" else "retained_ambiguous")
                    ),
                    reason=f"model_review_{review.verdict}_{review.confidence}",
                    confidence=review.confidence,
                    assertion=assertion,
                )
            )
        return HybridResult(
            decisions=tuple(sorted(decisions, key=lambda item: item.mention_id)),
            verification_call_count=1,
            reviewed_count=len(review_items),
            approximate_input_tokens=max(1, len(payload) // 4),
        )
