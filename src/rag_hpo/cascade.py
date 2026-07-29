from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from rag_hpo.assertion import AssertionDecision, analyze_assertion
from rag_hpo.fasthpocr import FastHPOAnnotation
from rag_hpo.lexical import LexicalMention, NativeLexicalRecognizer
from rag_hpo.prompts import load_prompts
from rag_hpo.registry import HPORegistry

CascadeStatus = Literal[
    "accepted",
    "accepted_verified",
    "abstained",
    "rejected_context",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerificationDecision(_StrictModel):
    mention_id: str
    verdict: Literal["select", "unsupported", "ambiguous"]
    selected_hpo_id: str | None
    confidence: Literal["high", "medium", "low"]


class VerificationBatch(_StrictModel):
    decisions: list[VerificationDecision]


class VerificationProvider(Protocol):
    def request(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[BaseModel],
        temperature: float = 0.2,
    ) -> tuple[Any, str]: ...


@dataclass(frozen=True)
class CascadeDecision:
    mention_id: str
    phrase: str
    start_offset: int
    end_offset: int
    candidate_hpo_ids: tuple[str, ...]
    selected_hpo_id: str | None
    selected_term: str | None
    status: CascadeStatus
    reason: str
    methods: tuple[str, ...]
    evidence: str
    assertion: AssertionDecision


@dataclass(frozen=True)
class CascadeResult:
    decisions: tuple[CascadeDecision, ...]
    verification_call_count: int
    verification_item_count: int
    approximate_input_tokens: int


class PrecisionCascade:
    def __init__(
        self,
        registry: HPORegistry,
        *,
        recognizer: NativeLexicalRecognizer | None = None,
        provider: VerificationProvider | None = None,
    ) -> None:
        self.registry = registry
        self.recognizer = recognizer or NativeLexicalRecognizer(registry)
        self.provider = provider
        self._concepts = {
            concept.hp_id: concept for concept in registry.concepts if not concept.obsolete
        }

    def annotate(
        self,
        text: str,
        *,
        fast_annotations: list[FastHPOAnnotation] | None = None,
        verify: bool = False,
    ) -> CascadeResult:
        native_mentions = self.recognizer.recognize(text)
        fast_by_span = self._fast_by_span(fast_annotations or [])
        consumed_fast: set[tuple[int, int]] = set()
        decisions: list[CascadeDecision] = []
        for index, mention in enumerate(native_mentions):
            span = (mention.start_offset, mention.end_offset)
            fast_ids = fast_by_span.get(span, ())
            if fast_ids:
                consumed_fast.add(span)
            decisions.append(
                self._initial_decision(
                    text,
                    mention,
                    fast_ids=fast_ids,
                    mention_id=f"n{index:04d}",
                )
            )
        for index, annotation in enumerate(fast_annotations or []):
            span = (annotation.start_offset, annotation.end_offset)
            if span in consumed_fast:
                continue
            candidates = annotation.candidate_hpo_ids or (
                (annotation.hpo_id,) if annotation.hpo_id else ()
            )
            candidates = tuple(sorted(hp_id for hp_id in candidates if hp_id in self._concepts))
            if not candidates:
                continue
            assertion = analyze_assertion(text, *span)
            decisions.append(
                CascadeDecision(
                    mention_id=f"f{index:04d}",
                    phrase=annotation.phrase,
                    start_offset=span[0],
                    end_offset=span[1],
                    candidate_hpo_ids=candidates,
                    selected_hpo_id=None,
                    selected_term=None,
                    status=("abstained" if assertion.accepted else "rejected_context"),
                    reason=(
                        "fasthpocr_only_requires_verification"
                        if assertion.accepted
                        else f"assertion_{assertion.status}"
                    ),
                    methods=("fasthpocr",),
                    evidence="fasthpocr",
                    assertion=assertion,
                )
            )
        decisions.sort(key=lambda item: (item.start_offset, item.end_offset, item.mention_id))

        call_count = 0
        item_count = 0
        approximate_tokens = 0
        if verify and self.provider is not None:
            decisions, item_count, approximate_tokens = self._verify(text, decisions)
            call_count = int(item_count > 0)
        return CascadeResult(
            decisions=tuple(decisions),
            verification_call_count=call_count,
            verification_item_count=item_count,
            approximate_input_tokens=approximate_tokens,
        )

    @staticmethod
    def _fast_by_span(
        annotations: list[FastHPOAnnotation],
    ) -> dict[tuple[int, int], tuple[str, ...]]:
        values: dict[tuple[int, int], set[str]] = defaultdict(set)
        for annotation in annotations:
            values[(annotation.start_offset, annotation.end_offset)].update(
                annotation.candidate_hpo_ids or ((annotation.hpo_id,) if annotation.hpo_id else ())
            )
        return {span: tuple(sorted(hp_ids)) for span, hp_ids in values.items() if hp_ids}

    def _initial_decision(
        self,
        text: str,
        mention: LexicalMention,
        *,
        fast_ids: tuple[str, ...],
        mention_id: str,
    ) -> CascadeDecision:
        assertion = analyze_assertion(
            text,
            mention.start_offset,
            mention.end_offset,
        )
        methods = ("native", "fasthpocr") if fast_ids else ("native",)
        candidates = mention.candidate_hpo_ids
        if not assertion.accepted:
            return CascadeDecision(
                mention_id=mention_id,
                phrase=mention.phrase,
                start_offset=mention.start_offset,
                end_offset=mention.end_offset,
                candidate_hpo_ids=candidates,
                selected_hpo_id=None,
                selected_term=None,
                status="rejected_context",
                reason=f"assertion_{assertion.status}",
                methods=methods,
                evidence=mention.evidence,
                assertion=assertion,
            )

        fast_agrees = bool(fast_ids and (set(candidates) & set(fast_ids)))
        fast_disagrees = bool(fast_ids and not fast_agrees)
        exact_unique = mention.evidence == "exact" and len(candidates) == 1
        native_specific = exact_unique and mention.token_count >= 3
        exact_agreement = exact_unique and mention.token_count >= 2 and fast_agrees
        agreement = (
            mention.evidence == "morphological"
            and len(candidates) == 1
            and mention.token_count >= 2
            and fast_agrees
        )
        if ((native_specific or exact_agreement) and not fast_disagrees) or agreement:
            selected = candidates[0]
            return CascadeDecision(
                mention_id=mention_id,
                phrase=mention.phrase,
                start_offset=mention.start_offset,
                end_offset=mention.end_offset,
                candidate_hpo_ids=candidates,
                selected_hpo_id=selected,
                selected_term=self._concepts[selected].label,
                status="accepted",
                reason=(
                    "exact_cross_method_agreement"
                    if exact_agreement
                    else (
                        "exact_specific_asserted"
                        if native_specific
                        else "morphological_cross_method_agreement"
                    )
                ),
                methods=methods,
                evidence=mention.evidence,
                assertion=assertion,
            )
        return CascadeDecision(
            mention_id=mention_id,
            phrase=mention.phrase,
            start_offset=mention.start_offset,
            end_offset=mention.end_offset,
            candidate_hpo_ids=candidates,
            selected_hpo_id=None,
            selected_term=None,
            status="abstained",
            reason=("cross_method_disagreement" if fast_disagrees else "requires_verification"),
            methods=methods,
            evidence=mention.evidence,
            assertion=assertion,
        )

    def _verify(
        self,
        text: str,
        decisions: list[CascadeDecision],
    ) -> tuple[list[CascadeDecision], int, int]:
        unresolved = [
            decision
            for decision in decisions
            if decision.status == "abstained" and decision.candidate_hpo_ids
        ]
        if not unresolved:
            return decisions, 0, 0
        items = []
        for decision in unresolved:
            candidates = []
            for hp_id in decision.candidate_hpo_ids:
                concept = self._concepts[hp_id]
                candidates.append(
                    {
                        "hpo_id": hp_id,
                        "label": concept.label,
                        "definition": concept.definition[:500],
                        "parents": concept.parents,
                    }
                )
            items.append(
                {
                    "mention_id": decision.mention_id,
                    "phrase": decision.phrase,
                    "sentence_context": decision.assertion.sentence,
                    "evidence": decision.evidence,
                    "methods": decision.methods,
                    "candidates": candidates,
                }
            )
        payload = json.dumps({"items": items}, ensure_ascii=False, sort_keys=True)
        prompts = load_prompts()
        response, _ = self.provider.request(  # type: ignore[union-attr]
            system_message=prompts["cascade_verification"],
            user_message=payload,
            response_model=VerificationBatch,
            temperature=0.0,
        )
        if not isinstance(response, VerificationBatch):
            raise ValueError("cascade verifier returned an unexpected response type")
        expected = {decision.mention_id: decision for decision in unresolved}
        received: dict[str, VerificationDecision] = {}
        for value in response.decisions:
            if value.mention_id not in expected or value.mention_id in received:
                raise ValueError("cascade verifier returned an unknown or duplicate mention")
            received[value.mention_id] = value
        if set(received) != set(expected):
            raise ValueError("cascade verifier omitted one or more mentions")

        output: list[CascadeDecision] = []
        for decision in decisions:
            verification = received.get(decision.mention_id)
            if verification is None:
                output.append(decision)
                continue
            selected = verification.selected_hpo_id
            if verification.verdict == "select":
                if selected not in decision.candidate_hpo_ids:
                    raise ValueError("cascade verifier selected an ID outside the candidate set")
                if verification.confidence == "high":
                    output.append(
                        replace(
                            decision,
                            selected_hpo_id=selected,
                            selected_term=self._concepts[selected].label,
                            status="accepted_verified",
                            reason="model_verified_high_confidence",
                        )
                    )
                    continue
            elif selected is not None:
                raise ValueError("cascade verifier supplied an ID for a non-selection verdict")
            output.append(
                replace(
                    decision,
                    reason=f"model_{verification.verdict}_{verification.confidence}",
                )
            )
        return output, len(items), max(1, len(payload) // 4)
