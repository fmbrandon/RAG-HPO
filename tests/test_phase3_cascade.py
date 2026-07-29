from __future__ import annotations

from typing import Any

import pytest

from rag_hpo.assertion import analyze_assertion
from rag_hpo.cascade import (
    PrecisionCascade,
    VerificationBatch,
)
from rag_hpo.fasthpocr import FastHPOAnnotation
from rag_hpo.hybrid import (
    HybridPredictionVerifier,
    ModelFinding,
    PredictionReviewBatch,
)
from rag_hpo.lexical import NativeLexicalRecognizer
from rag_hpo.registry import (
    HPORegistry,
    RegistryConcept,
    RegistryPhrase,
)


def _phrase(hp_id: str, phrase: str, key: str) -> RegistryPhrase:
    return RegistryPhrase(
        record_key=key * 64,
        phrase=phrase,
        normalized_phrase=phrase.casefold(),
        source="hpo-label",
        scope="LABEL",
    )


def _registry() -> HPORegistry:
    return HPORegistry(
        data_version="test",
        concepts=[
            RegistryConcept(
                hp_id="HP:0000001",
                label="Fever",
                phrases=[
                    _phrase("HP:0000001", "Fever", "1"),
                    _phrase("HP:0000001", "High fever", "8"),
                ],
            ),
            RegistryConcept(
                hp_id="HP:0000002",
                label="Short stature",
                phrases=[_phrase("HP:0000002", "Short stature", "2")],
            ),
            RegistryConcept(
                hp_id="HP:0000003",
                label="Abnormal affect",
                phrases=[_phrase("HP:0000003", "Abnormal affect", "3")],
            ),
            RegistryConcept(
                hp_id="HP:0000004",
                label="Affect abnormality",
                phrases=[_phrase("HP:0000004", "Abnormal affect", "4")],
            ),
            RegistryConcept(
                hp_id="HP:0000005",
                label="Difficulty breathing",
                phrases=[_phrase("HP:0000005", "Difficulty breathing", "5")],
            ),
            RegistryConcept(
                hp_id="HP:0000006",
                label="Breathing dysregulation",
                phrases=[_phrase("HP:0000006", "Difficult breathing", "6")],
            ),
            RegistryConcept(
                hp_id="HP:0000007",
                label="Autism spectrum disorder",
                phrases=[_phrase("HP:0000007", "ASD", "7")],
            ),
            RegistryConcept(
                hp_id="HP:0000008",
                label="Global developmental delay",
                phrases=[
                    _phrase(
                        "HP:0000008",
                        "Global developmental delay",
                        "9",
                    )
                ],
            ),
        ],
    )


@pytest.mark.parametrize(
    ("text", "phrase", "expected"),
    [
        ("The patient has fever.", "fever", "affirmed"),
        ("The patient denies fever.", "fever", "negated"),
        ("The patient's temperature was normal.", "temperature", "normal"),
        ("No cough, but fever developed.", "fever", "affirmed"),
        ("Possible fever was discussed.", "fever", "uncertain"),
        ("The fever has resolved.", "fever", "resolved"),
        ("Monitor for fever.", "fever", "hypothetical"),
        ("Family History:\nFever in the mother.", "Fever", "family_history"),
    ],
)
def test_assertion_context_rules(text: str, phrase: str, expected: str) -> None:
    start = text.index(phrase)
    decision = analyze_assertion(text, start, start + len(phrase))
    assert decision.status == expected


def test_assertion_rejects_invalid_span() -> None:
    with pytest.raises(ValueError, match="outside"):
        analyze_assertion("text", -1, 2)


def test_native_lexical_recognizer_is_longest_first_and_ambiguity_preserving() -> None:
    recognizer = NativeLexicalRecognizer(_registry())
    mentions = recognizer.recognize(
        "Short stature with abnormal affect and difficult breathing; ASD noted."
    )
    assert [mention.phrase.casefold() for mention in mentions] == [
        "short stature",
        "abnormal affect",
        "difficult breathing",
    ]
    assert mentions[0].candidate_hpo_ids == ("HP:0000002",)
    assert mentions[1].candidate_hpo_ids == ("HP:0000003", "HP:0000004")
    assert mentions[2].candidate_hpo_ids == ("HP:0000006",)
    assert all(mention.phrase != "ASD" for mention in mentions)


def test_native_morphology_retains_tied_candidates() -> None:
    recognizer = NativeLexicalRecognizer(_registry())
    mention = recognizer.recognize("Difficulties breathing were reported.")[0]
    assert mention.evidence == "morphological"
    assert mention.candidate_hpo_ids == ("HP:0000005", "HP:0000006")


def test_offline_cascade_accepts_exact_and_abstains_on_ambiguity() -> None:
    text = (
        "Fever, global developmental delay, and abnormal affect. The patient denies short stature."
    )
    result = PrecisionCascade(_registry()).annotate(text)
    by_phrase = {decision.phrase.casefold(): decision for decision in result.decisions}
    assert by_phrase["fever"].status == "abstained"
    assert by_phrase["global developmental delay"].status == "accepted"
    assert by_phrase["global developmental delay"].selected_hpo_id == "HP:0000008"
    assert by_phrase["abnormal affect"].status == "abstained"
    assert by_phrase["short stature"].status == "rejected_context"
    assert result.verification_call_count == 0


def test_cross_method_does_not_override_morphological_ambiguity() -> None:
    text = "Difficulties breathing were present."
    native = NativeLexicalRecognizer(_registry()).recognize(text)[0]
    fast = FastHPOAnnotation(
        phrase=native.phrase,
        hpo_id="HP:0000005",
        hpo_term="Difficulty breathing",
        start_offset=native.start_offset,
        end_offset=native.end_offset,
        candidate_hpo_ids=("HP:0000005",),
        resolution="registry",
    )
    result = PrecisionCascade(_registry()).annotate(
        text,
        fast_annotations=[fast],
    )
    assert result.decisions[0].status == "abstained"
    assert result.decisions[0].reason == "requires_verification"


def test_cross_method_unique_morphological_agreement_is_accepted() -> None:
    text = "High fevers were present."
    native = NativeLexicalRecognizer(_registry()).recognize(text)[0]
    fast = FastHPOAnnotation(
        phrase=native.phrase,
        hpo_id="HP:0000001",
        hpo_term="Fever",
        start_offset=native.start_offset,
        end_offset=native.end_offset,
        candidate_hpo_ids=("HP:0000001",),
        resolution="registry",
    )
    decision = (
        PrecisionCascade(_registry())
        .annotate(
            text,
            fast_annotations=[fast],
        )
        .decisions[0]
    )
    assert decision.status == "accepted"
    assert decision.reason == "morphological_cross_method_agreement"


class Verifier:
    def __init__(self, selected: str = "HP:0000003") -> None:
        self.selected = selected
        self.calls = 0

    def request(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> tuple[Any, str]:
        self.calls += 1
        assert "specific" in system_message
        assert "sentence_context" in user_message
        assert response_model is VerificationBatch
        assert temperature == 0.0
        value = VerificationBatch.model_validate(
            {
                "decisions": [
                    {
                        "mention_id": "n0000",
                        "verdict": "select",
                        "selected_hpo_id": self.selected,
                        "confidence": "high",
                    }
                ]
            }
        )
        return value, value.model_dump_json()


def test_batched_verifier_accepts_only_high_confidence_supplied_id() -> None:
    provider = Verifier()
    result = PrecisionCascade(
        _registry(),
        provider=provider,  # type: ignore[arg-type]
    ).annotate("Abnormal affect.", verify=True)
    assert provider.calls == 1
    assert result.verification_call_count == 1
    assert result.verification_item_count == 1
    assert result.approximate_input_tokens > 0
    assert result.decisions[0].status == "accepted_verified"
    assert result.decisions[0].selected_hpo_id == "HP:0000003"


def test_batched_verifier_rejects_out_of_set_id() -> None:
    with pytest.raises(ValueError, match="outside"):
        PrecisionCascade(
            _registry(),
            provider=Verifier("HP:9999999"),  # type: ignore[arg-type]
        ).annotate("Abnormal affect.", verify=True)


class PredictionVerifier:
    def __init__(
        self,
        *,
        verdict: str = "supported",
        confidence: str = "high",
        incomplete: bool = False,
    ) -> None:
        self.verdict = verdict
        self.confidence = confidence
        self.incomplete = incomplete
        self.calls = 0

    def request(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> tuple[Any, str]:
        self.calls += 1
        assert "already proposed" in system_message
        assert "sentence_context" in user_message
        assert response_model is PredictionReviewBatch
        assert temperature == 0.0
        decisions = []
        if not self.incomplete:
            decisions.append(
                {
                    "mention_id": "m0000",
                    "verdict": self.verdict,
                    "confidence": self.confidence,
                }
            )
        value = PredictionReviewBatch.model_validate({"decisions": decisions})
        return value, value.model_dump_json()


def test_hybrid_accepts_model_fasthpocr_agreement_without_provider_call() -> None:
    provider = PredictionVerifier()
    result = HybridPredictionVerifier(
        _registry(),
        provider=provider,  # type: ignore[arg-type]
    ).verify(
        "The patient has high fever.",
        [ModelFinding(phrase="high fever", hpo_id="HP:0000001")],
        high_confidence_ids={"HP:0000001"},
    )
    assert provider.calls == 0
    assert result.decisions[0].status == "accepted"
    assert result.decisions[0].reason == "model_fasthpocr_agreement"


def test_hybrid_flags_but_retains_nonaffirmed_prediction_locally() -> None:
    result = HybridPredictionVerifier(_registry(), provider=None).verify(
        "The patient denies short stature.",
        [ModelFinding(phrase="short stature", hpo_id="HP:0000002")],
        high_confidence_ids=set(),
    )
    assert result.decisions[0].status == "retained_ambiguous"
    assert result.decisions[0].reason == "assertion_negated_retained"


def test_hybrid_only_rejects_high_confidence_unsupported_review() -> None:
    provider = PredictionVerifier(verdict="unsupported", confidence="high")
    result = HybridPredictionVerifier(
        _registry(),
        provider=provider,  # type: ignore[arg-type]
    ).verify(
        "The patient has short stature.",
        [ModelFinding(phrase="short stature", hpo_id="HP:0000002")],
        high_confidence_ids=set(),
    )
    assert result.decisions[0].status == "rejected"
    assert result.verification_call_count == 1
    assert result.reviewed_count == 1


@pytest.mark.parametrize(
    ("verdict", "confidence"),
    [("unsupported", "medium"), ("ambiguous", "high")],
)
def test_hybrid_retains_uncertain_reviews(verdict: str, confidence: str) -> None:
    provider = PredictionVerifier(verdict=verdict, confidence=confidence)
    result = HybridPredictionVerifier(
        _registry(),
        provider=provider,  # type: ignore[arg-type]
    ).verify(
        "The patient has short stature.",
        [ModelFinding(phrase="short stature", hpo_id="HP:0000002")],
        high_confidence_ids=set(),
    )
    assert result.decisions[0].status == "retained_ambiguous"


def test_hybrid_retains_when_phrase_is_not_located_or_provider_missing() -> None:
    provider = PredictionVerifier(verdict="unsupported", confidence="high")
    result = HybridPredictionVerifier(
        _registry(),
        provider=provider,  # type: ignore[arg-type]
    ).verify(
        "The patient is affected.",
        [ModelFinding(phrase="short stature", hpo_id="HP:0000002")],
        high_confidence_ids=set(),
    )
    assert provider.calls == 0
    assert result.decisions[0].status == "retained_ambiguous"
    assert result.decisions[0].reason == "phrase_not_located"

    result = HybridPredictionVerifier(_registry(), provider=None).verify(
        "The patient has short stature.",
        [ModelFinding(phrase="short stature", hpo_id="HP:0000002")],
        high_confidence_ids=set(),
    )
    assert result.decisions[0].status == "retained_ambiguous"
    assert result.decisions[0].reason == "verification_unavailable"


def test_hybrid_rejects_incomplete_review_response() -> None:
    provider = PredictionVerifier(incomplete=True)
    with pytest.raises(ValueError, match="incomplete"):
        HybridPredictionVerifier(
            _registry(),
            provider=provider,  # type: ignore[arg-type]
        ).verify(
            "The patient has short stature.",
            [ModelFinding(phrase="short stature", hpo_id="HP:0000002")],
            high_confidence_ids=set(),
        )
