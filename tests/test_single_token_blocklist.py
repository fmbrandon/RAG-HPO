import pytest

from rag_hpo.lexical import SINGLE_TOKEN_MODIFIER_BLOCKLIST
from rag_hpo.staged_pipeline import Mention, StagedAnnotationPipeline


@pytest.mark.parametrize(
    "mention_text",
    ["left", "right", "severe", "chronic", "recurrent", "acute", "bilateral", "onset"],
)
def test_standalone_modifiers_are_filtered(mention_text: str) -> None:
    assert mention_text.lower() in SINGLE_TOKEN_MODIFIER_BLOCKLIST
    mention = Mention(
        phrase=mention_text,
        start=0,
        end=len(mention_text),
        methods={"native"},
        evidence_segments=[(0, len(mention_text))],
    )
    merged = StagedAnnotationPipeline._merge_mentions([mention])
    assert len(merged) == 0


@pytest.mark.parametrize(
    "mention_text",
    [
        "left eyelid swelling",
        "severe abdominal pain",
        "chronic constipation",
        "recurrent infections",
        "bilateral cataracts",
    ],
)
def test_modifiers_are_preserved_inside_complete_mentions(mention_text: str) -> None:
    tokens = mention_text.split()
    for token in tokens:
        if token.lower() in SINGLE_TOKEN_MODIFIER_BLOCKLIST:
            assert True
            break
    mention = Mention(
        phrase=mention_text,
        start=0,
        end=len(mention_text),
        methods={"native"},
        evidence_segments=[(0, len(mention_text))],
    )
    merged = StagedAnnotationPipeline._merge_mentions([mention])
    assert len(merged) == 1
    assert merged[0].phrase == mention_text


@pytest.mark.parametrize(
    "phrase, expected_head",
    [
        ("severe pain", "pain"),
        ("chronic abdominal pain", "abdominal pain"),
        ("recurrent infections", "infections"),
    ],
)
def test_head_phrase_expansion_preserves_full_source_span(phrase: str, expected_head: str) -> None:
    mention = Mention(
        phrase=phrase,
        start=0,
        end=len(phrase),
        methods={"native"},
        evidence_segments=[(0, len(phrase))],
    )
    merged = StagedAnnotationPipeline._merge_mentions([mention])
    assert len(merged) == 1
    assert merged[0].phrase == phrase
    assert expected_head in merged[0].phrase_variants


@pytest.mark.parametrize(
    "canonical_phrase",
    [
        "acute lymphoblastic leukemia",
        "chronic kidney disease",
        "left ventricular hypertrophy",
    ],
)
def test_canonical_phrases_preserve_full_source_span(canonical_phrase: str) -> None:
    mention = Mention(
        phrase=canonical_phrase,
        start=0,
        end=len(canonical_phrase),
        methods={"native"},
        evidence_segments=[(0, len(canonical_phrase))],
    )
    merged = StagedAnnotationPipeline._merge_mentions([mention])
    assert len(merged) == 1
    assert merged[0].phrase == canonical_phrase
