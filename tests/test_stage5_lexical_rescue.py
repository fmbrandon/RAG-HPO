from __future__ import annotations

import pytest

from rag_hpo.lexical_rescue import LexicalRescueEngine, RescueVariant


def test_lexical_rescue_allowlist_adjective_to_noun() -> None:
    engine = LexicalRescueEngine()
    variants = engine.generate_variants("microspherophakic lens")
    assert len(variants) == 1
    assert variants[0].variant_phrase == "microspherophakia lens"
    assert variants[0].transformation_type == "allowlist_adjective_to_noun"
    assert variants[0].confidence_level == 2
    assert variants[0].original_phrase == "microspherophakic lens"


def test_lexical_rescue_syntactic_head_swap() -> None:
    engine = LexicalRescueEngine()
    variants = engine.generate_variants("dislocation of the crystalline lens")
    assert len(variants) >= 1
    head_swap = [v for v in variants if v.transformation_type == "syntactic_head_swap"][0]
    assert head_swap.variant_phrase == "lens dislocation"
    assert head_swap.confidence_level == 3


def test_lexical_rescue_eligibility_and_exclusions() -> None:
    engine = LexicalRescueEngine()
    # Positive eligibility
    assert engine.is_eligible("microspherophakic lens", assertion_status="affirmed") is True
    assert engine.is_eligible("corneal opacification", assertion_status="affirmed") is True

    # Excluded: Negated / non-affirmed findings
    assert engine.is_eligible("corneal opacification", assertion_status="family_history") is False

    # Excluded: Disease / Syndrome entities (Anti-Inference protection)
    assert engine.is_eligible("Marfan syndrome") is False
    assert engine.is_eligible("Peutz-Jeghers syndrome") is False
    assert engine.is_eligible("PJS") is False


def test_lexical_rescue_opacification_to_opacity() -> None:
    engine = LexicalRescueEngine()
    variants = engine.generate_variants("corneal opacification")
    assert len(variants) == 1
    assert variants[0].variant_phrase == "corneal opacity"
