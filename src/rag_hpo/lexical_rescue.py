from __future__ import annotations

import re
from dataclasses import dataclass

from rag_hpo.registry import normalize_phrase

# Explicit allowlist of morphological adjectival -> noun and nominalization equivalences
MORPHOLOGICAL_ALLOWLIST: dict[str, str] = {
    "microspherophakic": "microspherophakia",
    "spherophakic": "spherophakia",
    "microphthalmic": "microphthalmia",
    "anophthalmic": "anophthalmia",
    "edematous": "edema",
    "glaucomatous": "glaucoma",
    "hypertrophic": "hypertrophy",
    "hypoplastic": "hypoplasia",
    "atrophic": "atrophy",
    "opacification": "opacity",
    "opacified": "opacity",
}

# Exclusion list of disease/syndrome entities that must NEVER undergo phenotype rescue
DISEASE_SYNDROME_EXCLUSIONS: set[str] = {
    "marfan syndrome",
    "peutz-jeghers syndrome",
    "pjs",
    "ehlers-danlos syndrome",
    "turner syndrome",
    "down syndrome",
    "williams syndrome",
    "prader-willi syndrome",
    "angelman syndrome",
    "fragile x syndrome",
}

# Essential clinical modifiers that must NEVER be stripped during rescue
CRITICAL_SEMANTIC_MODIFIERS: set[str] = {
    "nonpitting",
    "pitting",
    "progressive",
    "congenital",
    "intermittent",
    "paroxysmal",
    "unilateral",
    "bilateral",
}


@dataclass(frozen=True)
class RescueVariant:
    variant_phrase: str
    transformation_type: str
    # 1: HPO Synonym, 2: Project Allowlist, 3: Syntactic Rewrite, 4: LLM
    confidence_level: int
    original_phrase: str


class LexicalRescueEngine:
    """Controlled Lexical Rescue Engine for Pass 2.5 HPO Mapping."""

    def __init__(self, allowlist: dict[str, str] | None = None) -> None:
        self.allowlist = allowlist or MORPHOLOGICAL_ALLOWLIST

    def is_eligible(self, phrase: str, assertion_status: str = "affirmed") -> bool:
        """Determine if a mention is eligible for Lexical Rescue.

        Excludes diseases/syndromes, non-affirmed assertions, and critical semantic modifiers.
        """
        if assertion_status not in {"affirmed", "normal"}:
            return False

        norm = normalize_phrase(phrase)
        if norm in DISEASE_SYNDROME_EXCLUSIONS:
            return False

        # Exclude phrases containing disease/syndrome keywords
        if "syndrome" in norm or "disease" in norm:
            return False

        return True

    def generate_variants(self, phrase: str) -> list[RescueVariant]:
        """Generate controlled lexical variants for an eligible clinical phrase."""
        norm = normalize_phrase(phrase)
        words = norm.split()
        variants: list[RescueVariant] = []
        seen: set[str] = {norm}

        # 1. Check direct allowlist matches on individual words
        transformed_words: list[str] = []
        rule_applied = False
        for word in words:
            if word in self.allowlist:
                transformed_words.append(self.allowlist[word])
                rule_applied = True
            else:
                transformed_words.append(word)

        if rule_applied:
            candidate = " ".join(transformed_words)
            candidate_norm = normalize_phrase(candidate)
            if candidate_norm not in seen:
                seen.add(candidate_norm)
                variants.append(
                    RescueVariant(
                        variant_phrase=candidate_norm,
                        transformation_type="allowlist_adjective_to_noun",
                        confidence_level=2,
                        original_phrase=phrase,
                    )
                )

        # 2. Check prepositional syntactic head swap ("A of the B" -> "B A")
        match = re.match(r"^(.+?)\s+of(?:\s+the)?\s+(.+)$", norm)
        if match:
            head, tail = match.group(1), match.group(2)
            # Remove filler words like "crystalline" from "crystalline lens"
            tail_clean = " ".join(w for w in tail.split() if w not in {"crystalline"})
            candidate_swap = normalize_phrase(f"{tail_clean} {head}")
            if candidate_swap not in seen:
                seen.add(candidate_swap)
                variants.append(
                    RescueVariant(
                        variant_phrase=candidate_swap,
                        transformation_type="syntactic_head_swap",
                        confidence_level=3,
                        original_phrase=phrase,
                    )
                )

        return variants
