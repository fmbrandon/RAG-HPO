from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from rag_hpo.registry import HPORegistry, normalize_phrase

EvidenceKind = Literal["exact", "morphological"]

_TOKEN = re.compile(r"\w+(?:[\u2019']\w+)?", re.UNICODE)
_SHORT_ACRONYM = re.compile(r"^[A-Za-z]{1,3}$")
_SINGLE_TOKEN_BLOCKLIST = {
    "all",
    "child",
    "adult",
    "male",
    "female",
    "finding",
    "history",
    "normal",
    "patient",
}


@dataclass(frozen=True)
class TextToken:
    text: str
    normalized: str
    start: int
    end: int


@dataclass(frozen=True)
class LexicalMention:
    phrase: str
    start_offset: int
    end_offset: int
    candidate_hpo_ids: tuple[str, ...]
    record_keys: tuple[str, ...]
    evidence: EvidenceKind
    token_count: int


def _tokens(text: str) -> list[TextToken]:
    return [
        TextToken(
            text=match.group(0),
            normalized=normalize_phrase(match.group(0)),
            start=match.start(),
            end=match.end(),
        )
        for match in _TOKEN.finditer(text)
    ]


def _stem(token: str) -> str:
    value = token
    if len(value) > 6 and value.endswith("ies"):
        value = value[:-3] + "y"
    if len(value) > 7 and value.endswith("ulty"):
        return value[:-1]
    if len(value) > 7 and value.endswith("ity"):
        return value[:-1]
    if len(value) > 7 and value.endswith("ing"):
        value = value[:-3]
        if len(value) > 3 and value[-1] == value[-2]:
            value = value[:-1]
        return value
    if len(value) > 6 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 6 and value.endswith("es"):
        return value[:-2]
    if len(value) > 5 and value.endswith("s"):
        return value[:-1]
    return value


def _signature(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({_stem(token) for token in tokens}))


class NativeLexicalRecognizer:
    """Deterministic lexical HPO recognizer built only from the canonical registry."""

    def __init__(
        self,
        registry: HPORegistry,
        *,
        max_tokens: int = 12,
        enable_morphology: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.enable_morphology = enable_morphology
        exact: dict[tuple[str, ...], dict[str, set[str]]] = defaultdict(
            lambda: {"hp_ids": set(), "record_keys": set()}
        )
        morphological: dict[tuple[int, tuple[str, ...]], set[str]] = defaultdict(set)

        active = {concept.hp_id for concept in registry.concepts if not concept.obsolete}
        records = [
            (concept.hp_id, phrase.phrase, phrase.record_key)
            for concept in registry.concepts
            if not concept.obsolete
            for phrase in concept.phrases
        ]
        records.extend(
            (extension.hp_id, extension.phrase, extension.record_key)
            for extension in registry.extensions
            if extension.hp_id in active
        )
        for hp_id, phrase, record_key in records:
            normalized_tokens = tuple(normalize_phrase(phrase).split())
            if (
                not normalized_tokens
                or len(normalized_tokens) > max_tokens
                or self._excluded_single_token(normalized_tokens)
            ):
                continue
            exact[normalized_tokens]["hp_ids"].add(hp_id)
            exact[normalized_tokens]["record_keys"].add(record_key)
            if enable_morphology and len(normalized_tokens) <= 8:
                morphological[(len(normalized_tokens), _signature(normalized_tokens))].add(hp_id)
        self._exact = {
            key: (
                tuple(sorted(value["hp_ids"])),
                tuple(sorted(value["record_keys"])),
            )
            for key, value in exact.items()
        }
        self._morphological = {key: tuple(sorted(value)) for key, value in morphological.items()}
        self._lengths = tuple(sorted({len(value) for value in self._exact}, reverse=True))

    @staticmethod
    def _excluded_single_token(tokens: tuple[str, ...]) -> bool:
        if len(tokens) != 1:
            return False
        token = tokens[0]
        return token in _SINGLE_TOKEN_BLOCKLIST or _SHORT_ACRONYM.fullmatch(token) is not None

    def recognize(self, text: str) -> list[LexicalMention]:
        tokens = _tokens(text)
        matches: dict[tuple[int, int], LexicalMention] = {}
        for start_index, _start_token in enumerate(tokens):
            remaining = len(tokens) - start_index
            for length in self._lengths:
                if length > remaining:
                    continue
                span_tokens = tokens[start_index : start_index + length]
                values = tuple(token.normalized for token in span_tokens)
                exact = self._exact.get(values)
                if exact is not None:
                    hp_ids, record_keys = exact
                    mention = self._mention(
                        text,
                        span_tokens,
                        hp_ids,
                        record_keys,
                        "exact",
                    )
                    matches[(mention.start_offset, mention.end_offset)] = mention
                    continue
                if not self.enable_morphology or length > 8:
                    continue
                morphological_ids = self._morphological.get((length, _signature(values)))
                if morphological_ids:
                    mention = self._mention(
                        text,
                        span_tokens,
                        morphological_ids,
                        (),
                        "morphological",
                    )
                    matches.setdefault(
                        (mention.start_offset, mention.end_offset),
                        mention,
                    )
        return self._longest_nonoverlapping(list(matches.values()))

    @staticmethod
    def _mention(
        text: str,
        tokens: list[TextToken],
        hp_ids: tuple[str, ...],
        record_keys: tuple[str, ...],
        evidence: EvidenceKind,
    ) -> LexicalMention:
        start = tokens[0].start
        end = tokens[-1].end
        return LexicalMention(
            phrase=text[start:end],
            start_offset=start,
            end_offset=end,
            candidate_hpo_ids=hp_ids,
            record_keys=record_keys,
            evidence=evidence,
            token_count=len(tokens),
        )

    @staticmethod
    def _longest_nonoverlapping(
        mentions: list[LexicalMention],
    ) -> list[LexicalMention]:
        accepted: list[LexicalMention] = []
        for mention in sorted(
            mentions,
            key=lambda value: (
                -value.token_count,
                value.start_offset,
                value.evidence != "exact",
                value.candidate_hpo_ids,
            ),
        ):
            if any(
                mention.start_offset < other.end_offset and mention.end_offset > other.start_offset
                for other in accepted
            ):
                continue
            accepted.append(mention)
        return sorted(
            accepted,
            key=lambda value: (
                value.start_offset,
                value.end_offset,
                value.candidate_hpo_ids,
            ),
        )
