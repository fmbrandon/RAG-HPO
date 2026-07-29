from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

AssertionStatus = Literal[
    "affirmed",
    "normal",
    "negated",
    "family_history",
    "uncertain",
    "resolved",
    "hypothetical",
]

_BOUNDARY = re.compile(r"[\n.!?;]")
_CLAUSE_BOUNDARY = re.compile(r"(?:[,;:]|\bbut\b|\bhowever\b|\balthough\b)", re.I)
_INDEPENDENT_AND_BOUNDARY = re.compile(
    r"\band\b(?=\s+"
    r"(?:(?:the|a|an|his|her|their|this)\s+)?"
    r"(?:(?:abdominal|chest|cranial|cardiac|renal|thyroid|pelvic)\s+)?"
    r"(?:patient|proband|child|adult|examination|exam|imaging|"
    r"ct|mri|ultrasound|sonography|radiograph|x-ray|scan|biopsy|"
    r"colonoscopy|endoscopy|echocardiogram|laboratory|labs?|bloodwork)"
    r"\s+(?:was|were|is|are|had|has|showed|revealed|demonstrated|"
    r"identified|found|noted)\b)",
    re.I,
)
_SECTION = re.compile(r"^\s*([A-Za-z][A-Za-z /_-]{1,48}):\s*$")
_FAMILY_SECTION = re.compile(r"\b(?:family|pedigree|maternal|paternal)\b", re.I)
_FAMILY_CUE = re.compile(
    r"\b(?:family history|mother|father|maternal|paternal|parent|sibling|"
    r"brother|sister|son|daughter)\b",
    re.I,
)
_NEGATION_CUE = re.compile(
    r"\b(?:no|not|denies|denied|without|absence of|absent|negative for|"
    r"free of|never had|no evidence of)\b",
    re.I,
)
_NORMAL_CUE = re.compile(
    r"\b(?:normal|normally|unremarkable|within normal limits|intact)\b",
    re.I,
)
_UNCERTAIN_CUE = re.compile(
    r"\b(?:possible|possibly|suspected|suspicious for|may have|might have|"
    r"could have|concern for|question of|rule out|r/o|differential(?: diagnosis)?)\b",
    re.I,
)
_RESOLVED_CUE = re.compile(
    r"\b(?:resolved|resolution of|no longer|previously present but|"
    r"former(?:ly)?|remote history of)\b",
    re.I,
)
_HYPOTHETICAL_CUE = re.compile(
    r"\b(?:at risk for|risk of|monitor for|watch for|if (?:the )?patient "
    r"(?:develops|has)|in case of)\b",
    re.I,
)
_POST_NEGATION = re.compile(
    r"^\s*(?:was|were|is|are|has been|have been)?\s*"
    r"(?:not present|absent|negative|ruled out)\b",
    re.I,
)
_POST_NORMAL = re.compile(
    r"^\s*(?:was|were|is|are|appears?|remains?)?\s*"
    r"(?:normal|unremarkable|within normal limits|intact)\b",
    re.I,
)
_POST_RESOLVED = re.compile(
    r"^\s*(?:has|have|had|was|were|is|are)?\s*"
    r"(?:resolved|no longer present)\b",
    re.I,
)


@dataclass(frozen=True)
class AssertionDecision:
    status: AssertionStatus
    cue: str | None
    sentence: str
    section: str | None

    @property
    def accepted(self) -> bool:
        return self.status == "affirmed"


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for match in _BOUNDARY.finditer(text, 0, start):
        left = match.end()
    right_match = _BOUNDARY.search(text, end)
    right = right_match.start() if right_match else len(text)
    return left, right


def _section_at(text: str, start: int) -> str | None:
    section: str | None = None
    consumed = 0
    for line in text[:start].splitlines(keepends=True):
        consumed += len(line)
        match = _SECTION.match(line.rstrip("\r\n"))
        if match:
            section = match.group(1).strip()
        elif line.strip() == "":
            continue
        elif consumed < start and len(line.strip()) > 80:
            # Long prose does not cancel a heading, but it is not itself one.
            continue
    return section


def _last_clause(value: str) -> str:
    matches = [
        *_CLAUSE_BOUNDARY.finditer(value),
        *_INDEPENDENT_AND_BOUNDARY.finditer(value),
    ]
    matches.sort(key=lambda match: match.start())
    return value[matches[-1].end() :] if matches else value


def _cue(pattern: re.Pattern[str], value: str) -> str | None:
    matches = list(pattern.finditer(value))
    return matches[-1].group(0) if matches else None


def analyze_assertion(text: str, start: int, end: int) -> AssertionDecision:
    if start < 0 or end < start or end > len(text):
        raise ValueError("assertion span is outside the source text")
    sentence_start, sentence_end = _sentence_bounds(text, start, end)
    raw_sentence = text[sentence_start:sentence_end]
    leading = len(raw_sentence) - len(raw_sentence.lstrip())
    sentence_start += leading
    sentence = raw_sentence.strip()
    relative_start = start - sentence_start
    relative_end = end - sentence_start
    before = _last_clause(sentence[:relative_start])[-120:]
    mention_text = sentence[relative_start:relative_end]
    after = sentence[relative_end:][:80]
    section = _section_at(text, start)

    if section and _FAMILY_SECTION.search(section):
        return AssertionDecision("family_history", section, sentence, section)
    if cue := _cue(_FAMILY_CUE, before):
        return AssertionDecision("family_history", cue, sentence, section)
    if cue := _cue(_FAMILY_CUE, mention_text):
        return AssertionDecision("family_history", cue, sentence, section)
    if cue := _cue(_HYPOTHETICAL_CUE, before):
        return AssertionDecision("hypothetical", cue, sentence, section)
    if cue := _cue(_HYPOTHETICAL_CUE, mention_text):
        return AssertionDecision("hypothetical", cue, sentence, section)
    if cue := _cue(_RESOLVED_CUE, before):
        return AssertionDecision("resolved", cue, sentence, section)
    if cue := _cue(_RESOLVED_CUE, mention_text):
        return AssertionDecision("resolved", cue, sentence, section)
    if cue := _cue(_UNCERTAIN_CUE, before):
        return AssertionDecision("uncertain", cue, sentence, section)
    if cue := _cue(_UNCERTAIN_CUE, mention_text):
        return AssertionDecision("uncertain", cue, sentence, section)
    if cue := _cue(_NEGATION_CUE, before):
        return AssertionDecision("negated", cue, sentence, section)
    if cue := _cue(_NEGATION_CUE, mention_text):
        return AssertionDecision("negated", cue, sentence, section)
    if cue := _cue(_NORMAL_CUE, before):
        return AssertionDecision("normal", cue, sentence, section)
    if cue := _cue(_NORMAL_CUE, mention_text):
        return AssertionDecision("normal", cue, sentence, section)
    if match := _POST_NEGATION.search(after):
        return AssertionDecision("negated", match.group(0).strip(), sentence, section)
    if match := _POST_NORMAL.search(after):
        return AssertionDecision("normal", match.group(0).strip(), sentence, section)
    if match := _POST_RESOLVED.search(after):
        return AssertionDecision("resolved", match.group(0).strip(), sentence, section)
    return AssertionDecision("affirmed", None, sentence, section)
