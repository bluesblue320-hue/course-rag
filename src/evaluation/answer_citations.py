"""Strict ``[来源N]`` citation parsing for final answers.

Only the exact production format ``[来源N]`` with ``N`` a positive integer
with no leading zeros is accepted.  Look-alike fragments such as ``[来源0]``,
``[来源01]``, ``[来源A]``, or ``【来源1】`` are collected separately as
malformed so they can be reported without stopping evaluation.
"""

import re
from typing import Iterable

from src.evaluation.answer_models import (
    CitationOccurrence,
    CitationParseResult,
)

_VALID_CITATION = re.compile(r"\[来源([1-9]\d*)\]")
# Source-like look-alikes only.  Ordinary Markdown bracket text such as
# ``[FastAPI]``, ``[Python]``, or ``[1]`` is NOT a citation and must not be
# reported as malformed.  Only fragments that explicitly start with the
# source keyword (Chinese or English, case-insensitive) are candidates.
_MALFORMED_LOOKALIKE = re.compile(
    r"\[(?:来源|source)[^\[\]]*\]|【(?:来源|source)[^】]*】",
    re.IGNORECASE,
)

# All source-like citation markup, valid or malformed, in either bracket
# style.  Used to strip citation markup before lexical fact matching so that
# ``[来源1]`` never contributes a fake phrase and removal never fuses the
# surrounding text (the marker is replaced by a neutral separator).
_CITATION_MARKUP = re.compile(
    r"\[(?:来源|source)[^\[\]]*\]|【(?:来源|source)[^】]*】",
    re.IGNORECASE,
)
_NEUTRAL_SEPARATOR = " | "


def remove_citation_markup(answer: str) -> str:
    """Return an answer copy with source-like citation markup neutralized.

    Every source-like fragment (valid ``[来源N]`` and malformed look-alikes
    such as ``[来源0]``, ``[来源A]``, ``[source1]``, ``【来源1】``) is
    replaced by a neutral separator.  Ordinary Markdown bracket text such as
    ``[FastAPI]`` or ``[Python]`` is left untouched.

    The separator is used instead of an empty string so that
    ``"Service[来源1]层"`` does not become ``"Service层"`` and create an
    artificial phrase match after normalization.
    """
    if not isinstance(answer, str):
        raise TypeError("待处理的答案必须是字符串")
    return _CITATION_MARKUP.sub(_NEUTRAL_SEPARATOR, answer)


def parse_citations(answer: str) -> CitationParseResult:
    """Parse every strict citation and malformed look-alike in one answer.

    The original answer is never modified.  Duplicate citations are kept as
    separate occurrences so occurrence counts and unique-source counts can be
    computed independently.
    """
    if not isinstance(answer, str):
        raise TypeError("待解析的答案必须是字符串")

    occurrences: list[CitationOccurrence] = []
    for match in _VALID_CITATION.finditer(answer):
        source_number = int(match.group(1))
        occurrences.append(
            CitationOccurrence(
                source_number=source_number,
                start=match.start(),
                end=match.end(),
            )
        )

    malformed: list[str] = []
    for match in _MALFORMED_LOOKALIKE.finditer(answer):
        fragment = match.group(0)
        if _is_strictly_valid_fragment(fragment):
            continue
        if _is_inside_valid_citation(occurrences, match.start(), match.end()):
            continue
        malformed.append(fragment)

    return CitationParseResult(
        valid_syntax=tuple(occurrences),
        malformed_fragments=tuple(malformed),
    )


def _is_strictly_valid_fragment(fragment: str) -> bool:
    """Return True when a matched fragment is itself a strict citation."""
    return _VALID_CITATION.fullmatch(fragment) is not None


def _is_inside_valid_citation(
    occurrences: Iterable[CitationOccurrence],
    start: int,
    end: int,
) -> bool:
    """Ignore look-alike spans that overlap an already parsed strict citation.

    This prevents a strict citation like ``[来源12]`` from also being
    reported as a malformed fragment by a broader pattern.
    """
    for occurrence in occurrences:
        if occurrence.start <= start and end <= occurrence.end:
            return True
    return False


def citation_numbers(occurrences: Iterable[CitationOccurrence]) -> tuple[int, ...]:
    """Return the cited source numbers in the order they appear."""
    return tuple(occurrence.source_number for occurrence in occurrences)


def unique_cited_sources(occurrences: Iterable[CitationOccurrence]) -> tuple[int, ...]:
    """Return distinct cited source numbers in first-appearance order."""
    seen: set[int] = set()
    unique: list[int] = []
    for occurrence in occurrences:
        if occurrence.source_number not in seen:
            seen.add(occurrence.source_number)
            unique.append(occurrence.source_number)
    return tuple(unique)
