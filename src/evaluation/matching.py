"""Match retrieved chunks against labelled evidence expectations."""

import re
import unicodedata

from src.evaluation.models import EvidenceExpectation, RetrievedSource

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize text for term matching without changing its meaning.

    The transformation applies Unicode NFKC folding, case folding, whitespace
    collapsing, and trimming. It never removes CJK characters and never
    performs aggressive tokenization, so Chinese terms stay intact.
    """
    if not isinstance(value, str):
        raise TypeError("待标准化的内容必须是字符串")
    folded = unicodedata.normalize("NFKC", value).casefold()
    return _WHITESPACE.sub(" ", folded).strip()


def _page_number_matches(
    expectation: EvidenceExpectation,
    source: RetrievedSource,
) -> bool:
    if expectation.page_number is None:
        return True
    return expectation.page_number == source.page_number


def evidence_matches(
    expectation: EvidenceExpectation,
    source: RetrievedSource,
) -> bool:
    """Return True when one retrieved chunk satisfies one evidence group."""
    if expectation.document_id != source.document_id:
        return False
    if not _page_number_matches(expectation, source):
        return False
    if not expectation.required_terms:
        return False

    normalized_text = normalize_text(source.text)
    return all(
        normalize_text(term) in normalized_text
        for term in expectation.required_terms
    )


def text_matches_expectation(expectation: EvidenceExpectation, text: str) -> bool:
    """Return True when raw chunk text contains every required term."""
    if not expectation.required_terms:
        return False
    normalized_text = normalize_text(text)
    return all(
        normalize_text(term) in normalized_text
        for term in expectation.required_terms
    )


def matched_evidence_indexes(
    expectations: tuple[EvidenceExpectation, ...],
    sources: tuple[RetrievedSource, ...],
    k: int,
) -> frozenset[int]:
    """Return indexes of evidence groups matched by any of the Top-K sources.

    Each evidence group is counted at most once even when several retrieved
    chunks satisfy it.
    """
    if k <= 0:
        raise ValueError("k 必须大于 0")

    matched: set[int] = set()
    for source in sources[:k]:
        for index, expectation in enumerate(expectations):
            if index in matched:
                continue
            if evidence_matches(expectation, source):
                matched.add(index)
    return frozenset(matched)


def count_matched_evidence(
    expectations: tuple[EvidenceExpectation, ...],
    sources: tuple[RetrievedSource, ...],
    k: int,
) -> int:
    """Return how many distinct evidence groups the Top-K sources cover."""
    return len(matched_evidence_indexes(expectations, sources, k))


def first_relevant_rank(
    expectations: tuple[EvidenceExpectation, ...],
    sources: tuple[RetrievedSource, ...],
) -> int | None:
    """Return the one-based position of the first chunk matching any evidence."""
    if not expectations:
        return None
    for position, source in enumerate(sources, start=1):
        if any(
            evidence_matches(expectation, source)
            for expectation in expectations
        ):
            return position
    return None
