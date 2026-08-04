"""Evidence matching between retrieval sources and ground truth labels."""

import re
import unicodedata
from dataclasses import dataclass

from .models import EvidenceExpectation, EvaluationCase

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for term comparison without aggressive tokenization."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    normalized = normalized.casefold()
    return _WHITESPACE_PATTERN.sub(" ", normalized)


@dataclass(frozen=True)
class EvidenceMatch:
    """Describe which evidence groups matched which retrieval ranks."""

    first_relevant_rank: int | None
    matched_evidence_count_at_1: int
    matched_evidence_count_at_3: int
    matched_evidence_count_at_5: int
    total_evidence_count: int
    hit_at_1: bool | None
    hit_at_3: bool | None
    hit_at_5: bool | None
    reciprocal_rank: float | None


def matches_expectation(
    chunk_text: str,
    chunk_page_number: int | None,
    expectation: EvidenceExpectation,
) -> bool:
    """Return whether one chunk satisfies one evidence group."""
    if expectation.page_number is not None and (
        chunk_page_number != expectation.page_number
    ):
        return False
    normalized_chunk = normalize_text(chunk_text)
    return all(
        normalize_text(term) in normalized_chunk
        for term in expectation.required_terms
    )


def compute_evidence_match(
    sources: list[dict[str, object]],
    case: EvaluationCase,
) -> EvidenceMatch:
    """Match every source against every evidence group once."""
    groups = case.expected_evidence
    total = len(groups)
    if not case.answerable or total == 0:
        return EvidenceMatch(
            first_relevant_rank=None,
            matched_evidence_count_at_1=0,
            matched_evidence_count_at_3=0,
            matched_evidence_count_at_5=0,
            total_evidence_count=total,
            hit_at_1=None,
            hit_at_3=None,
            hit_at_5=None,
            reciprocal_rank=None,
        )

    group_matched_rank: list[int | None] = [None] * total
    for source in sources:
        rank = int(source["rank"])
        chunk_text = str(source["text"])
        source_document_id = str(source.get("document_id", ""))
        chunk_page = source.get("page_number")
        chunk_page_number = (
            int(chunk_page) if chunk_page is not None else None
        )
        for index, group in enumerate(groups):
            if group_matched_rank[index] is not None:
                continue
            if group.document_id != source_document_id:
                continue
            if matches_expectation(chunk_text, chunk_page_number, group):
                group_matched_rank[index] = rank

    matched_ranks = [
        rank for rank in group_matched_rank if rank is not None
    ]

    def count_within(k: int) -> int:
        return len([rank for rank in matched_ranks if rank <= k])

    first_relevant_rank = min(matched_ranks) if matched_ranks else None
    return EvidenceMatch(
        first_relevant_rank=first_relevant_rank,
        matched_evidence_count_at_1=count_within(1),
        matched_evidence_count_at_3=count_within(3),
        matched_evidence_count_at_5=count_within(5),
        total_evidence_count=total,
        hit_at_1=count_within(1) > 0,
        hit_at_3=count_within(3) > 0,
        hit_at_5=count_within(5) > 0,
        reciprocal_rank=(
            1.0 / first_relevant_rank if first_relevant_rank else 0.0
        ),
    )
