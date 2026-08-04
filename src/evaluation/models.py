"""Domain models for offline RAG evaluation."""

from dataclasses import dataclass, field
from typing import Literal


class EvaluationError(Exception):
    """Base error for invalid evaluation data or failed runs."""


class EvaluationDataError(EvaluationError):
    """Raised when the evaluation dataset or manifest is invalid."""


@dataclass(frozen=True)
class EvidenceExpectation:
    """Describe one expected evidence group for a question."""

    document_id: str
    page_number: int | None
    required_terms: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCase:
    """Describe one labeled evaluation question."""

    id: str
    split: Literal["calibration", "test"]
    question: str
    answerable: bool
    category: str
    difficulty: str
    expected_evidence: tuple[EvidenceExpectation, ...]
    notes: str


@dataclass(frozen=True)
class EvaluationResult:
    """Describe one question's retrieval and decision outcome."""

    case_id: str
    split: str
    question: str
    answerable: bool
    category: str
    difficulty: str
    max_relevance_score: float | None
    first_relevant_rank: int | None
    matched_evidence_count_at_1: int
    matched_evidence_count_at_3: int
    matched_evidence_count_at_5: int
    total_evidence_count: int
    hit_at_1: bool | None
    hit_at_3: bool | None
    hit_at_5: bool | None
    reciprocal_rank: float | None
    sources: tuple[dict[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CorpusDocument:
    """Describe one evaluation corpus document from the manifest."""

    document_id: str
    path: str
    filename: str
    content_type: str
