"""Immutable data models shared by the offline evaluation toolkit."""

from dataclasses import dataclass
from typing import Literal

Split = Literal["calibration", "test"]

SPLITS: tuple[str, ...] = ("calibration", "test")
CATEGORIES: tuple[str, ...] = (
    "direct",
    "paraphrase",
    "multi_evidence",
    "out_of_scope_far",
    "out_of_scope_near",
)
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
MULTI_EVIDENCE_CATEGORY = "multi_evidence"
RETRIEVAL_K_VALUES: tuple[int, ...] = (1, 3, 5)
MIN_REQUIRED_TOP_K = 5


@dataclass(frozen=True)
class EvidenceExpectation:
    """Describe one labelled evidence group that a chunk may satisfy."""

    document_id: str
    page_number: int | None
    required_terms: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCase:
    """Describe one labelled evaluation question."""

    id: str
    split: Split
    question: str
    answerable: bool
    category: str
    difficulty: str
    expected_evidence: tuple[EvidenceExpectation, ...]
    notes: str


@dataclass(frozen=True)
class RetrievedSource:
    """Describe one retrieved chunk exactly as the production index returns it."""

    rank: int
    score: float
    text: str
    chunk_index: int
    document_id: str
    filename: str
    page_number: int | None


@dataclass(frozen=True)
class EvaluationResult:
    """Describe the retrieval outcome of one evaluation case.

    Retrieval fields are ``None`` for unanswerable cases so they never enter
    a Hit@K, Recall@K, or MRR denominator.
    """

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
    sources: tuple[RetrievedSource, ...]

    def matched_evidence_count_at(self, k: int) -> int:
        """Return the number of distinct evidence groups matched within Top-K."""
        counts = {
            1: self.matched_evidence_count_at_1,
            3: self.matched_evidence_count_at_3,
            5: self.matched_evidence_count_at_5,
        }
        if k not in counts:
            raise ValueError(f"不支持的 K 值: {k}")
        return counts[k]

    def hit_at(self, k: int) -> bool | None:
        """Return the Hit@K flag, or None when the case is unanswerable."""
        flags = {
            1: self.hit_at_1,
            3: self.hit_at_3,
            5: self.hit_at_5,
        }
        if k not in flags:
            raise ValueError(f"不支持的 K 值: {k}")
        return flags[k]

    def recall_at(self, k: int) -> float | None:
        """Return evidence-group recall within Top-K for answerable cases."""
        if not self.answerable or self.total_evidence_count == 0:
            return None
        return self.matched_evidence_count_at(k) / self.total_evidence_count
