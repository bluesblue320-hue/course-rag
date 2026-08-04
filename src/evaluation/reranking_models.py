"""Immutable data models for the A/B reranking evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateResult:
    """Vector retrieval metrics computed over the full candidate pool.

    These tell us whether the correct evidence entered the candidate pool
    at all, before the reranker has a chance to re-order anything.
    """

    first_relevant_rank: int | None
    hit_at_5: bool | None
    hit_at_10: bool | None
    hit_at_15: bool | None
    recall_at_5: float | None
    recall_at_10: float | None
    recall_at_15: float | None
    reciprocal_rank: float | None


@dataclass(frozen=True)
class BranchResult:
    """Retrieval metrics for one branch (vector-only or reranked).

    Uses the same K values (1, 3, 5) as the baseline evaluation so the
    two branches are directly comparable.
    """

    first_relevant_rank: int | None
    hit_at_1: bool | None
    hit_at_3: bool | None
    hit_at_5: bool | None
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    reciprocal_rank: float | None
    sources: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class RerankingCaseResult:
    """Per-case A/B comparison result.

    For unanswerable cases, all retrieval metrics are ``None`` and the case
    does not enter any Hit@K / Recall@K / MRR denominator.
    """

    case_id: str
    split: str
    category: str
    difficulty: str
    answerable: bool
    question: str

    # Candidate pool (Top-15 vector retrieval)
    candidate: CandidateResult | None
    max_retrieval_score: float | None
    candidate_count: int

    # Vector-only branch (first final_top_k of the candidate pool)
    vector: BranchResult | None

    # Reranked branch (candidate pool re-scored, then truncated)
    reranked: BranchResult | None
    reranker_applied: bool
    reranker_fallback: bool

    # Rank change: reranked_first_relevant_rank - vector_first_relevant_rank
    # Negative = improved (moved up), positive = regressed (moved down)
    rank_change: int | None
    improved: bool
    regressed: bool
    unchanged: bool


@dataclass(frozen=True)
class LatencyRecord:
    """One timing sample for latency measurement."""

    vector_retrieval_ms: float
    rerank_ms: float | None
    end_to_end_ms: float


@dataclass(frozen=True)
class LatencySummary:
    """Aggregate latency statistics for one stage."""

    count: int
    mean: float | None
    median: float | None
    p95: float | None

    @staticmethod
    def empty() -> "LatencySummary":
        return LatencySummary(count=0, mean=None, median=None, p95=None)
