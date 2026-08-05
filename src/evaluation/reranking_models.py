"""Immutable data models for the A/B reranking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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

    The three max-retrieval-score fields are derived independently:

    * ``max_retrieval_score`` — the highest vector retrieval score in the
      full candidate pool (alias: ``candidate_max_retrieval_score``).
    * ``vector_max_retrieval_score`` — the highest vector retrieval score in
      the vector-only branch's full candidate view (the same pool).
    * ``reranked_max_retrieval_score`` — the highest vector retrieval score
      in the complete reranked candidate list **before** truncation.

    ``vector_max_retrieval_score`` and ``reranked_max_retrieval_score`` must
    be derived independently; when the reranking stage loses candidates or
    corrupts retrieval scores they diverge and the decision-invariance check
    fails, which is exactly the drift the check exists to detect.
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
    vector_max_retrieval_score: float | None
    reranked_max_retrieval_score: float | None
    vector_decision_at_comparison: bool | None
    reranked_decision_at_comparison: bool | None
    vector_decision_at_recommended: bool | None
    reranked_decision_at_recommended: bool | None
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

    @property
    def candidate_max_retrieval_score(self) -> float | None:
        """Alias for ``max_retrieval_score`` (candidate pool max)."""
        return self.max_retrieval_score


@dataclass(frozen=True)
class DecisionInvarianceResult:
    """Result of checking refusal-decision invariance for one case (or aggregate).

    The refusal decision is computed from ``has_sufficient_context`` over the
    highest **vector** retrieval score of each branch.  The check is real: it
    computes the decision independently for the vector-only and reranked views
    at both the comparison and recommended thresholds and fails when they
    disagree.  The runner derives the two branch scores independently from the
    full candidate pool, so data-flow drift (lost candidates, corrupted
    retrieval scores) surfaces as a genuine failure.
    """

    passed: bool
    inconsistent_case_ids: tuple[str, ...] = ()
    vector_decision_at_comparison: bool | None = None
    reranked_decision_at_comparison: bool | None = None
    vector_decision_at_recommended: bool | None = None
    reranked_decision_at_recommended: bool | None = None


@dataclass(frozen=True)
class RankChangeClassification:
    """Exclusive classification of one answerable case's rank change.

    Every answerable case belongs to exactly one of ``improved`` /
    ``regressed`` / ``unchanged``.  ``rank_change`` is only computed when both
    branches have a relevant rank; it stays ``None`` when either branch
    misses (no fabricated ranks are used).
    """

    rank_change: int | None
    improved: bool
    regressed: bool
    unchanged: bool


def classify_rank_change(
    vector_rank: int | None,
    reranked_rank: int | None,
    *,
    answerable: bool,
) -> RankChangeClassification:
    """Classify one case into exactly one of improved / regressed / unchanged.

    Rules:
    * unanswerable -> none of the three (does not enter the rank sets)
    * both miss (``None``/``None``) -> unchanged
    * vector miss, reranked hit -> improved
    * vector hit, reranked miss -> regressed
    * both hit: reranked rank < vector rank -> improved
    * both hit: reranked rank > vector rank -> regressed
    * both hit, equal ranks -> unchanged
    """
    if not answerable:
        return RankChangeClassification(None, False, False, False)
    if vector_rank is None and reranked_rank is None:
        return RankChangeClassification(None, False, False, True)
    if vector_rank is None and reranked_rank is not None:
        return RankChangeClassification(None, True, False, False)
    if vector_rank is not None and reranked_rank is None:
        return RankChangeClassification(None, False, True, False)
    if reranked_rank < vector_rank:
        return RankChangeClassification(
            reranked_rank - vector_rank, True, False, False
        )
    if reranked_rank > vector_rank:
        return RankChangeClassification(
            reranked_rank - vector_rank, False, True, False
        )
    return RankChangeClassification(0, False, False, True)


# ---------------------------------------------------------------------------
# Reranker run provenance
# ---------------------------------------------------------------------------


RerankerBackend = Literal["cross_encoder", "fake", "custom"]


@dataclass(frozen=True)
class RerankerRunIdentity:
    """Explicit provenance of the reranker actually used in a run.

    This is a typed field, never inferred from the model-name string.  Only
    the default CLI factory that successfully constructs a real
    ``CrossEncoderReranker`` yields ``backend="cross_encoder"`` and
    ``real_model_run=True``; every test substitute or custom factory defaults
    to ``real_model_run=False`` so it can never justify a production enable
    recommendation.
    """

    backend: RerankerBackend
    real_model_run: bool
    model_name: str
    model_revision: str | None = None


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
