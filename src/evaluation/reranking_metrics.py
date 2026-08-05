"""Compute A/B comparison metrics for the reranking evaluation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from src.evaluation.reranking_models import (
    BranchResult,
    CandidateResult,
    LatencyRecord,
    LatencySummary,
    RerankingCaseResult,
)


@dataclass(frozen=True)
class BranchMetrics:
    """Aggregate Hit@K, Recall@K, and MRR over answerable cases."""

    case_count: int
    hit_at_1: float | None
    hit_at_3: float | None
    hit_at_5: float | None
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    mean_reciprocal_rank: float | None


@dataclass(frozen=True)
class CandidateMetrics:
    """Aggregate candidate-pool metrics."""

    case_count: int
    hit_at_5: float | None
    hit_at_10: float | None
    hit_at_15: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    recall_at_15: float | None
    mean_reciprocal_rank: float | None


@dataclass(frozen=True)
class MetricDelta:
    """Absolute and relative difference between two metric values."""

    vector_value: float | None
    reranked_value: float | None
    absolute_delta: float | None
    relative_delta: float | None


@dataclass(frozen=True)
class MetricDeltas:
    """All metric deltas for the overall comparison."""

    hit_at_1: MetricDelta
    hit_at_3: MetricDelta
    hit_at_5: MetricDelta
    recall_at_1: MetricDelta
    recall_at_3: MetricDelta
    recall_at_5: MetricDelta
    mean_reciprocal_rank: MetricDelta


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _compute_branch_metrics(
    results: Sequence[RerankingCaseResult],
    branch_selector: str,
) -> BranchMetrics:
    """Compute aggregate metrics for one branch (vector or reranked)."""
    hits: dict[int, list[float]] = {1: [], 3: [], 5: []}
    recalls: dict[int, list[float]] = {1: [], 3: [], 5: []}
    reciprocal_ranks: list[float] = []
    count = 0

    for case in results:
        if not case.answerable:
            continue
        branch: BranchResult | None = (
            case.vector if branch_selector == "vector" else case.reranked
        )
        if branch is None:
            continue
        count += 1
        for k, hit_key in [(1, "hit_at_1"), (3, "hit_at_3"), (5, "hit_at_5")]:
            hit = getattr(branch, hit_key)
            if hit is not None:
                hits[k].append(1.0 if hit else 0.0)
            recall = getattr(branch, f"recall_at_{k}")
            if recall is not None:
                recalls[k].append(recall)
        if branch.reciprocal_rank is not None:
            reciprocal_ranks.append(branch.reciprocal_rank)

    return BranchMetrics(
        case_count=count,
        hit_at_1=_mean(hits[1]),
        hit_at_3=_mean(hits[3]),
        hit_at_5=_mean(hits[5]),
        recall_at_1=_mean(recalls[1]),
        recall_at_3=_mean(recalls[3]),
        recall_at_5=_mean(recalls[5]),
        mean_reciprocal_rank=_mean(reciprocal_ranks),
    )


def compute_vector_metrics(
    results: Sequence[RerankingCaseResult],
) -> BranchMetrics:
    """Compute metrics for the vector-only branch."""
    return _compute_branch_metrics(results, "vector")


def compute_reranked_metrics(
    results: Sequence[RerankingCaseResult],
) -> BranchMetrics:
    """Compute metrics for the reranked branch."""
    return _compute_branch_metrics(results, "reranked")


def compute_candidate_metrics(
    results: Sequence[RerankingCaseResult],
) -> CandidateMetrics:
    """Compute candidate-pool metrics."""
    hits: dict[int, list[float]] = {5: [], 10: [], 15: []}
    recalls: dict[int, list[float]] = {5: [], 10: [], 15: []}
    reciprocal_ranks: list[float] = []
    count = 0

    for case in results:
        if not case.answerable or case.candidate is None:
            continue
        count += 1
        for k in (5, 10, 15):
            hit = getattr(case.candidate, f"hit_at_{k}")
            if hit is not None:
                hits[k].append(1.0 if hit else 0.0)
            recall = getattr(case.candidate, f"recall_at_{k}")
            if recall is not None:
                recalls[k].append(recall)
        if case.candidate.reciprocal_rank is not None:
            reciprocal_ranks.append(case.candidate.reciprocal_rank)

    return CandidateMetrics(
        case_count=count,
        hit_at_5=_mean(hits[5]),
        hit_at_10=_mean(hits[10]),
        hit_at_15=_mean(hits[15]),
        recall_at_5=_mean(recalls[5]),
        recall_at_10=_mean(recalls[10]),
        recall_at_15=_mean(recalls[15]),
        mean_reciprocal_rank=_mean(reciprocal_ranks),
    )


def _make_delta(
    vector_value: float | None,
    reranked_value: float | None,
) -> MetricDelta:
    """Compute absolute and relative deltas.

    Relative delta is ``None`` (not Infinity) when the vector value is 0.
    """
    if vector_value is None or reranked_value is None:
        return MetricDelta(
            vector_value=vector_value,
            reranked_value=reranked_value,
            absolute_delta=None,
            relative_delta=None,
        )
    absolute = reranked_value - vector_value
    relative = absolute / vector_value if vector_value != 0 else None
    return MetricDelta(
        vector_value=vector_value,
        reranked_value=reranked_value,
        absolute_delta=absolute,
        relative_delta=relative,
    )


def compute_metric_deltas(
    vector: BranchMetrics,
    reranked: BranchMetrics,
) -> MetricDeltas:
    """Compute deltas between vector-only and reranked branches."""
    return MetricDeltas(
        hit_at_1=_make_delta(vector.hit_at_1, reranked.hit_at_1),
        hit_at_3=_make_delta(vector.hit_at_3, reranked.hit_at_3),
        hit_at_5=_make_delta(vector.hit_at_5, reranked.hit_at_5),
        recall_at_1=_make_delta(vector.recall_at_1, reranked.recall_at_1),
        recall_at_3=_make_delta(vector.recall_at_3, reranked.recall_at_3),
        recall_at_5=_make_delta(vector.recall_at_5, reranked.recall_at_5),
        mean_reciprocal_rank=_make_delta(
            vector.mean_reciprocal_rank,
            reranked.mean_reciprocal_rank,
        ),
    )


def group_branch_metrics(
    results: Sequence[RerankingCaseResult],
    branch_selector: str,
    attribute: str,
) -> dict[str, BranchMetrics]:
    """Compute branch metrics per value of one result attribute."""
    grouped: dict[str, list[RerankingCaseResult]] = {}
    for case in results:
        if not case.answerable:
            continue
        branch = (
            case.vector if branch_selector == "vector" else case.reranked
        )
        if branch is None:
            continue
        key = str(getattr(case, attribute))
        grouped.setdefault(key, []).append(case)
    return {
        key: _compute_branch_metrics(tuple(values), branch_selector)
        for key, values in sorted(grouped.items())
    }


def summarize_latency(
    records: Sequence[LatencyRecord],
    field: str,
) -> LatencySummary:
    """Compute mean / median / p95 for one latency field."""
    values = [
        float(getattr(r, field))
        for r in records
        if getattr(r, field) is not None
    ]
    if not values:
        return LatencySummary.empty()
    sorted_values = sorted(values)
    n = len(sorted_values)
    p95_index = min(int(0.95 * (n - 1)), n - 1)
    return LatencySummary(
        count=n,
        mean=statistics.fmean(values),
        median=statistics.median(values),
        p95=sorted_values[p95_index],
    )
