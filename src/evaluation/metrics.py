"""Compute retrieval quality and answer/refusal decision metrics."""

import statistics
from dataclasses import dataclass

from src.evaluation.models import RETRIEVAL_K_VALUES, EvaluationResult
from src.exceptions import EvaluationError
from src.rag_service import has_sufficient_context


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregate Hit@K, Recall@K, and MRR over answerable cases only."""

    case_count: int
    hit_at_1: float | None
    hit_at_3: float | None
    hit_at_5: float | None
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    mean_reciprocal_rank: float | None


@dataclass(frozen=True)
class DecisionMetrics:
    """Aggregate the answer/refusal confusion matrix at one threshold."""

    threshold: float
    case_count: int
    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int
    decision_accuracy: float
    answerable_precision: float | None
    answerable_recall: float | None
    false_answer_rate: float | None
    false_refusal_rate: float | None
    answer_rate: float
    refusal_rate: float


@dataclass(frozen=True)
class ScoreDistribution:
    """Summarize the max-relevance-score distribution of one case group."""

    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    p25: float | None
    p75: float | None


def answerable_results(
    results: tuple[EvaluationResult, ...],
) -> tuple[EvaluationResult, ...]:
    """Return only the answerable cases, which alone carry retrieval metrics."""
    return tuple(result for result in results if result.answerable)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def compute_retrieval_metrics(
    results: tuple[EvaluationResult, ...],
) -> RetrievalMetrics:
    """Compute macro-averaged retrieval metrics over answerable cases.

    Unanswerable cases have no ground-truth evidence, so they are excluded
    from every numerator and denominator instead of counting as failures.
    """
    scoped = answerable_results(results)
    hits: dict[int, list[float]] = {k: [] for k in RETRIEVAL_K_VALUES}
    recalls: dict[int, list[float]] = {k: [] for k in RETRIEVAL_K_VALUES}
    reciprocal_ranks: list[float] = []

    for result in scoped:
        for k in RETRIEVAL_K_VALUES:
            hit = result.hit_at(k)
            if hit is not None:
                hits[k].append(1.0 if hit else 0.0)
            recall = result.recall_at(k)
            if recall is not None:
                recalls[k].append(recall)
        if result.reciprocal_rank is not None:
            reciprocal_ranks.append(result.reciprocal_rank)

    return RetrievalMetrics(
        case_count=len(scoped),
        hit_at_1=_mean(hits[1]),
        hit_at_3=_mean(hits[3]),
        hit_at_5=_mean(hits[5]),
        recall_at_1=_mean(recalls[1]),
        recall_at_3=_mean(recalls[3]),
        recall_at_5=_mean(recalls[5]),
        mean_reciprocal_rank=_mean(reciprocal_ranks),
    )


def group_retrieval_metrics(
    results: tuple[EvaluationResult, ...],
    attribute: str,
) -> dict[str, RetrievalMetrics]:
    """Compute retrieval metrics per value of one result attribute."""
    grouped: dict[str, list[EvaluationResult]] = {}
    for result in answerable_results(results):
        key = str(getattr(result, attribute))
        grouped.setdefault(key, []).append(result)
    return {
        key: compute_retrieval_metrics(tuple(values))
        for key, values in sorted(grouped.items())
    }


def reciprocal_rank_from_rank(rank: int | None) -> float:
    """Return 1/rank for a hit, or 0.0 when no evidence was retrieved."""
    if rank is None:
        return 0.0
    if rank <= 0:
        raise ValueError("rank 必须大于 0")
    return 1.0 / rank


def predict_answerable(result: EvaluationResult, threshold: float) -> bool:
    """Reuse the production boundary so evaluation cannot drift from runtime."""
    return has_sufficient_context(result.max_relevance_score, threshold)


def compute_decision_metrics(
    results: tuple[EvaluationResult, ...],
    threshold: float,
) -> DecisionMetrics:
    """Compute the confusion matrix and derived rates at one threshold.

    Rates whose denominator is zero are reported as ``None`` instead of a
    silently wrong number. An empty result set is rejected outright.
    """
    if not results:
        raise EvaluationError("决策指标需要至少一个评估结果")

    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    for result in results:
        predicted = predict_answerable(result, threshold)
        if result.answerable and predicted:
            true_positive += 1
        elif result.answerable and not predicted:
            false_negative += 1
        elif not result.answerable and not predicted:
            true_negative += 1
        else:
            false_positive += 1

    total = len(results)
    actual_answerable = true_positive + false_negative
    actual_unanswerable = true_negative + false_positive
    predicted_answerable = true_positive + false_positive

    return DecisionMetrics(
        threshold=float(threshold),
        case_count=total,
        true_positive=true_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        false_positive=false_positive,
        decision_accuracy=(true_positive + true_negative) / total,
        answerable_precision=(
            true_positive / predicted_answerable
            if predicted_answerable
            else None
        ),
        answerable_recall=(
            true_positive / actual_answerable if actual_answerable else None
        ),
        false_answer_rate=(
            false_positive / actual_unanswerable if actual_unanswerable else None
        ),
        false_refusal_rate=(
            false_negative / actual_answerable if actual_answerable else None
        ),
        answer_rate=predicted_answerable / total,
        refusal_rate=(false_negative + true_negative) / total,
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Return a linear-interpolated percentile from a sorted value list."""
    if not sorted_values:
        raise ValueError("分位数需要至少一个数值")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * weight


def summarize_scores(scores: list[float]) -> ScoreDistribution:
    """Summarize a score list with the standard library only."""
    values = sorted(float(score) for score in scores)
    if not values:
        return ScoreDistribution(
            count=0,
            minimum=None,
            maximum=None,
            mean=None,
            median=None,
            p25=None,
            p75=None,
        )
    return ScoreDistribution(
        count=len(values),
        minimum=values[0],
        maximum=values[-1],
        mean=statistics.fmean(values),
        median=statistics.median(values),
        p25=_percentile(values, 0.25),
        p75=_percentile(values, 0.75),
    )


def score_distributions(
    results: tuple[EvaluationResult, ...],
) -> dict[str, ScoreDistribution]:
    """Summarize max scores separately for answerable and unanswerable cases."""
    answerable_scores = [
        result.max_relevance_score
        for result in results
        if result.answerable and result.max_relevance_score is not None
    ]
    unanswerable_scores = [
        result.max_relevance_score
        for result in results
        if not result.answerable and result.max_relevance_score is not None
    ]
    return {
        "answerable": summarize_scores(answerable_scores),
        "unanswerable": summarize_scores(unanswerable_scores),
    }
