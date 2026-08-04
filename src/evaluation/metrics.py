"""Retrieval and answer/refusal decision metrics."""

import math
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

from src.rag_service import has_sufficient_context

from .models import EvaluationResult


def _safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def retrieval_metrics(
    results: Sequence[EvaluationResult],
) -> dict[str, Any]:
    """Compute Hit@K, Recall@K, and MRR over answerable cases only."""
    answerable = [result for result in results if result.answerable]
    hit_1: list[bool] = []
    hit_3: list[bool] = []
    hit_5: list[bool] = []
    recall_1: list[float] = []
    recall_3: list[float] = []
    recall_5: list[float] = []
    reciprocal_ranks: list[float] = []

    for result in answerable:
        total = result.total_evidence_count
        if total <= 0:
            continue
        if result.hit_at_1 is not None:
            hit_1.append(result.hit_at_1)
        if result.hit_at_3 is not None:
            hit_3.append(result.hit_at_3)
        if result.hit_at_5 is not None:
            hit_5.append(result.hit_at_5)
        recall_1.append(result.matched_evidence_count_at_1 / total)
        recall_3.append(result.matched_evidence_count_at_3 / total)
        recall_5.append(result.matched_evidence_count_at_5 / total)
        reciprocal_ranks.append(
            float(result.reciprocal_rank or 0.0)
        )

    def hit_rate(values: list[bool]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    return {
        "hit_at_1": hit_rate(hit_1),
        "hit_at_3": hit_rate(hit_3),
        "hit_at_5": hit_rate(hit_5),
        "recall_at_1": _safe_mean(recall_1),
        "recall_at_3": _safe_mean(recall_3),
        "recall_at_5": _safe_mean(recall_5),
        "mrr": _safe_mean(reciprocal_ranks),
        "answerable_count": len(answerable),
    }


def decision_metrics(
    results: Sequence[EvaluationResult],
    threshold: float,
) -> dict[str, Any]:
    """Compute the answer/refusal confusion matrix at one threshold."""
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for result in results:
        predicted = has_sufficient_context(
            result.max_relevance_score,
            threshold,
        )
        if result.answerable and predicted:
            tp += 1
        elif result.answerable and not predicted:
            fn += 1
        elif not result.answerable and not predicted:
            tn += 1
        else:
            fp += 1

    total = len(results)
    actual_answerable = tp + fn
    actual_unanswerable = fp + tn

    def rate(numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            return None
        return round(numerator / denominator, 6)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "decision_accuracy": rate(tp + tn, total),
        "answerable_precision": rate(tp, tp + fp),
        "answerable_recall": rate(tp, actual_answerable),
        "false_answer_rate": rate(fp, actual_unanswerable),
        "false_refusal_rate": rate(fn, actual_answerable),
        "answer_rate": rate(tp + fp, total),
        "refusal_rate": rate(fn + tn, total),
    }


def aggregate_retrieval_by(
    results: Sequence[EvaluationResult],
    key_name: str,
) -> dict[str, dict[str, Any]]:
    """Split retrieval metrics by a case attribute such as category."""
    groups: dict[str, list[EvaluationResult]] = {}
    for result in results:
        groups.setdefault(getattr(result, key_name), []).append(result)
    return {
        key: retrieval_metrics(group)
        for key, group in sorted(groups.items())
    }


def score_distribution(
    results: Sequence[EvaluationResult],
) -> dict[str, Any]:
    """Summarize max relevance scores separately by label."""
    scores_by_label: dict[str, list[float]] = {
        "answerable": [],
        "unanswerable": [],
    }
    for result in results:
        if result.max_relevance_score is None:
            continue
        bucket = "answerable" if result.answerable else "unanswerable"
        scores_by_label[bucket].append(float(result.max_relevance_score))

    def summarize(scores: list[float]) -> dict[str, Any]:
        if not scores:
            return {"count": 0}
        ordered = sorted(scores)
        return {
            "count": len(ordered),
            "min": round(ordered[0], 6),
            "max": round(ordered[-1], 6),
            "mean": round(statistics.fmean(ordered), 6),
            "median": round(statistics.median(ordered), 6),
            "p25": round(statistics.quantiles(ordered, n=4)[0], 6),
            "p75": round(statistics.quantiles(ordered, n=4)[2], 6),
        }

    return {
        "answerable": summarize(scores_by_label["answerable"]),
        "unanswerable": summarize(scores_by_label["unanswerable"]),
    }


def is_finite_score(value: float | None) -> bool:
    """Return whether a score is a finite number."""
    return value is not None and math.isfinite(value)
