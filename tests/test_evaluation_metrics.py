"""Offline tests for retrieval and decision metrics."""

import pytest

from src.evaluation.metrics import (
    aggregate_retrieval_by,
    decision_metrics,
    retrieval_metrics,
    score_distribution,
)

from evaluation_helpers import make_result


def _answerable(
    case_id: str,
    *,
    hit_1: bool = True,
    hit_3: bool = True,
    hit_5: bool = True,
    matched: int = 1,
    total: int = 1,
    rank: int | None = 1,
    max_score: float = 0.6,
    category: str = "direct",
    difficulty: str = "easy",
    split: str = "calibration",
) -> object:
    return make_result(
        case_id,
        split=split,
        answerable=True,
        category=category,
        difficulty=difficulty,
        max_score=max_score,
        first_relevant_rank=rank,
        matched_at_1=matched if hit_1 else 0,
        matched_at_3=matched if hit_3 else 0,
        matched_at_5=matched if hit_5 else 0,
        total_evidence=total,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
    )


def _unanswerable(
    case_id: str,
    *,
    max_score: float = 0.1,
    split: str = "calibration",
) -> object:
    return make_result(
        case_id,
        split=split,
        answerable=False,
        max_score=max_score,
        matched_at_1=0,
        matched_at_3=0,
        matched_at_5=0,
        total_evidence=0,
        hit_at_1=None,
        hit_at_3=None,
        hit_at_5=None,
        reciprocal_rank=None,
    )


def test_hit_at_1() -> None:
    results = [_answerable("a", hit_1=True), _answerable("b", hit_1=False)]
    assert retrieval_metrics(results)["hit_at_1"] == 0.5


def test_hit_at_3() -> None:
    results = [
        _answerable("a", hit_3=True, hit_1=False, rank=3),
        _answerable("b", hit_3=False, hit_1=False),
    ]
    assert retrieval_metrics(results)["hit_at_3"] == 0.5


def test_hit_at_5() -> None:
    results = [
        _answerable("a", hit_5=True, hit_1=False, hit_3=False, rank=5),
        _answerable("b", hit_5=False, hit_1=False, hit_3=False),
    ]
    assert retrieval_metrics(results)["hit_at_5"] == 0.5


def test_no_hits_yields_zero_rates() -> None:
    results = [
        _answerable("a", hit_1=False, hit_3=False, hit_5=False, rank=None)
    ]
    metrics = retrieval_metrics(results)
    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 0.0
    assert metrics["hit_at_5"] == 0.0
    assert metrics["mrr"] == 0.0


def test_recall_at_k() -> None:
    results = [
        _answerable("a", matched=1, total=2, rank=1),
        _answerable("b", matched=2, total=2, rank=2),
    ]
    metrics = retrieval_metrics(results)
    assert metrics["recall_at_1"] == pytest.approx(0.75)
    assert metrics["recall_at_5"] == pytest.approx(0.75)


def test_mrr_rank_one() -> None:
    results = [_answerable("a", rank=1), _answerable("b", rank=1)]
    assert retrieval_metrics(results)["mrr"] == 1.0


def test_mrr_rank_two() -> None:
    results = [_answerable("a", rank=2, hit_1=False)]
    assert retrieval_metrics(results)["mrr"] == 0.5


def test_mrr_rank_three() -> None:
    results = [_answerable("a", rank=3, hit_1=False, hit_3=False, hit_5=True)]
    assert retrieval_metrics(results)["mrr"] == pytest.approx(1 / 3)


def test_mrr_zero_without_hits() -> None:
    results = [_answerable("a", rank=None, hit_1=False, hit_3=False, hit_5=False)]
    assert retrieval_metrics(results)["mrr"] == 0.0


def test_unanswerable_cases_excluded_from_retrieval_denominator() -> None:
    results = [
        _answerable("a"),
        _unanswerable("u1"),
        _unanswerable("u2"),
    ]
    metrics = retrieval_metrics(results)
    assert metrics["answerable_count"] == 1
    assert metrics["hit_at_1"] == 1.0


def test_category_aggregation() -> None:
    results = [
        _answerable("a", category="paraphrase"),
        _answerable("b", category="direct"),
    ]
    grouped = aggregate_retrieval_by(results, "category")
    assert set(grouped.keys()) == {"direct", "paraphrase"}
    assert grouped["paraphrase"]["hit_at_1"] == 1.0


def test_difficulty_aggregation() -> None:
    results = [
        _answerable("a", difficulty="hard"),
        _answerable("b", difficulty="easy"),
    ]
    grouped = aggregate_retrieval_by(results, "difficulty")
    assert set(grouped.keys()) == {"easy", "hard"}
    assert grouped["hard"]["hit_at_1"] == 1.0


def test_decision_tp() -> None:
    results = [_answerable("a", max_score=0.6)]
    metrics = decision_metrics(results, 0.35)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["tn"] == 0
    assert metrics["fn"] == 0


def test_decision_tn() -> None:
    results = [_unanswerable("u", max_score=0.1)]
    metrics = decision_metrics(results, 0.35)
    assert metrics["tn"] == 1
    assert metrics["tp"] == 0


def test_decision_fp() -> None:
    results = [_unanswerable("u", max_score=0.6)]
    metrics = decision_metrics(results, 0.35)
    assert metrics["fp"] == 1
    assert metrics["false_answer_rate"] == 1.0


def test_decision_fn() -> None:
    results = [_answerable("a", max_score=0.1)]
    metrics = decision_metrics(results, 0.35)
    assert metrics["fn"] == 1
    assert metrics["false_refusal_rate"] == 1.0


def test_decision_accuracy() -> None:
    results = [
        _answerable("a", max_score=0.6),
        _unanswerable("u", max_score=0.1),
        _unanswerable("v", max_score=0.9),
    ]
    metrics = decision_metrics(results, 0.35)
    assert metrics["decision_accuracy"] == pytest.approx(2 / 3)
    assert metrics["answer_rate"] == pytest.approx(2 / 3)
    assert metrics["refusal_rate"] == pytest.approx(1 / 3)


def test_decision_rates_with_zero_denominator() -> None:
    all_answerable = [_answerable("a", max_score=0.1)]
    metrics = decision_metrics(all_answerable, 0.35)
    assert metrics["false_answer_rate"] is None
    assert metrics["answerable_precision"] is None


def test_score_equal_to_threshold_answers() -> None:
    results = [_answerable("a", max_score=0.35)]
    assert decision_metrics(results, 0.35)["tp"] == 1


def test_empty_results_refuse() -> None:
    metrics = decision_metrics([_unanswerable("u", max_score=None)], 0.35)
    assert metrics["tn"] == 1


def test_non_finite_score_is_treated_as_refusal() -> None:
    import math

    results = [_answerable("a", max_score=float("nan"))]
    metrics = decision_metrics(results, 0.35)
    assert metrics["fn"] == 1

    infinite = [_answerable("b", max_score=float("inf"))]
    assert decision_metrics(infinite, 0.35)["fn"] == 1


def test_score_distribution_separates_labels() -> None:
    results = [
        _answerable("a", max_score=0.8),
        _answerable("b", max_score=0.6),
        _unanswerable("u", max_score=0.2),
        _unanswerable("v", max_score=0.4),
    ]
    distribution = score_distribution(results)
    assert distribution["answerable"]["count"] == 2
    assert distribution["answerable"]["median"] == 0.7
    assert distribution["answerable"]["min"] == 0.6
    assert distribution["answerable"]["max"] == 0.8
    assert distribution["unanswerable"]["count"] == 2
    assert distribution["unanswerable"]["median"] == 0.3
    assert distribution["unanswerable"]["p25"] < distribution["unanswerable"]["p75"]
    assert distribution["answerable"]["mean"] > distribution["unanswerable"]["mean"]
