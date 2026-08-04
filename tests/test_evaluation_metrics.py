"""Tests for retrieval metrics, decision metrics, and score distributions."""

import pytest

from src.evaluation.metrics import (
    answerable_results,
    compute_decision_metrics,
    compute_retrieval_metrics,
    group_retrieval_metrics,
    predict_answerable,
    reciprocal_rank_from_rank,
    score_distributions,
    summarize_scores,
)
from src.exceptions import EvaluationError
from tests.evaluation_helpers import make_result


class TestAnswerableResults:
    def test_it_keeps_only_answerable_cases(self) -> None:
        answerable = make_result(case_id="a", answerable=True)
        unanswerable = make_result(case_id="b", answerable=False)

        assert answerable_results((answerable, unanswerable)) == (answerable,)

    def test_it_returns_empty_when_nothing_is_answerable(self) -> None:
        assert answerable_results((make_result(answerable=False),)) == ()


class TestComputeRetrievalMetrics:
    def test_it_ignores_unanswerable_cases_in_every_denominator(self) -> None:
        results = (
            make_result(case_id="a", answerable=True, matched=(1, 1, 1)),
            make_result(case_id="b", answerable=False),
            make_result(case_id="c", answerable=False),
        )

        metrics = compute_retrieval_metrics(results)

        assert metrics.case_count == 1
        assert metrics.hit_at_1 == 1.0
        assert metrics.hit_at_5 == 1.0

    def test_it_macro_averages_hits_across_cases(self) -> None:
        results = (
            make_result(case_id="a", matched=(1, 1, 1), first_relevant_rank=1),
            make_result(case_id="b", matched=(0, 1, 1), first_relevant_rank=2),
            make_result(case_id="c", matched=(0, 0, 0), first_relevant_rank=None),
        )

        metrics = compute_retrieval_metrics(results)

        assert metrics.case_count == 3
        assert metrics.hit_at_1 == pytest.approx(1 / 3)
        assert metrics.hit_at_3 == pytest.approx(2 / 3)
        assert metrics.hit_at_5 == pytest.approx(2 / 3)
        assert metrics.mean_reciprocal_rank == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_it_macro_averages_recall_over_evidence_groups(self) -> None:
        results = (
            make_result(case_id="a", matched=(1, 2, 2), total_evidence_count=2),
            make_result(case_id="b", matched=(0, 1, 2), total_evidence_count=2),
        )

        metrics = compute_retrieval_metrics(results)

        assert metrics.recall_at_1 == pytest.approx((0.5 + 0.0) / 2)
        assert metrics.recall_at_3 == pytest.approx((1.0 + 0.5) / 2)
        assert metrics.recall_at_5 == pytest.approx(1.0)

    def test_it_returns_none_instead_of_dividing_by_zero(self) -> None:
        metrics = compute_retrieval_metrics(())

        assert metrics.case_count == 0
        assert metrics.hit_at_1 is None
        assert metrics.hit_at_3 is None
        assert metrics.hit_at_5 is None
        assert metrics.recall_at_1 is None
        assert metrics.recall_at_5 is None
        assert metrics.mean_reciprocal_rank is None

    def test_it_skips_recall_when_a_case_has_no_evidence_group(self) -> None:
        results = (
            make_result(case_id="a", matched=(0, 0, 0), total_evidence_count=0),
        )

        metrics = compute_retrieval_metrics(results)

        assert metrics.case_count == 1
        assert metrics.hit_at_1 == pytest.approx(0.0)
        assert metrics.recall_at_5 is None


class TestGroupRetrievalMetrics:
    def test_it_groups_by_the_requested_attribute(self) -> None:
        results = (
            make_result(case_id="a", category="direct", matched=(1, 1, 1)),
            make_result(case_id="b", category="paraphrase", matched=(0, 0, 0)),
            make_result(case_id="c", category="paraphrase", matched=(1, 1, 1)),
        )

        grouped = group_retrieval_metrics(results, "category")

        assert set(grouped) == {"direct", "paraphrase"}
        assert grouped["direct"].case_count == 1
        assert grouped["paraphrase"].case_count == 2
        assert grouped["paraphrase"].hit_at_1 == pytest.approx(0.5)

    def test_it_sorts_group_keys_for_stable_reports(self) -> None:
        results = (
            make_result(case_id="a", difficulty="medium"),
            make_result(case_id="b", difficulty="easy"),
            make_result(case_id="c", difficulty="hard"),
        )

        assert list(group_retrieval_metrics(results, "difficulty")) == [
            "easy",
            "hard",
            "medium",
        ]

    def test_it_excludes_unanswerable_cases_from_every_group(self) -> None:
        results = (
            make_result(case_id="a", category="direct"),
            make_result(case_id="b", category="out_of_scope_far", answerable=False),
        )

        grouped = group_retrieval_metrics(results, "category")

        assert set(grouped) == {"direct"}


class TestReciprocalRankFromRank:
    @pytest.mark.parametrize(
        ("rank", "expected"),
        [(1, 1.0), (2, 0.5), (4, 0.25), (None, 0.0)],
    )
    def test_it_maps_rank_to_reciprocal_rank(
        self,
        rank: int | None,
        expected: float,
    ) -> None:
        assert reciprocal_rank_from_rank(rank) == pytest.approx(expected)

    def test_it_rejects_a_non_positive_rank(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_from_rank(0)


class TestPredictAnswerable:
    def test_it_answers_when_the_score_reaches_the_threshold(self) -> None:
        result = make_result(max_relevance_score=0.35)

        assert predict_answerable(result, 0.35) is True

    def test_it_refuses_just_below_the_threshold(self) -> None:
        result = make_result(max_relevance_score=0.3499)

        assert predict_answerable(result, 0.35) is False

    def test_it_refuses_when_there_is_no_retrieval_score(self) -> None:
        result = make_result(max_relevance_score=None)

        assert predict_answerable(result, 0.35) is False


class TestComputeDecisionMetrics:
    def _mixed_results(self) -> tuple:
        return (
            # true positive: answerable and above the threshold
            make_result(case_id="tp", answerable=True, max_relevance_score=0.80),
            # false negative: answerable but below the threshold
            make_result(case_id="fn", answerable=True, max_relevance_score=0.10),
            # true negative: unanswerable and below the threshold
            make_result(case_id="tn", answerable=False, max_relevance_score=0.20),
            # false positive: unanswerable but above the threshold
            make_result(case_id="fp", answerable=False, max_relevance_score=0.90),
        )

    def test_it_builds_the_confusion_matrix(self) -> None:
        metrics = compute_decision_metrics(self._mixed_results(), 0.5)

        assert (
            metrics.true_positive,
            metrics.false_negative,
            metrics.true_negative,
            metrics.false_positive,
        ) == (1, 1, 1, 1)
        assert metrics.case_count == 4
        assert metrics.threshold == pytest.approx(0.5)

    def test_it_derives_every_rate_from_the_matrix(self) -> None:
        metrics = compute_decision_metrics(self._mixed_results(), 0.5)

        assert metrics.decision_accuracy == pytest.approx(0.5)
        assert metrics.answerable_precision == pytest.approx(0.5)
        assert metrics.answerable_recall == pytest.approx(0.5)
        assert metrics.false_answer_rate == pytest.approx(0.5)
        assert metrics.false_refusal_rate == pytest.approx(0.5)
        assert metrics.answer_rate == pytest.approx(0.5)
        assert metrics.refusal_rate == pytest.approx(0.5)

    def test_it_reports_none_when_no_case_is_unanswerable(self) -> None:
        results = (make_result(answerable=True, max_relevance_score=0.8),)

        metrics = compute_decision_metrics(results, 0.5)

        assert metrics.false_answer_rate is None
        assert metrics.answerable_recall == pytest.approx(1.0)

    def test_it_reports_none_when_no_case_is_answerable(self) -> None:
        results = (make_result(answerable=False, max_relevance_score=0.1),)

        metrics = compute_decision_metrics(results, 0.5)

        assert metrics.false_refusal_rate is None
        assert metrics.answerable_recall is None

    def test_it_reports_none_precision_when_nothing_is_answered(self) -> None:
        results = (
            make_result(case_id="a", answerable=True, max_relevance_score=0.1),
            make_result(case_id="b", answerable=False, max_relevance_score=0.1),
        )

        metrics = compute_decision_metrics(results, 0.5)

        assert metrics.answerable_precision is None
        assert metrics.answer_rate == pytest.approx(0.0)
        assert metrics.refusal_rate == pytest.approx(1.0)

    def test_a_higher_threshold_never_increases_false_answers(self) -> None:
        results = self._mixed_results()

        low = compute_decision_metrics(results, 0.15)
        high = compute_decision_metrics(results, 0.95)

        assert high.false_positive <= low.false_positive
        assert high.false_negative >= low.false_negative

    def test_it_rejects_an_empty_result_set(self) -> None:
        with pytest.raises(EvaluationError):
            compute_decision_metrics((), 0.35)


class TestSummarizeScores:
    def test_it_summarizes_a_score_list(self) -> None:
        distribution = summarize_scores([0.1, 0.2, 0.3, 0.4, 0.5])

        assert distribution.count == 5
        assert distribution.minimum == pytest.approx(0.1)
        assert distribution.maximum == pytest.approx(0.5)
        assert distribution.mean == pytest.approx(0.3)
        assert distribution.median == pytest.approx(0.3)
        assert distribution.p25 == pytest.approx(0.2)
        assert distribution.p75 == pytest.approx(0.4)

    def test_it_handles_a_single_value_without_interpolating(self) -> None:
        distribution = summarize_scores([0.42])

        assert distribution.count == 1
        assert distribution.p25 == pytest.approx(0.42)
        assert distribution.p75 == pytest.approx(0.42)

    def test_it_returns_all_none_for_an_empty_list(self) -> None:
        distribution = summarize_scores([])

        assert distribution.count == 0
        assert distribution.minimum is None
        assert distribution.maximum is None
        assert distribution.mean is None
        assert distribution.median is None
        assert distribution.p25 is None
        assert distribution.p75 is None

    def test_it_does_not_depend_on_input_order(self) -> None:
        ascending = summarize_scores([0.1, 0.5, 0.9])
        descending = summarize_scores([0.9, 0.5, 0.1])

        assert ascending == descending


class TestScoreDistributions:
    def test_it_splits_answerable_and_unanswerable_scores(self) -> None:
        results = (
            make_result(case_id="a", answerable=True, max_relevance_score=0.8),
            make_result(case_id="b", answerable=True, max_relevance_score=0.6),
            make_result(case_id="c", answerable=False, max_relevance_score=0.2),
        )

        distributions = score_distributions(results)

        assert set(distributions) == {"answerable", "unanswerable"}
        assert distributions["answerable"].count == 2
        assert distributions["unanswerable"].count == 1
        assert distributions["unanswerable"].maximum == pytest.approx(0.2)

    def test_it_skips_cases_without_a_score(self) -> None:
        results = (
            make_result(case_id="a", answerable=True, max_relevance_score=None),
            make_result(case_id="b", answerable=False, max_relevance_score=None),
        )

        distributions = score_distributions(results)

        assert distributions["answerable"].count == 0
        assert distributions["unanswerable"].count == 0
