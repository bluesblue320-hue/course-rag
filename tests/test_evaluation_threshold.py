"""Tests for the threshold grid, weighted cost, sweep, and selection rules."""

import pytest

from src.evaluation.metrics import DecisionMetrics
from src.evaluation.threshold import (
    ThresholdCandidate,
    find_candidate,
    generate_thresholds,
    scan_thresholds,
    select_recommended_threshold,
    validate_weight,
    weighted_cost,
)
from src.exceptions import ThresholdConfigurationError
from tests.evaluation_helpers import make_result


def make_metrics(
    threshold: float,
    false_positive: int,
    false_negative: int,
    decision_accuracy: float = 0.5,
) -> DecisionMetrics:
    """Build decision metrics directly so selection rules can be isolated."""
    return DecisionMetrics(
        threshold=threshold,
        case_count=10,
        true_positive=5 - false_negative,
        false_negative=false_negative,
        true_negative=5 - false_positive,
        false_positive=false_positive,
        decision_accuracy=decision_accuracy,
        answerable_precision=None,
        answerable_recall=None,
        false_answer_rate=None,
        false_refusal_rate=None,
        answer_rate=0.5,
        refusal_rate=0.5,
    )


def make_candidate(
    threshold: float,
    false_positive: int,
    false_negative: int,
    cost: float,
    decision_accuracy: float = 0.5,
) -> ThresholdCandidate:
    """Build one candidate with an explicit cost for tie-breaker tests."""
    return ThresholdCandidate(
        threshold=threshold,
        metrics=make_metrics(
            threshold,
            false_positive,
            false_negative,
            decision_accuracy,
        ),
        weighted_cost=cost,
    )


class TestGenerateThresholds:
    def test_it_includes_both_endpoints(self) -> None:
        thresholds = generate_thresholds(0.2, 0.6, 0.1)

        assert thresholds[0] == pytest.approx(0.2)
        assert thresholds[-1] == pytest.approx(0.6)

    def test_it_avoids_binary_float_drift(self) -> None:
        thresholds = generate_thresholds(0.0, 1.0, 0.01)

        # A naive float accumulation produces 0.35000000000000003 here.
        assert 0.35 in thresholds
        assert 0.07 in thresholds
        assert len(thresholds) == 101

    def test_it_appends_the_endpoint_when_the_step_does_not_divide_evenly(
        self,
    ) -> None:
        thresholds = generate_thresholds(0.2, 0.55, 0.1)

        assert thresholds == (0.2, 0.3, 0.4, 0.5, 0.55)

    def test_it_returns_a_single_value_when_start_equals_end(self) -> None:
        assert generate_thresholds(0.35, 0.35, 0.01) == (0.35,)

    def test_the_grid_is_strictly_increasing(self) -> None:
        thresholds = generate_thresholds(0.2, 0.6, 0.01)

        assert list(thresholds) == sorted(thresholds)
        assert len(set(thresholds)) == len(thresholds)

    @pytest.mark.parametrize(
        ("start", "end", "step"),
        [
            (0.2, 0.6, 0.0),
            (0.2, 0.6, -0.01),
            (0.6, 0.2, 0.01),
            (-0.1, 0.6, 0.01),
            (0.2, 1.5, 0.01),
        ],
    )
    def test_it_rejects_an_invalid_grid(
        self,
        start: float,
        end: float,
        step: float,
    ) -> None:
        with pytest.raises(ThresholdConfigurationError):
            generate_thresholds(start, end, step)

    @pytest.mark.parametrize("bad", ["0.2", None, True, float("nan")])
    def test_it_rejects_a_non_numeric_bound(self, bad: object) -> None:
        with pytest.raises(ThresholdConfigurationError):
            generate_thresholds(bad, 0.6, 0.01)  # type: ignore[arg-type]


class TestValidateWeight:
    @pytest.mark.parametrize("value", [0, 0.0, 1, 3.0, 12.5])
    def test_it_accepts_non_negative_finite_numbers(self, value: float) -> None:
        assert validate_weight(value, "w") == pytest.approx(float(value))

    @pytest.mark.parametrize(
        "value",
        [-0.1, float("inf"), float("nan"), True, "3", None],
    )
    def test_it_rejects_everything_else(self, value: object) -> None:
        with pytest.raises(ThresholdConfigurationError):
            validate_weight(value, "w")  # type: ignore[arg-type]


class TestWeightedCost:
    def test_it_weights_false_answers_and_false_refusals(self) -> None:
        metrics = make_metrics(0.35, false_positive=2, false_negative=3)

        assert weighted_cost(metrics, 3.0, 1.0) == pytest.approx(2 * 3.0 + 3 * 1.0)

    def test_zero_weights_produce_zero_cost(self) -> None:
        metrics = make_metrics(0.35, false_positive=4, false_negative=4)

        assert weighted_cost(metrics, 0.0, 0.0) == pytest.approx(0.0)

    def test_a_perfect_split_costs_nothing(self) -> None:
        metrics = make_metrics(0.35, false_positive=0, false_negative=0)

        assert weighted_cost(metrics, 3.0, 1.0) == pytest.approx(0.0)


class TestScanThresholds:
    def _results(self) -> tuple:
        return (
            make_result(case_id="a", answerable=True, max_relevance_score=0.7),
            make_result(case_id="b", answerable=True, max_relevance_score=0.4),
            make_result(case_id="c", answerable=False, max_relevance_score=0.3),
            make_result(case_id="d", answerable=False, max_relevance_score=0.6),
        )

    def test_it_returns_one_candidate_per_threshold(self) -> None:
        thresholds = generate_thresholds(0.3, 0.7, 0.1)

        candidates = scan_thresholds(self._results(), thresholds)

        assert len(candidates) == len(thresholds)
        assert [candidate.threshold for candidate in candidates] == list(thresholds)

    def test_it_recomputes_the_matrix_at_each_threshold(self) -> None:
        candidates = scan_thresholds(self._results(), (0.35, 0.65))

        low, high = candidates
        assert (low.metrics.true_positive, low.metrics.false_positive) == (2, 1)
        assert (low.metrics.false_negative, low.metrics.true_negative) == (0, 1)
        assert (high.metrics.true_positive, high.metrics.false_positive) == (1, 0)
        assert (high.metrics.false_negative, high.metrics.true_negative) == (1, 2)

    def test_the_cost_uses_the_supplied_weights(self) -> None:
        candidates = scan_thresholds(self._results(), (0.35,), 3.0, 1.0)

        # One false answer at 0.35, no false refusal.
        assert candidates[0].weighted_cost == pytest.approx(3.0)

    def test_it_never_touches_an_embedding_service(self) -> None:
        # scan_thresholds only reads recorded scores; results carry no service.
        results = tuple(
            make_result(case_id=str(index), max_relevance_score=0.5)
            for index in range(3)
        )

        candidates = scan_thresholds(results, (0.4, 0.6))

        assert [candidate.metrics.case_count for candidate in candidates] == [3, 3]

    def test_it_rejects_an_empty_result_set(self) -> None:
        with pytest.raises(ThresholdConfigurationError):
            scan_thresholds((), (0.35,))

    def test_it_rejects_an_empty_threshold_list(self) -> None:
        with pytest.raises(ThresholdConfigurationError):
            scan_thresholds(self._results(), ())

    def test_it_rejects_an_invalid_weight(self) -> None:
        with pytest.raises(ThresholdConfigurationError):
            scan_thresholds(self._results(), (0.35,), -1.0, 1.0)


class TestSelectRecommendedThreshold:
    def test_it_picks_the_lowest_weighted_cost(self) -> None:
        candidates = (
            make_candidate(0.30, false_positive=3, false_negative=0, cost=9.0),
            make_candidate(0.40, false_positive=1, false_negative=1, cost=4.0),
            make_candidate(0.50, false_positive=0, false_negative=5, cost=5.0),
        )

        assert select_recommended_threshold(candidates).threshold == 0.40

    def test_a_cost_tie_prefers_fewer_false_answers(self) -> None:
        candidates = (
            make_candidate(0.30, false_positive=2, false_negative=0, cost=6.0),
            make_candidate(0.40, false_positive=1, false_negative=3, cost=6.0),
        )

        assert select_recommended_threshold(candidates).threshold == 0.40

    def test_a_false_answer_tie_prefers_fewer_false_refusals(self) -> None:
        candidates = (
            make_candidate(0.30, false_positive=1, false_negative=3, cost=6.0),
            make_candidate(0.40, false_positive=1, false_negative=2, cost=6.0),
        )

        assert select_recommended_threshold(candidates).threshold == 0.40

    def test_a_full_matrix_tie_prefers_higher_accuracy(self) -> None:
        candidates = (
            make_candidate(
                0.30,
                false_positive=1,
                false_negative=1,
                cost=4.0,
                decision_accuracy=0.70,
            ),
            make_candidate(
                0.40,
                false_positive=1,
                false_negative=1,
                cost=4.0,
                decision_accuracy=0.90,
            ),
        )

        assert select_recommended_threshold(candidates).threshold == 0.40

    def test_a_complete_tie_prefers_the_more_conservative_threshold(self) -> None:
        candidates = (
            make_candidate(0.30, false_positive=1, false_negative=1, cost=4.0),
            make_candidate(0.55, false_positive=1, false_negative=1, cost=4.0),
            make_candidate(0.45, false_positive=1, false_negative=1, cost=4.0),
        )

        assert select_recommended_threshold(candidates).threshold == 0.55

    def test_the_selection_does_not_depend_on_input_order(self) -> None:
        first = make_candidate(0.30, false_positive=3, false_negative=0, cost=9.0)
        second = make_candidate(0.40, false_positive=1, false_negative=1, cost=4.0)
        third = make_candidate(0.50, false_positive=0, false_negative=5, cost=5.0)

        forward = select_recommended_threshold((first, second, third))
        backward = select_recommended_threshold((third, second, first))

        assert forward == backward

    def test_it_rejects_an_empty_candidate_list(self) -> None:
        with pytest.raises(ThresholdConfigurationError):
            select_recommended_threshold(())


class TestFindCandidate:
    def test_it_finds_a_swept_threshold(self) -> None:
        candidates = scan_thresholds(
            (make_result(max_relevance_score=0.5),),
            generate_thresholds(0.2, 0.6, 0.01),
        )

        found = find_candidate(candidates, 0.35)

        assert found is not None
        assert found.threshold == pytest.approx(0.35)

    def test_it_returns_none_when_the_threshold_was_not_swept(self) -> None:
        candidates = scan_thresholds(
            (make_result(max_relevance_score=0.5),),
            (0.30, 0.40),
        )

        assert find_candidate(candidates, 0.35) is None
