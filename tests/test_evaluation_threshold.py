"""Offline tests for threshold scanning and selection."""

import pytest

from src.evaluation.metrics import decision_metrics
from src.evaluation.threshold import (
    ThresholdError,
    generate_thresholds,
    scan_decision_metrics,
    select_recommended_threshold,
    validate_scan_parameters,
)

from evaluation_helpers import make_result


def _results(score_pairs: list[tuple[bool, float]]) -> list[object]:
    results = []
    for index, (answerable, score) in enumerate(score_pairs):
        results.append(
            make_result(
                f"c{index}",
                answerable=answerable,
                max_score=score,
                matched_at_1=1 if answerable else 0,
                matched_at_3=1 if answerable else 0,
                matched_at_5=1 if answerable else 0,
                total_evidence=1 if answerable else 0,
                hit_at_1=answerable or None,
                hit_at_3=answerable or None,
                hit_at_5=answerable or None,
                reciprocal_rank=1.0 if answerable else None,
            )
        )
    return results


def test_scan_covers_start_and_end() -> None:
    thresholds = generate_thresholds(0.20, 0.22, 0.01)
    assert thresholds[0] == 0.20
    assert thresholds[-1] == 0.22
    assert len(thresholds) == 3


def test_scan_has_no_float_drift() -> None:
    thresholds = generate_thresholds(0.20, 0.60, 0.01)
    assert 0.35 in thresholds
    assert 0.46 in thresholds
    assert len(thresholds) == 41
    assert thresholds[0] == 0.20
    assert thresholds[-1] == 0.60
    assert all(0.20 <= value <= 0.60 for value in thresholds)


def test_step_does_not_accumulate_drift() -> None:
    thresholds = generate_thresholds(0.20, 0.60, 0.01)
    differences = [
        round((second - first) * 10000) for first, second in zip(thresholds, thresholds[1:])
    ]
    assert set(differences) == {100}


def test_start_above_end_is_rejected() -> None:
    with pytest.raises(ThresholdError, match="start"):
        generate_thresholds(0.60, 0.20, 0.01)


def test_non_positive_step_is_rejected() -> None:
    with pytest.raises(ThresholdError, match="step"):
        generate_thresholds(0.20, 0.60, 0.0)
    with pytest.raises(ThresholdError, match="step"):
        generate_thresholds(0.20, 0.60, -0.01)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_out_of_range_or_non_finite_bounds_are_rejected(value: float) -> None:
    with pytest.raises(ThresholdError):
        generate_thresholds(value, 0.60, 0.01)


def test_weighted_cost_is_computed() -> None:
    results = _results(
        [
            (True, 0.9),
            (True, 0.9),
            (False, 0.9),
            (False, 0.1),
        ]
    )
    rows = scan_decision_metrics(results, [0.35], 3.0, 1.0)
    assert rows[0].weighted_cost == 3.0
    assert rows[0].false_answer_count == 1
    assert rows[0].false_refusal_count == 0


def test_selection_prefers_lower_weighted_cost() -> None:
    results = _results(
        [
            (True, 0.9),
            (False, 0.8),
            (False, 0.1),
        ]
    )
    rows = scan_decision_metrics(results, [0.35, 0.85], 3.0, 1.0)
    selected = select_recommended_threshold(rows)
    assert selected.threshold == 0.85


def test_selection_first_tiebreak_prefers_fewer_false_answers() -> None:
    rows = [
        make_threshold_row(0.30, weighted_cost=2.0, fp=2, fn=2),
        make_threshold_row(0.50, weighted_cost=2.0, fp=1, fn=5),
    ]
    selected = select_recommended_threshold(rows)
    assert selected.threshold == 0.50


def test_selection_second_tiebreak_prefers_fewer_false_refusals() -> None:
    rows = [
        make_threshold_row(0.30, weighted_cost=2.0, fp=2, fn=2),
        make_threshold_row(0.50, weighted_cost=2.0, fp=2, fn=1),
    ]
    selected = select_recommended_threshold(rows)
    assert selected.threshold == 0.50


def test_selection_third_tiebreak_prefers_higher_accuracy() -> None:
    rows = [
        make_threshold_row(0.30, weighted_cost=2.0, fp=2, fn=2, accuracy=0.5),
        make_threshold_row(0.50, weighted_cost=2.0, fp=2, fn=2, accuracy=0.7),
    ]
    selected = select_recommended_threshold(rows)
    assert selected.threshold == 0.50


def test_selection_final_tiebreak_prefers_higher_threshold() -> None:
    rows = [
        make_threshold_row(0.30, weighted_cost=2.0, fp=2, fn=2, accuracy=0.5),
        make_threshold_row(0.50, weighted_cost=2.0, fp=2, fn=2, accuracy=0.5),
    ]
    selected = select_recommended_threshold(rows)
    assert selected.threshold == 0.50


def test_selection_from_empty_rows_is_rejected() -> None:
    with pytest.raises(ThresholdError, match="空"):
        select_recommended_threshold([])


def test_current_threshold_comparison_uses_same_decision_rule() -> None:
    results = _results([(True, 0.35), (True, 0.34), (False, 0.9)])
    current = decision_metrics(results, 0.35)
    assert current["tp"] == 1
    assert current["fn"] == 1
    assert current["fp"] == 1


def test_calibration_selection_does_not_see_test_rows() -> None:
    calibration = _results([(False, 0.9), (True, 0.9), (True, 0.1)])
    rows = scan_decision_metrics(calibration, [0.35, 0.85], 3.0, 1.0)
    recommended = select_recommended_threshold(rows)
    assert recommended.threshold == 0.85


def make_threshold_row(
    threshold: float,
    *,
    weighted_cost: float,
    fp: int,
    fn: int,
    accuracy: float | None = None,
) -> object:
    from src.evaluation.threshold import ThresholdRow

    metrics = {
        "tp": 0,
        "fp": fp,
        "tn": 0,
        "fn": fn,
        "decision_accuracy": accuracy,
        "answerable_precision": None,
        "answerable_recall": None,
        "false_answer_rate": None,
        "false_refusal_rate": None,
        "answer_rate": None,
        "refusal_rate": None,
    }
    return ThresholdRow(
        threshold=threshold,
        metrics=metrics,
        weighted_cost=weighted_cost,
        false_answer_count=fp,
        false_refusal_count=fn,
    )


def test_validate_scan_parameters_accepts_boundary_values() -> None:
    validate_scan_parameters(0.0, 1.0, 0.01)
