"""Relevance threshold scanning and deterministic selection."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .metrics import decision_metrics


class ThresholdError(ValueError):
    """Raised when threshold scan parameters are invalid."""


@dataclass(frozen=True)
class ThresholdRow:
    """Describe decision metrics and weighted cost at one threshold."""

    threshold: float
    metrics: dict[str, Any]
    weighted_cost: float
    false_answer_count: int
    false_refusal_count: int


def validate_scan_parameters(
    start: float,
    end: float,
    step: float,
) -> None:
    """Validate threshold scan bounds without mutating state."""
    for value in (start, end, step):
        from decimal import Decimal as _Decimal

        try:
            parsed = _Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ThresholdError("阈值参数必须是数字") from exc
        if not parsed.is_finite():
            raise ThresholdError("阈值参数必须是有限数字")
    if not 0 <= start <= 1:
        raise ThresholdError("threshold_start 必须位于 [0, 1]")
    if not 0 <= end <= 1:
        raise ThresholdError("threshold_end 必须位于 [0, 1]")
    if start > end:
        raise ThresholdError("threshold_start 不能大于 threshold_end")
    if step <= 0:
        raise ThresholdError("threshold_step 必须大于 0")


def generate_thresholds(start: float, end: float, step: float) -> list[float]:
    """Generate an inclusive threshold list without float accumulation drift."""
    validate_scan_parameters(start, end, step)
    start_decimal = Decimal(str(start))
    end_decimal = Decimal(str(end))
    step_decimal = Decimal(str(step))

    thresholds: list[float] = []
    current = start_decimal
    while current <= end_decimal:
        thresholds.append(float(current))
        current += step_decimal
    return thresholds


def scan_decision_metrics(
    results: list[Any],
    thresholds: list[float],
    false_answer_weight: float,
    false_refusal_weight: float,
) -> list[ThresholdRow]:
    """Compute decision metrics for every threshold without re-retrieval."""
    rows: list[ThresholdRow] = []
    for threshold in thresholds:
        metrics = decision_metrics(results, threshold)
        false_answer_count = int(metrics["fp"])
        false_refusal_count = int(metrics["fn"])
        weighted_cost = (
            false_answer_count * false_answer_weight
            + false_refusal_count * false_refusal_weight
        )
        rows.append(
            ThresholdRow(
                threshold=threshold,
                metrics=metrics,
                weighted_cost=weighted_cost,
                false_answer_count=false_answer_count,
                false_refusal_count=false_refusal_count,
            )
        )
    return rows


def select_recommended_threshold(rows: list[ThresholdRow]) -> ThresholdRow:
    """Select one threshold deterministically from calibration rows."""
    if not rows:
        raise ThresholdError("阈值扫描结果为空，无法选择推荐阈值")

    best = rows[0]
    for candidate in rows[1:]:
        if _is_better(candidate, best):
            best = candidate
    return best


def _is_better(candidate: ThresholdRow, best: ThresholdRow) -> bool:
    if candidate.weighted_cost != best.weighted_cost:
        return candidate.weighted_cost < best.weighted_cost
    if candidate.false_answer_count != best.false_answer_count:
        return candidate.false_answer_count < best.false_answer_count
    if candidate.false_refusal_count != best.false_refusal_count:
        return candidate.false_refusal_count < best.false_refusal_count
    candidate_accuracy = candidate.metrics["decision_accuracy"]
    best_accuracy = best.metrics["decision_accuracy"]
    if candidate_accuracy != best_accuracy:
        if candidate_accuracy is None or best_accuracy is None:
            return candidate_accuracy is not None
        return candidate_accuracy > best_accuracy
    return candidate.threshold > best.threshold
