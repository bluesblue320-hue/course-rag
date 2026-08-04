"""Sweep relevance thresholds and pick a recommended value deterministically."""

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from src.evaluation.metrics import DecisionMetrics, compute_decision_metrics
from src.evaluation.models import EvaluationResult
from src.exceptions import ThresholdConfigurationError

DEFAULT_THRESHOLD_START = 0.20
DEFAULT_THRESHOLD_END = 0.60
DEFAULT_THRESHOLD_STEP = 0.01
DEFAULT_FALSE_ANSWER_WEIGHT = 3.0
DEFAULT_FALSE_REFUSAL_WEIGHT = 1.0


@dataclass(frozen=True)
class ThresholdCandidate:
    """Describe one swept threshold with its decision metrics and cost."""

    threshold: float
    metrics: DecisionMetrics
    weighted_cost: float


def _to_decimal(value: float, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThresholdConfigurationError(f"{field} 必须是数字")
    if not math.isfinite(float(value)):
        raise ThresholdConfigurationError(f"{field} 必须是有限数字")
    try:
        return Decimal(str(float(value)))
    except InvalidOperation as exc:  # pragma: no cover - guarded by isfinite
        raise ThresholdConfigurationError(f"{field} 无法解析为十进制数") from exc


def validate_weight(value: float, field: str) -> float:
    """Reject weights that are not finite non-negative numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThresholdConfigurationError(f"{field} 必须是数字")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0:
        raise ThresholdConfigurationError(f"{field} 必须是非负有限数字")
    return weight


def generate_thresholds(
    start: float,
    end: float,
    step: float,
) -> tuple[float, ...]:
    """Return an inclusive threshold grid built with Decimal arithmetic.

    Decimal accumulation keeps values such as 0.35 exact instead of producing
    artefacts like 0.35000000000000004. Both endpoints are always present.
    """
    start_decimal = _to_decimal(start, "threshold-start")
    end_decimal = _to_decimal(end, "threshold-end")
    step_decimal = _to_decimal(step, "threshold-step")

    if step_decimal <= 0:
        raise ThresholdConfigurationError("threshold-step 必须大于 0")
    if start_decimal > end_decimal:
        raise ThresholdConfigurationError(
            "threshold-start 不能大于 threshold-end"
        )
    for name, value in (
        ("threshold-start", start_decimal),
        ("threshold-end", end_decimal),
    ):
        if value < 0 or value > 1:
            raise ThresholdConfigurationError(f"{name} 必须位于 [0, 1] 区间")

    values: list[Decimal] = []
    current = start_decimal
    while current < end_decimal:
        values.append(current)
        current += step_decimal
    values.append(end_decimal)
    return tuple(float(value) for value in values)


def weighted_cost(
    metrics: DecisionMetrics,
    false_answer_weight: float,
    false_refusal_weight: float,
) -> float:
    """Return FP * false_answer_weight + FN * false_refusal_weight."""
    return (
        metrics.false_positive * false_answer_weight
        + metrics.false_negative * false_refusal_weight
    )


def scan_thresholds(
    results: tuple[EvaluationResult, ...],
    thresholds: tuple[float, ...],
    false_answer_weight: float = DEFAULT_FALSE_ANSWER_WEIGHT,
    false_refusal_weight: float = DEFAULT_FALSE_REFUSAL_WEIGHT,
) -> tuple[ThresholdCandidate, ...]:
    """Score every threshold using already recorded max scores.

    The sweep never re-runs retrieval and never calls an embedding model; it
    only re-applies the production decision boundary to stored scores.
    """
    if not results:
        raise ThresholdConfigurationError("阈值扫描需要至少一个评估结果")
    if not thresholds:
        raise ThresholdConfigurationError("阈值列表不能为空")

    answer_weight = validate_weight(false_answer_weight, "false-answer-weight")
    refusal_weight = validate_weight(
        false_refusal_weight,
        "false-refusal-weight",
    )

    candidates: list[ThresholdCandidate] = []
    for threshold in thresholds:
        metrics = compute_decision_metrics(results, threshold)
        candidates.append(
            ThresholdCandidate(
                threshold=threshold,
                metrics=metrics,
                weighted_cost=weighted_cost(
                    metrics,
                    answer_weight,
                    refusal_weight,
                ),
            )
        )
    return tuple(candidates)


def _selection_key(candidate: ThresholdCandidate) -> tuple[float, ...]:
    """Return the deterministic ordering key, smallest tuple wins.

    Order: lowest weighted cost, then fewer false answers, then fewer false
    refusals, then higher accuracy, then the higher (more conservative)
    threshold.
    """
    return (
        candidate.weighted_cost,
        float(candidate.metrics.false_positive),
        float(candidate.metrics.false_negative),
        -candidate.metrics.decision_accuracy,
        -candidate.threshold,
    )


def select_recommended_threshold(
    candidates: tuple[ThresholdCandidate, ...],
) -> ThresholdCandidate:
    """Pick one threshold from calibration candidates using fixed tie-breakers.

    The caller must pass calibration candidates only. Selecting on the test
    split would leak the held-out data into threshold tuning.
    """
    if not candidates:
        raise ThresholdConfigurationError("推荐阈值需要至少一个候选")
    return min(candidates, key=_selection_key)


def find_candidate(
    candidates: tuple[ThresholdCandidate, ...],
    threshold: float,
) -> ThresholdCandidate | None:
    """Return the swept candidate matching one threshold, if it was swept."""
    for candidate in candidates:
        if math.isclose(candidate.threshold, threshold, rel_tol=0, abs_tol=1e-12):
            return candidate
    return None
