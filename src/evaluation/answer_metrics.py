"""Compute deterministic answer-quality metrics.

The framework performs **annotated fact coverage** (lexical, annotation
based), **supported fact citation coverage**, **citation validity**, **known
contradiction detection**, and a **strict annotated pass rate**.  It is not a
semantic-correctness proof, does not detect every hallucination, and never
uses BLEU, ROUGE, string-similarity, embeddings, or an LLM judge.

The text normalization is deliberately transparent and conservative: it only
folds Unicode, case, whitespace, and common full-width punctuation.  It
never performs fuzzy matching, edit-distance thresholds, synonym expansion,
or tokenization.
"""

import math
import re
import statistics
import unicodedata
from collections.abc import Iterable

from src.evaluation.answer_models import (
    AggregateAnswerMetrics,
    AnswerAnnotation,
    AnswerResponse,
    AnswerSource,
    CaseAnswerMetrics,
    CitationOccurrence,
    FactScoring,
    LatencySummary,
    RequiredFact,
)
from src.evaluation.answer_citations import CitationParseResult
from src.evaluation.matching import normalize_text, text_matches_expectation
from src.evaluation.models import EvaluationCase, EvidenceExpectation
from src.exceptions import AnswerEvaluationError
from src.rag_service import INSUFFICIENT_CONTEXT_ANSWER

_WHITESPACE = re.compile(r"\s+")
_CJK = "".join(
    chr(code_point)
    for code_point in range(0x4E00, 0x9FFF + 1)
)
_SPACE_BEFORE_CJK = re.compile(rf"\s+([{_CJK}])")
_SPACE_AFTER_CJK = re.compile(rf"([{_CJK}])\s+")
_FULLWIDTH_PUNCT = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


def normalize_answer_text(value: str) -> str:
    """Normalize an answer for conservative lexical fact matching.

    Applied transformations:
      - strip
      - Unicode NFKC
      - casefold
      - collapse consecutive whitespace
      - remove whitespace adjacent to CJK characters so ordinary spacing
        cannot break Chinese phrase matching
      - unify common full-width punctuation to ASCII

    The function never removes CJK characters and never performs aggressive
    tokenization, so Chinese phrases stay intact.
    """
    if not isinstance(value, str):
        raise TypeError("待标准化的内容必须是字符串")
    folded = unicodedata.normalize("NFKC", value).casefold()
    folded = folded.translate(_FULLWIDTH_PUNCT)
    collapsed = _WHITESPACE.sub(" ", folded).strip()
    collapsed = _SPACE_BEFORE_CJK.sub(r"\1", collapsed)
    collapsed = _SPACE_AFTER_CJK.sub(r"\1", collapsed)
    return collapsed.strip()


def phrase_appears(phrase: str, answer: str) -> bool:
    """Return True when a normalized phrase is a substring of an answer."""
    normalized_phrase = normalize_answer_text(phrase)
    if not normalized_phrase:
        return False
    return normalized_phrase in normalize_answer_text(answer)


def fact_covered(fact: RequiredFact, answer: str) -> tuple[bool, str | None]:
    """Return whether any accepted phrase appears in the normalized answer.

    This is lexical annotated fact coverage, not complete semantic
    correctness: a fact is covered if and only if one of its accepted
    phrases, after identical normalization, is a substring of the final
    answer.
    """
    for phrase in fact.accepted_phrases:
        if phrase_appears(phrase, answer):
            return True, phrase
    return False, None


def forbidden_phrase_hits(
    forbidden_phrases: tuple[str, ...],
    answer: str,
) -> tuple[str, ...]:
    """Return every forbidden phrase that appears in the normalized answer."""
    normalized_answer = normalize_answer_text(answer)
    return tuple(
        phrase
        for phrase in forbidden_phrases
        if phrase_appears(phrase, normalized_answer)
    )


def fact_grounded_by_citation(
    fact: RequiredFact,
    case: EvaluationCase,
    response: AnswerResponse,
    valid_occurrences: tuple[CitationOccurrence, ...],
    *,
    covered: bool,
) -> tuple[bool, tuple[int, ...]]:
    """Check whether cited sources support one annotated fact.

    Only annotated required facts are evaluated.  A fact is grounded when at
    least one legal cited source satisfies at least one of its supporting
    evidence expectations (document id, page number, and required terms all
    matching).  V1 uses the answer-level citation set; sentence-level
    claim-to-citation alignment is explicitly out of scope.
    """
    if not covered:
        return False, ()

    supporting_indexes = fact.supporting_evidence_indexes
    supporting_expectations = [
        case.expected_evidence[index - 1] for index in supporting_indexes
    ]
    legal_source_ranks = {
        occurrence.source_number
        for occurrence in valid_occurrences
        if occurrence.source_number <= len(response.sources)
    }

    matching_citations: set[int] = set()
    for rank in sorted(legal_source_ranks):
        source = response.sources[rank - 1]
        if _source_supports_any_expectation(source, supporting_expectations):
            matching_citations.add(rank)

    return bool(matching_citations), tuple(sorted(matching_citations))


def _source_supports_any_expectation(
    source: AnswerSource,
    expectations: list[EvidenceExpectation],
) -> bool:
    for expectation in expectations:
        if (
            expectation.document_id == source.document_id
            and (
                expectation.page_number is None
                or expectation.page_number == source.page_number
            )
            and text_matches_expectation(expectation, source.text)
        ):
            return True
    return False


def compute_case_metrics(
    case: EvaluationCase,
    annotation: AnswerAnnotation,
    response: AnswerResponse,
    citation_result: CitationParseResult,
) -> CaseAnswerMetrics:
    """Compute every per-case metric from already-validated inputs."""
    occurrences = citation_result.valid_syntax
    valid_occurrence_count = len(occurrences)
    citation_occurrence_count = valid_occurrence_count
    malformed_count = len(citation_result.malformed_fragments)

    legal_ranks = set(range(1, len(response.sources) + 1))
    invalid_numbers = sorted(
        {
            occurrence.source_number
            for occurrence in occurrences
            if occurrence.source_number not in legal_ranks
        }
    )
    invalid_occurrence_count = sum(
        1
        for occurrence in occurrences
        if occurrence.source_number not in legal_ranks
    )
    valid_citation_occurrence_count = valid_occurrence_count - invalid_occurrence_count

    if citation_occurrence_count > 0:
        citation_validity_rate = valid_citation_occurrence_count / citation_occurrence_count
    else:
        citation_validity_rate = None

    unique_cited_source_count = len(
        {
            occurrence.source_number
            for occurrence in occurrences
            if occurrence.source_number in legal_ranks
        }
    )

    forbidden_hits = forbidden_phrase_hits(annotation.forbidden_phrases, response.answer)
    contradiction_free = len(forbidden_hits) == 0

    fact_scorings: list[FactScoring] = []
    covered_count = 0
    grounded_count = 0
    for fact in annotation.required_facts:
        covered, matched_phrase = fact_covered(fact, response.answer)
        if covered:
            covered_count += 1
        grounded, supporting_citations = fact_grounded_by_citation(
            fact,
            case,
            response,
            occurrences,
            covered=covered,
        )
        if grounded:
            grounded_count += 1
        fact_scorings.append(
            FactScoring(
                fact_id=fact.fact_id,
                covered=covered,
                matched_phrase=matched_phrase,
                supporting_evidence_indexes=fact.supporting_evidence_indexes,
                supporting_citation_numbers=supporting_citations,
                grounded_by_citation=grounded,
            )
        )

    required_fact_count = len(annotation.required_facts)
    if case.answerable:
        if required_fact_count == 0:
            raise AnswerEvaluationError(
                f"case {case.id} 是 answerable 但没有 required_facts"
            )
        fact_coverage = covered_count / required_fact_count
        grounded_fact_coverage = grounded_count / required_fact_count
    else:
        fact_coverage = None
        grounded_fact_coverage = None

    decision_correct = _decision_correct(case.answerable, response.answer_status)

    strict_pass = _strict_pass(
        case=case,
        response=response,
        covered_count=covered_count,
        required_fact_count=required_fact_count,
        grounded_count=grounded_count,
        invalid_occurrence_count=invalid_occurrence_count,
        malformed_count=malformed_count,
        forbidden_hits=forbidden_hits,
        valid_occurrence_count=valid_citation_occurrence_count,
    )

    return CaseAnswerMetrics(
        case_id=case.id,
        split=case.split,
        category=case.category,
        difficulty=case.difficulty,
        answerable=case.answerable,
        answer_status=response.answer_status,
        decision_correct=decision_correct,
        question=case.question,
        reference_answer=annotation.reference_answer,
        answer=response.answer,
        required_fact_count=required_fact_count,
        covered_fact_count=covered_count,
        grounded_fact_count=grounded_count,
        fact_coverage=fact_coverage,
        grounded_fact_coverage=grounded_fact_coverage,
        complete_fact_coverage=(
            covered_count == required_fact_count if case.answerable else False
        ),
        complete_grounded_fact_coverage=(
            grounded_count == required_fact_count if case.answerable else False
        ),
        forbidden_phrase_hits=forbidden_hits,
        contradiction_free=contradiction_free,
        citation_occurrence_count=citation_occurrence_count,
        unique_cited_source_count=unique_cited_source_count,
        valid_citation_occurrence_count=valid_citation_occurrence_count,
        invalid_citation_occurrence_count=invalid_occurrence_count,
        invalid_citation_numbers=tuple(invalid_numbers),
        malformed_citation_count=malformed_count,
        malformed_citation_fragments=citation_result.malformed_fragments,
        citation_validity_rate=citation_validity_rate,
        facts=tuple(fact_scorings),
        max_relevance_score=response.max_relevance_score,
        relevance_threshold=response.relevance_threshold,
        retrieval_elapsed_ms=response.retrieval_elapsed_ms,
        generation_elapsed_ms=response.generation_elapsed_ms,
        total_elapsed_ms=response.total_elapsed_ms,
        reranker_applied=response.reranker_applied,
        reranker_fallback=response.reranker_fallback,
        strict_pass=strict_pass,
    )


def _decision_correct(answerable: bool, answer_status: str) -> bool:
    if answerable:
        return answer_status == "answered"
    return answer_status == "insufficient_context"


def _strict_pass(
    *,
    case: EvaluationCase,
    response: AnswerResponse,
    covered_count: int,
    required_fact_count: int,
    grounded_count: int,
    invalid_occurrence_count: int,
    malformed_count: int,
    forbidden_hits: tuple[str, ...],
    valid_occurrence_count: int,
) -> bool:
    if case.answerable:
        return (
            response.answer_status == "answered"
            and required_fact_count > 0
            and covered_count == required_fact_count
            and grounded_count == required_fact_count
            and invalid_occurrence_count == 0
            and malformed_count == 0
            and not forbidden_hits
            and valid_occurrence_count > 0
        )

    return (
        response.answer_status == "insufficient_context"
        and response.answer == INSUFFICIENT_CONTEXT_ANSWER
        and valid_occurrence_count == 0
        and malformed_count == 0
    )


def _finite_series(series: Iterable[float]) -> list[float]:
    return [
        float(value)
        for value in series
        if math.isfinite(value) and value >= 0.0
    ]


def summarize_latency(series: Iterable[float]) -> LatencySummary:
    """Summarize one latency series with deterministic order and no NaN.

    Small-dataset p95 uses linear interpolation over the sorted values, which
    keeps the definition stable and testable.
    """
    values = sorted(_finite_series(series))
    if not values:
        return LatencySummary(
            count=0,
            mean=None,
            median=None,
            p95=None,
            minimum=None,
            maximum=None,
        )
    return LatencySummary(
        count=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        p95=_percentile(values, 0.95),
        minimum=values[0],
        maximum=values[-1],
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * weight


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_aggregate_metrics(
    case_metrics: tuple[CaseAnswerMetrics, ...],
) -> AggregateAnswerMetrics:
    """Aggregate per-case metrics over one group.

    Average fact coverage uses only answerable cases as the denominator, and
    false refusals count as zero coverage instead of being excluded.
    Citation validity is occurrence-weighted globally, not a simple mean of
    per-case rates.
    """
    total = len(case_metrics)
    answerable_count = sum(1 for case in case_metrics if case.answerable)
    unanswerable_count = total - answerable_count
    answered_count = sum(1 for case in case_metrics if case.answer_status == "answered")
    refused_count = sum(
        1
        for case in case_metrics
        if case.answer_status == "insufficient_context"
    )
    true_answer_count = sum(
        1
        for case in case_metrics
        if case.answerable and case.answer_status == "answered"
    )
    false_refusal_count = sum(
        1
        for case in case_metrics
        if case.answerable and case.answer_status == "insufficient_context"
    )
    true_refusal_count = sum(
        1
        for case in case_metrics
        if not case.answerable and case.answer_status == "insufficient_context"
    )
    false_answer_count = sum(
        1
        for case in case_metrics
        if not case.answerable and case.answer_status == "answered"
    )

    decision_correct_count = sum(1 for case in case_metrics if case.decision_correct)

    # Average fact coverage: answerable denominator; false refusals count 0.
    answerable_metrics = [case for case in case_metrics if case.answerable]
    if answerable_count:
        average_fact_coverage = sum(
            (case.fact_coverage or 0.0) for case in answerable_metrics
        ) / answerable_count
        average_grounded_fact_coverage = sum(
            (case.grounded_fact_coverage or 0.0) for case in answerable_metrics
        ) / answerable_count
        complete_fact_coverage_rate = _safe_ratio(
            sum(1 for case in answerable_metrics if case.complete_fact_coverage),
            answerable_count,
        )
        complete_grounded_fact_coverage_rate = _safe_ratio(
            sum(
                1
                for case in answerable_metrics
                if case.complete_grounded_fact_coverage
            ),
            answerable_count,
        )
    else:
        average_fact_coverage = None
        average_grounded_fact_coverage = None
        complete_fact_coverage_rate = None
        complete_grounded_fact_coverage_rate = None

    contradiction_free_count = sum(1 for case in case_metrics if case.contradiction_free)
    contradiction_free_rate = _safe_ratio(contradiction_free_count, total)

    citation_occurrence_count = sum(
        case.citation_occurrence_count for case in case_metrics
    )
    valid_citation_occurrence_count = sum(
        case.valid_citation_occurrence_count for case in case_metrics
    )
    invalid_citation_occurrence_count = sum(
        case.invalid_citation_occurrence_count for case in case_metrics
    )
    malformed_citation_count = sum(
        case.malformed_citation_count for case in case_metrics
    )
    citation_validity_rate = _safe_ratio(
        valid_citation_occurrence_count,
        citation_occurrence_count,
    )

    invalid_citation_case_count = sum(
        1 for case in case_metrics if case.invalid_citation_occurrence_count > 0
    )
    malformed_citation_case_count = sum(
        1 for case in case_metrics if case.malformed_citation_count > 0
    )
    invalid_citation_case_rate = _safe_ratio(invalid_citation_case_count, total)
    malformed_citation_case_rate = _safe_ratio(malformed_citation_case_count, total)

    strict_pass_count = sum(1 for case in case_metrics if case.strict_pass)
    strict_pass_rate = _safe_ratio(strict_pass_count, total)

    return AggregateAnswerMetrics(
        case_count=total,
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        answered_count=answered_count,
        refused_count=refused_count,
        true_answer_count=true_answer_count,
        false_refusal_count=false_refusal_count,
        true_refusal_count=true_refusal_count,
        false_answer_count=false_answer_count,
        decision_accuracy=_safe_ratio(decision_correct_count, total),
        false_answer_rate=_safe_ratio(false_answer_count, unanswerable_count),
        false_refusal_rate=_safe_ratio(false_refusal_count, answerable_count),
        average_fact_coverage=average_fact_coverage,
        average_grounded_fact_coverage=average_grounded_fact_coverage,
        complete_fact_coverage_rate=complete_fact_coverage_rate,
        complete_grounded_fact_coverage_rate=complete_grounded_fact_coverage_rate,
        contradiction_free_rate=contradiction_free_rate,
        citation_occurrence_count=citation_occurrence_count,
        valid_citation_occurrence_count=valid_citation_occurrence_count,
        invalid_citation_occurrence_count=invalid_citation_occurrence_count,
        malformed_citation_count=malformed_citation_count,
        citation_validity_rate=citation_validity_rate,
        invalid_citation_case_rate=invalid_citation_case_rate,
        malformed_citation_case_rate=malformed_citation_case_rate,
        strict_pass_count=strict_pass_count,
        strict_pass_rate=strict_pass_rate,
    )


def group_case_metrics(
    case_metrics: tuple[CaseAnswerMetrics, ...],
    attribute: str,
) -> dict[str, AggregateAnswerMetrics]:
    """Aggregate per-case metrics grouped by one case attribute."""
    grouped: dict[str, list[CaseAnswerMetrics]] = {}
    for case in case_metrics:
        key = str(getattr(case, attribute))
        grouped.setdefault(key, []).append(case)
    return {
        key: compute_aggregate_metrics(tuple(values))
        for key, values in sorted(grouped.items())
    }
