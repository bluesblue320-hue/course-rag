"""Immutable data models for the answer-quality evaluation framework.

Every model is a frozen dataclass.  None of these classes performs any
computation; scoring lives in :mod:`src.evaluation.answer_metrics` so the
data layer stays transparent and reusable.
"""

from dataclasses import dataclass
from typing import Any, Literal

AnswerStatus = Literal["answered", "insufficient_context"]
ANSWER_STATUSES: tuple[str, ...] = ("answered", "insufficient_context")


@dataclass(frozen=True)
class RequiredFact:
    """Describe one annotated fact the final answer must cover.

    ``supporting_evidence_indexes`` uses 1-based positions inside the owning
    case's ``expected_evidence`` tuple, matching the citation convention.
    """

    fact_id: str
    accepted_phrases: tuple[str, ...]
    supporting_evidence_indexes: tuple[int, ...]


@dataclass(frozen=True)
class AnswerAnnotation:
    """Describe one case's answer-quality ground truth.

    ``reference_answer`` is written for manual audit only.  It never takes
    part in automatic semantic-similarity scoring; the framework only checks
    that at least one ``accepted_phrase`` of each fact appears verbatim in
    the reference answer, as an annotation-integrity guard.
    """

    case_id: str
    reference_answer: str
    required_facts: tuple[RequiredFact, ...]
    forbidden_phrases: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class AnswerSource:
    """Describe one retrieved source as recorded in an answer responses file."""

    rank: int
    score: float
    text: str
    chunk_index: int
    document_id: str
    filename: str
    page_number: int | None


@dataclass(frozen=True)
class AnswerResponse:
    """Describe one case's final answer outcome from an offline responses file.

    ``answer_status`` is the structured decision field; the framework never
    guesses the status from the answer text.
    """

    case_id: str
    answer_status: AnswerStatus
    answer: str
    sources: tuple[AnswerSource, ...]
    max_relevance_score: float | None
    relevance_threshold: float
    retrieval_elapsed_ms: float
    generation_elapsed_ms: float
    total_elapsed_ms: float
    reranker_applied: bool
    reranker_fallback: bool


@dataclass(frozen=True)
class CitationOccurrence:
    """Describe one strictly-valid ``[来源N]`` citation inside an answer."""

    source_number: int
    start: int
    end: int


@dataclass(frozen=True)
class CitationParseResult:
    """Describe every citation found in one answer plus malformed look-alikes."""

    valid_syntax: tuple[CitationOccurrence, ...]
    malformed_fragments: tuple[str, ...]


@dataclass(frozen=True)
class FactScoring:
    """Describe the scoring of one annotated fact against an answer."""

    fact_id: str
    covered: bool
    matched_phrase: str | None
    supporting_evidence_indexes: tuple[int, ...]
    supporting_citation_numbers: tuple[int, ...]
    grounded_by_citation: bool


@dataclass(frozen=True)
class CaseAnswerMetrics:
    """Describe every per-case metric the reports need.

    Ratio fields are ``None`` only where the definition requires it
    (``fact_coverage`` / ``grounded_fact_coverage`` for unanswerable cases,
    ``citation_validity_rate`` when no strict citation exists).  NaN is never
    emitted.
    """

    case_id: str
    split: str
    category: str
    difficulty: str
    answerable: bool
    answer_status: str
    decision_correct: bool
    question: str
    reference_answer: str
    answer: str
    required_fact_count: int
    covered_fact_count: int
    grounded_fact_count: int
    fact_coverage: float | None
    grounded_fact_coverage: float | None
    complete_fact_coverage: bool
    complete_grounded_fact_coverage: bool
    forbidden_phrase_hits: tuple[str, ...]
    contradiction_free: bool
    citation_occurrence_count: int
    unique_cited_source_count: int
    valid_citation_occurrence_count: int
    invalid_citation_occurrence_count: int
    invalid_citation_numbers: tuple[int, ...]
    malformed_citation_count: int
    malformed_citation_fragments: tuple[str, ...]
    citation_validity_rate: float | None
    facts: tuple[FactScoring, ...]
    max_relevance_score: float | None
    relevance_threshold: float
    retrieval_elapsed_ms: float
    generation_elapsed_ms: float
    total_elapsed_ms: float
    reranker_applied: bool
    reranker_fallback: bool
    strict_pass: bool


@dataclass(frozen=True)
class LatencySummary:
    """Summarize one latency series with the standard library only."""

    count: int
    mean: float | None
    median: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class AggregateAnswerMetrics:
    """Describe the aggregate answer-quality metrics for one case group."""

    case_count: int
    answerable_count: int
    unanswerable_count: int
    answered_count: int
    refused_count: int
    true_answer_count: int
    false_refusal_count: int
    true_refusal_count: int
    false_answer_count: int
    decision_accuracy: float | None
    false_answer_rate: float | None
    false_refusal_rate: float | None
    average_fact_coverage: float | None
    average_grounded_fact_coverage: float | None
    complete_fact_coverage_rate: float | None
    complete_grounded_fact_coverage_rate: float | None
    contradiction_free_rate: float | None
    citation_occurrence_count: int
    valid_citation_occurrence_count: int
    invalid_citation_occurrence_count: int
    malformed_citation_count: int
    citation_validity_rate: float | None
    invalid_citation_case_rate: float | None
    malformed_citation_case_rate: float | None
    strict_pass_count: int
    strict_pass_rate: float | None


@dataclass(frozen=True)
class AnswerRunConfiguration:
    """Describe one reproducible answer-quality evaluation run.

    Fields that cannot be determined in offline mode are stored as ``None``
    instead of being fabricated.  Paths are repository-relative.
    """

    run_name: str
    mode: str
    model_label: str | None
    prompt_version: str | None
    embedding_model: str | None
    top_k: int | None
    relevance_threshold: float | None
    dataset_path: str
    annotations_path: str
    responses_path: str | None
    reranker_applied_any: bool | None
    reranker_fallback_any: bool | None


@dataclass(frozen=True)
class AnswerEvaluationRun:
    """Describe everything the report layer needs, with no hidden recomputation."""

    schema_version: str
    configuration: AnswerRunConfiguration
    cases: tuple[Any, ...]
    annotations: tuple[AnswerAnnotation, ...]
    responses: tuple[AnswerResponse, ...]
    case_metrics: tuple[CaseAnswerMetrics, ...]
    aggregate_metrics: AggregateAnswerMetrics
    metrics_by_split: dict[str, AggregateAnswerMetrics]
    metrics_by_category: dict[str, AggregateAnswerMetrics]
    metrics_by_difficulty: dict[str, AggregateAnswerMetrics]
    latency_retrieval: LatencySummary
    latency_generation: LatencySummary
    latency_total: LatencySummary

    @property
    def metrics_by_id(self) -> dict[str, CaseAnswerMetrics]:
        return {case.case_id: case for case in self.case_metrics}
