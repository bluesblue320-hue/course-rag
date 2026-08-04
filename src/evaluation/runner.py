"""Run the offline evaluation over the production retrieval pipeline."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.evaluation.corpus import LoadedCorpus
from src.evaluation.dataset import split_cases, validate_dataset_against_corpus
from src.evaluation.matching import count_matched_evidence, first_relevant_rank
from src.evaluation.metrics import (
    DecisionMetrics,
    RetrievalMetrics,
    ScoreDistribution,
    compute_decision_metrics,
    compute_retrieval_metrics,
    group_retrieval_metrics,
    reciprocal_rank_from_rank,
    score_distributions,
)
from src.evaluation.models import (
    MIN_REQUIRED_TOP_K,
    EvaluationCase,
    EvaluationResult,
    RetrievedSource,
)
from src.evaluation.threshold import (
    ThresholdCandidate,
    generate_thresholds,
    scan_thresholds,
    select_recommended_threshold,
    validate_weight,
)
from src.exceptions import EvaluationError
from src.knowledge_index import KnowledgeIndex
from src.rag_service import DEFAULT_MIN_RELEVANCE_SCORE


class EmbeddingProtocol(Protocol):
    """Describe the production embedding surface the runner depends on."""

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        ...

    def encode_query(self, query: str) -> np.ndarray:
        ...


@dataclass(frozen=True)
class RunConfiguration:
    """Describe one reproducible evaluation run, using repository-relative paths."""

    manifest_path: str
    dataset_path: str
    requested_top_k: int
    effective_top_k: int
    threshold_start: float
    threshold_end: float
    threshold_step: float
    current_threshold: float
    false_answer_weight: float
    false_refusal_weight: float


@dataclass(frozen=True)
class ChunkConfiguration:
    """Describe the production chunking parameters used to build the index."""

    chunk_size: int
    chunk_overlap: int
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class EvaluationRun:
    """Describe everything the report layer needs, with no hidden recomputation."""

    configuration: RunConfiguration
    chunk_configuration: ChunkConfiguration
    embedding_model: str
    cases: tuple[EvaluationCase, ...]
    results: tuple[EvaluationResult, ...]
    retrieval_metrics: RetrievalMetrics
    retrieval_metrics_by_category: dict[str, RetrievalMetrics]
    retrieval_metrics_by_difficulty: dict[str, RetrievalMetrics]
    retrieval_metrics_by_split: dict[str, RetrievalMetrics]
    calibration_candidates: tuple[ThresholdCandidate, ...]
    recommended_threshold: float
    calibration_metrics_current: DecisionMetrics
    calibration_metrics_recommended: DecisionMetrics
    test_metrics_current: DecisionMetrics
    test_metrics_recommended: DecisionMetrics
    score_distribution: dict[str, ScoreDistribution]

    @property
    def results_by_id(self) -> dict[str, EvaluationResult]:
        return {result.case_id: result for result in self.results}


def _to_sources(raw_results: list[dict[str, object]]) -> tuple[RetrievedSource, ...]:
    sources: list[RetrievedSource] = []
    for raw in raw_results:
        score = raw["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise EvaluationError("检索结果的分数无效")
        page_number = raw["page_number"]
        if page_number is not None and not isinstance(page_number, int):
            raise EvaluationError("检索结果的页码无效")
        sources.append(
            RetrievedSource(
                rank=int(raw["rank"]),  # type: ignore[arg-type]
                score=float(score),
                text=str(raw["text"]),
                chunk_index=int(raw["chunk_index"]),  # type: ignore[arg-type]
                document_id=str(raw["document_id"]),
                filename=str(raw["filename"]),
                page_number=page_number,
            )
        )
    return tuple(sources)


def _build_result(
    case: EvaluationCase,
    sources: tuple[RetrievedSource, ...],
) -> EvaluationResult:
    max_score = sources[0].score if sources else None
    total_evidence = len(case.expected_evidence)

    if case.answerable:
        matched_at_1 = count_matched_evidence(case.expected_evidence, sources, 1)
        matched_at_3 = count_matched_evidence(case.expected_evidence, sources, 3)
        matched_at_5 = count_matched_evidence(case.expected_evidence, sources, 5)
        rank = first_relevant_rank(case.expected_evidence, sources)
        hit_at_1: bool | None = matched_at_1 > 0
        hit_at_3: bool | None = matched_at_3 > 0
        hit_at_5: bool | None = matched_at_5 > 0
        # MRR uses the rank inside the retrieved Top-K list; a miss scores 0.
        reciprocal_rank: float | None = reciprocal_rank_from_rank(rank)
    else:
        matched_at_1 = matched_at_3 = matched_at_5 = 0
        rank = None
        hit_at_1 = hit_at_3 = hit_at_5 = None
        reciprocal_rank = None

    return EvaluationResult(
        case_id=case.id,
        split=case.split,
        question=case.question,
        answerable=case.answerable,
        category=case.category,
        difficulty=case.difficulty,
        max_relevance_score=max_score,
        first_relevant_rank=rank,
        matched_evidence_count_at_1=matched_at_1,
        matched_evidence_count_at_3=matched_at_3,
        matched_evidence_count_at_5=matched_at_5,
        total_evidence_count=total_evidence,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        reciprocal_rank=reciprocal_rank,
        sources=sources,
    )


def resolve_effective_top_k(top_k: int) -> int:
    """Return a Top-K large enough to compute Hit@5 and Recall@5."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise EvaluationError("top_k 必须是整数")
    if top_k <= 0:
        raise EvaluationError("top_k 必须大于 0")
    return max(top_k, MIN_REQUIRED_TOP_K)


def run_evaluation(
    *,
    corpus: LoadedCorpus,
    cases: tuple[EvaluationCase, ...],
    embedding_service: EmbeddingProtocol,
    embedding_model: str,
    manifest_path: str,
    dataset_path: str,
    top_k: int,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
    current_threshold: float = DEFAULT_MIN_RELEVANCE_SCORE,
    false_answer_weight: float = 3.0,
    false_refusal_weight: float = 1.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> EvaluationRun:
    """Execute retrieval once per question and derive every reported metric.

    Document embeddings are built exactly once, each question is embedded
    exactly once, and the threshold sweep reuses the recorded max scores
    instead of repeating retrieval.
    """
    if not cases:
        raise EvaluationError("评估数据集不能为空")

    validate_dataset_against_corpus(cases, corpus.chunks)

    effective_top_k = resolve_effective_top_k(top_k)
    answer_weight = validate_weight(false_answer_weight, "false-answer-weight")
    refusal_weight = validate_weight(
        false_refusal_weight,
        "false-refusal-weight",
    )

    document_embeddings = embedding_service.encode_documents(
        [chunk.text for chunk in corpus.chunks]
    )
    index = KnowledgeIndex()
    index.replace(list(corpus.chunks), document_embeddings)

    results: list[EvaluationResult] = []
    for case in cases:
        query_embedding = embedding_service.encode_query(case.question)
        raw_results = index.search(query_embedding, top_k=effective_top_k)
        results.append(_build_result(case, _to_sources(raw_results)))

    all_results = tuple(results)
    calibration_results = tuple(
        result for result in all_results if result.split == "calibration"
    )
    test_results = tuple(
        result for result in all_results if result.split == "test"
    )
    if not calibration_results:
        raise EvaluationError("calibration split 不能为空")
    if not test_results:
        raise EvaluationError("test split 不能为空")

    thresholds = generate_thresholds(
        threshold_start,
        threshold_end,
        threshold_step,
    )
    calibration_candidates = scan_thresholds(
        calibration_results,
        thresholds,
        answer_weight,
        refusal_weight,
    )
    recommended = select_recommended_threshold(calibration_candidates)

    return EvaluationRun(
        configuration=RunConfiguration(
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            requested_top_k=top_k,
            effective_top_k=effective_top_k,
            threshold_start=float(threshold_start),
            threshold_end=float(threshold_end),
            threshold_step=float(threshold_step),
            current_threshold=float(current_threshold),
            false_answer_weight=answer_weight,
            false_refusal_weight=refusal_weight,
        ),
        chunk_configuration=ChunkConfiguration(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_count=len(corpus.documents),
            chunk_count=len(corpus.chunks),
        ),
        embedding_model=embedding_model,
        cases=cases,
        results=all_results,
        retrieval_metrics=compute_retrieval_metrics(all_results),
        retrieval_metrics_by_category=group_retrieval_metrics(
            all_results,
            "category",
        ),
        retrieval_metrics_by_difficulty=group_retrieval_metrics(
            all_results,
            "difficulty",
        ),
        retrieval_metrics_by_split=group_retrieval_metrics(
            all_results,
            "split",
        ),
        calibration_candidates=calibration_candidates,
        recommended_threshold=recommended.threshold,
        calibration_metrics_current=compute_decision_metrics(
            calibration_results,
            current_threshold,
        ),
        calibration_metrics_recommended=recommended.metrics,
        test_metrics_current=compute_decision_metrics(
            test_results,
            current_threshold,
        ),
        test_metrics_recommended=compute_decision_metrics(
            test_results,
            recommended.threshold,
        ),
        score_distribution=score_distributions(all_results),
    )


def calibration_split(
    cases: tuple[EvaluationCase, ...],
) -> tuple[EvaluationCase, ...]:
    """Return the calibration cases used for threshold selection."""
    return split_cases(cases, "calibration")


def test_split(cases: tuple[EvaluationCase, ...]) -> tuple[EvaluationCase, ...]:
    """Return the held-out test cases, never used to tune the threshold."""
    return split_cases(cases, "test")
