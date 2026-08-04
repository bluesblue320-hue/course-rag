"""Run one offline evaluation with the production retrieval stack."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.documents import ChunkRecord
from src.embedding import EmbeddingService
from src.knowledge_index import KnowledgeIndex
from src.rag_service import DEFAULT_MIN_RELEVANCE_SCORE

from .corpus import load_corpus
from .dataset import (
    load_dataset,
    validate_dataset_document_ids,
    validate_dataset_integrity,
)
from .matching import compute_evidence_match
from .metrics import (
    aggregate_retrieval_by,
    decision_metrics,
    retrieval_metrics,
    score_distribution,
)
from .models import EvaluationCase, EvaluationDataError, EvaluationResult
from .threshold import (
    generate_thresholds,
    scan_decision_metrics,
    select_recommended_threshold,
)

REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class RunConfiguration:
    """Describe one immutable evaluation run."""

    manifest_path: str
    dataset_path: str
    top_k: int
    threshold_start: float
    threshold_end: float
    threshold_step: float
    current_threshold: float
    false_answer_weight: float
    false_refusal_weight: float
    chunk_size: int
    chunk_overlap: int


class EvaluationRunner:
    """Build the production index once and evaluate every labeled case."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        base_dir: Path | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._base_dir = base_dir

    def _build_index(
        self,
        chunks_by_document: dict[str, list[ChunkRecord]],
    ) -> KnowledgeIndex:
        ordered_chunks = [
            chunk
            for chunks in chunks_by_document.values()
            for chunk in chunks
        ]
        embeddings = self._embedding_service.encode_documents(
            [chunk.text for chunk in ordered_chunks]
        )
        index = KnowledgeIndex(ordered_chunks, embeddings)
        return index

    def _evaluate_case(
        self,
        case: EvaluationCase,
        index: KnowledgeIndex,
        top_k: int,
    ) -> EvaluationResult:
        query_embedding = self._embedding_service.encode_query(case.question)
        sources = index.search(query_embedding, top_k=top_k)
        match = compute_evidence_match(sources, case)
        max_score: float | None = None
        for source in sources:
            score = source.get("score")
            if score is not None:
                max_score = float(score)
                break
        return EvaluationResult(
            case_id=case.id,
            split=case.split,
            question=case.question,
            answerable=case.answerable,
            category=case.category,
            difficulty=case.difficulty,
            max_relevance_score=max_score,
            first_relevant_rank=match.first_relevant_rank,
            matched_evidence_count_at_1=match.matched_evidence_count_at_1,
            matched_evidence_count_at_3=match.matched_evidence_count_at_3,
            matched_evidence_count_at_5=match.matched_evidence_count_at_5,
            total_evidence_count=match.total_evidence_count,
            hit_at_1=match.hit_at_1,
            hit_at_3=match.hit_at_3,
            hit_at_5=match.hit_at_5,
            reciprocal_rank=match.reciprocal_rank,
            sources=tuple(sources),
        )

    def run(
        self,
        manifest_path: Path,
        dataset_path: Path,
        top_k: int,
        threshold_start: float = 0.20,
        threshold_end: float = 0.60,
        threshold_step: float = 0.01,
        current_threshold: float = DEFAULT_MIN_RELEVANCE_SCORE,
        false_answer_weight: float = 3.0,
        false_refusal_weight: float = 1.0,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run retrieval, decision evaluation, and threshold calibration."""
        manifest_path = Path(manifest_path)
        dataset_path = Path(dataset_path)
        if top_k <= 0:
            raise EvaluationDataError("top_k 必须大于 0")

        chunks_by_document = load_corpus(
            manifest_path,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            base_dir=self._base_dir,
        )
        cases = load_dataset(dataset_path)
        validate_dataset_document_ids(
            cases,
            set(chunks_by_document.keys()),
        )
        validate_dataset_integrity(cases, chunks_by_document)

        index = self._build_index(chunks_by_document)
        results = [
            self._evaluate_case(case, index, top_k) for case in cases
        ]

        calibration_results = [
            result for result in results if result.split == "calibration"
        ]
        test_results = [
            result for result in results if result.split == "test"
        ]

        thresholds = generate_thresholds(
            threshold_start,
            threshold_end,
            threshold_step,
        )
        calibration_rows = scan_decision_metrics(
            calibration_results,
            thresholds,
            false_answer_weight,
            false_refusal_weight,
        )
        recommended_row = select_recommended_threshold(calibration_rows)
        recommended_threshold = recommended_row.threshold

        if output_dir is not None:
            from .report import write_reports

            write_reports(
                output_dir,
                self._build_report_data(
                    manifest_path=manifest_path,
                    dataset_path=dataset_path,
                    top_k=top_k,
                    threshold_start=threshold_start,
                    threshold_end=threshold_end,
                    threshold_step=threshold_step,
                    current_threshold=current_threshold,
                    false_answer_weight=false_answer_weight,
                    false_refusal_weight=false_refusal_weight,
                    cases=cases,
                    results=results,
                    calibration_results=calibration_results,
                    test_results=test_results,
                    calibration_rows=calibration_rows,
                    recommended_row=recommended_row,
                ),
                results,
                current_threshold,
                recommended_threshold,
            )

        return self._build_report_data(
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            top_k=top_k,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            threshold_step=threshold_step,
            current_threshold=current_threshold,
            false_answer_weight=false_answer_weight,
            false_refusal_weight=false_refusal_weight,
            cases=cases,
            results=results,
            calibration_results=calibration_results,
            test_results=test_results,
            calibration_rows=calibration_rows,
            recommended_row=recommended_row,
        )

    def _build_report_data(
        self,
        *,
        manifest_path: Path,
        dataset_path: Path,
        top_k: int,
        threshold_start: float,
        threshold_end: float,
        threshold_step: float,
        current_threshold: float,
        false_answer_weight: float,
        false_refusal_weight: float,
        cases: list[EvaluationCase],
        results: list[EvaluationResult],
        calibration_results: list[EvaluationResult],
        test_results: list[EvaluationResult],
        calibration_rows: list[Any],
        recommended_row: Any,
    ) -> dict[str, Any]:
        recommended_threshold = recommended_row.threshold
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "embedding_model": self._embedding_service.model_name,
            "chunk_configuration": {
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
            },
            "dataset_counts": {
                "total": len(cases),
                "answerable": sum(1 for case in cases if case.answerable),
                "unanswerable": sum(1 for case in cases if not case.answerable),
            },
            "split_counts": {
                "calibration": len(calibration_results),
                "test": len(test_results),
            },
            "category_counts": _count_values(cases, "category"),
            "difficulty_counts": _count_values(cases, "difficulty"),
            "retrieval_metrics": {
                "overall": retrieval_metrics(results),
                "by_category": aggregate_retrieval_by(results, "category"),
                "by_difficulty": aggregate_retrieval_by(results, "difficulty"),
                "by_split": {
                    "calibration": retrieval_metrics(calibration_results),
                    "test": retrieval_metrics(test_results),
                },
            },
            "calibration_threshold_table": [
                {
                    "threshold": row.threshold,
                    **row.metrics,
                    "weighted_cost": row.weighted_cost,
                }
                for row in calibration_rows
            ],
            "current_threshold": current_threshold,
            "recommended_threshold": recommended_threshold,
            "calibration_metrics_current": {
                "decision": decision_metrics(calibration_results, current_threshold),
                "retrieval": retrieval_metrics(calibration_results),
            },
            "calibration_metrics_recommended": {
                "decision": decision_metrics(
                    calibration_results, recommended_threshold
                ),
                "retrieval": retrieval_metrics(calibration_results),
            },
            "test_metrics_current": {
                "decision": decision_metrics(test_results, current_threshold),
                "retrieval": retrieval_metrics(test_results),
            },
            "test_metrics_recommended": {
                "decision": decision_metrics(test_results, recommended_threshold),
                "retrieval": retrieval_metrics(test_results),
            },
            "score_distribution_summary": score_distribution(results),
            "failure_case_ids": _failure_case_ids(
                results,
                current_threshold,
                recommended_threshold,
            ),
            "run_configuration": {
                "manifest": manifest_path.name,
                "dataset": dataset_path.name,
                "top_k": top_k,
                "threshold_start": threshold_start,
                "threshold_end": threshold_end,
                "threshold_step": threshold_step,
                "current_threshold": current_threshold,
                "false_answer_weight": false_answer_weight,
                "false_refusal_weight": false_refusal_weight,
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
            },
        }


def _count_values(cases: list[EvaluationCase], attribute: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = getattr(case, attribute)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _failure_case_ids(
    results: list[EvaluationResult],
    current_threshold: float,
    recommended_threshold: float,
) -> dict[str, list[str]]:
    from src.rag_service import has_sufficient_context

    retrieval_failures: list[str] = []
    false_refusals: list[str] = []
    false_answers: list[str] = []
    for result in results:
        if result.answerable and result.hit_at_5 is not True:
            retrieval_failures.append(result.case_id)
        predicted_current = has_sufficient_context(
            result.max_relevance_score,
            current_threshold,
        )
        predicted_recommended = has_sufficient_context(
            result.max_relevance_score,
            recommended_threshold,
        )
        if result.answerable and not predicted_recommended:
            false_refusals.append(result.case_id)
        if not result.answerable and predicted_current:
            false_answers.append(result.case_id)
    return {
        "retrieval_failures": retrieval_failures,
        "false_refusals": false_refusals,
        "false_answers": false_answers,
    }
