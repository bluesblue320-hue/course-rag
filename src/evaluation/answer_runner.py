"""Run the answer-quality evaluation in one of two explicit modes.

Mode A (offline) scores an existing responses file and never loads an
embedding model, a reranker, an LLM, a database, or the network.  It is fully
deterministic: identical inputs and identical code produce byte-identical
reports.

Mode B (live) requires an explicit ``--live`` flag and instantiates the
production knowledge index, embedding service, PromptBuilder,
GenerationService, and RagService to produce new responses.  It never
touches the production threshold, never reads or writes production data, and
always closes the generation service in a ``finally`` block.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.evaluation.answer_annotations import load_answer_annotations
from src.evaluation.answer_citations import parse_citations
from src.evaluation.answer_metrics import (
    compute_aggregate_metrics,
    compute_case_metrics,
    group_case_metrics,
    summarize_latency,
)
from src.evaluation.answer_models import (
    AnswerAnnotation,
    AnswerEvaluationRun,
    AnswerResponse,
    AnswerRunConfiguration,
    AnswerSource,
    CaseAnswerMetrics,
)
from src.evaluation.answer_report import (
    SCHEMA_VERSION,
    write_answer_reports,
)
from src.evaluation.answer_responses import load_answer_responses
from src.evaluation.dataset import validate_dataset_against_corpus
from src.evaluation.models import EvaluationCase
from src.exceptions import AnswerEvaluationError

if TYPE_CHECKING:
    from src.evaluation.corpus import LoadedCorpus


class EmbeddingServiceProtocol(Protocol):
    """Describe the production embedding surface live mode depends on."""

    def encode_documents(self, texts: list[str]) -> Any:
        ...

    def encode_query(self, query: str) -> Any:
        ...


class GenerationServiceProtocol(Protocol):
    """Describe the production generation surface live mode depends on."""

    def generate(self, prompt: str) -> str:
        ...

    def close(self) -> None:
        ...


def _sources_to_answer_sources(
    raw_sources: list[dict[str, Any]],
) -> tuple[AnswerSource, ...]:
    sources: list[AnswerSource] = []
    for raw in raw_sources:
        page_number = raw.get("page_number")
        if page_number is not None:
            page_number = int(page_number)
        sources.append(
            AnswerSource(
                rank=int(raw.get("rank", 0)),
                score=float(raw.get("score", 0.0)),
                text=str(raw.get("text", "")),
                chunk_index=int(raw.get("chunk_index", 0)),
                document_id=str(raw.get("document_id", "")),
                filename=str(raw.get("filename", "")),
                page_number=page_number,
            )
        )
    return tuple(sources)


def _to_answer_response(
    case_id: str,
    raw: dict[str, Any],
) -> AnswerResponse:
    return AnswerResponse(
        case_id=case_id,
        answer_status=raw["answer_status"],  # type: ignore[arg-type]
        answer=str(raw["answer"]),
        sources=_sources_to_answer_sources(list(raw.get("sources", []))),
        max_relevance_score=raw.get("max_relevance_score"),
        relevance_threshold=float(raw.get("relevance_threshold", 0.0)),
        retrieval_elapsed_ms=float(raw.get("retrieval_elapsed_ms", 0.0)),
        generation_elapsed_ms=float(raw.get("generation_elapsed_ms", 0.0)),
        total_elapsed_ms=float(raw.get("total_elapsed_ms", 0.0)),
        reranker_applied=bool(raw.get("reranker_applied", False)),
        reranker_fallback=bool(raw.get("reranker_fallback", False)),
    )


def _build_run(
    *,
    cases: tuple[EvaluationCase, ...],
    annotations: tuple[AnswerAnnotation, ...],
    responses: tuple[AnswerResponse, ...],
    configuration: AnswerRunConfiguration,
) -> AnswerEvaluationRun:
    annotation_by_id = {annotation.case_id: annotation for annotation in annotations}
    response_by_id = {response.case_id: response for response in responses}

    case_metrics: list[CaseAnswerMetrics] = []
    for case in cases:
        annotation = annotation_by_id[case.id]
        response = response_by_id[case.id]
        citation_result = parse_citations(response.answer)
        case_metrics.append(
            compute_case_metrics(case, annotation, response, citation_result)
        )
    all_metrics = tuple(case_metrics)

    return AnswerEvaluationRun(
        schema_version=SCHEMA_VERSION,
        configuration=configuration,
        cases=cases,
        annotations=annotations,
        responses=responses,
        case_metrics=all_metrics,
        aggregate_metrics=compute_aggregate_metrics(all_metrics),
        metrics_by_split=group_case_metrics(all_metrics, "split"),
        metrics_by_category=group_case_metrics(all_metrics, "category"),
        metrics_by_difficulty=group_case_metrics(all_metrics, "difficulty"),
        latency_retrieval=summarize_latency(
            case.retrieval_elapsed_ms for case in all_metrics
        ),
        latency_generation=summarize_latency(
            case.generation_elapsed_ms for case in all_metrics
        ),
        latency_total=summarize_latency(
            case.total_elapsed_ms for case in all_metrics
        ),
    )


def _reranker_any_flag(
    responses: tuple[AnswerResponse, ...],
) -> tuple[bool | None, bool | None]:
    if not responses:
        return None, None
    return (
        any(response.reranker_applied for response in responses),
        any(response.reranker_fallback for response in responses),
    )


def run_offline_answer_evaluation(
    *,
    cases: tuple[EvaluationCase, ...],
    annotations: tuple[AnswerAnnotation, ...],
    responses: tuple[AnswerResponse, ...],
    dataset_path: str,
    annotations_path: str,
    responses_path: str,
    run_name: str,
    model_label: str | None = None,
    prompt_version: str | None = None,
) -> AnswerEvaluationRun:
    """Score existing responses with zero model, database, or network use.

    All three path arguments must already be safe display values
    (repository-relative paths or bare filenames); they are written to the
    report verbatim.
    """
    if len(cases) != len(annotations) or len(cases) != len(responses):
        raise AnswerEvaluationError("数据集、标注与回答结果的数量必须一致")

    reranker_applied_any, reranker_fallback_any = _reranker_any_flag(responses)
    configuration = AnswerRunConfiguration(
        run_name=run_name,
        mode="offline",
        model_label=model_label,
        prompt_version=prompt_version,
        embedding_model=None,
        top_k=None,
        relevance_threshold=None,
        dataset_path=dataset_path,
        annotations_path=annotations_path,
        responses_path=responses_path,
        reranker_applied_any=reranker_applied_any,
        reranker_fallback_any=reranker_fallback_any,
    )
    return _build_run(
        cases=cases,
        annotations=annotations,
        responses=responses,
        configuration=configuration,
    )


def run_live_answer_evaluation(
    *,
    corpus: LoadedCorpus,
    cases: tuple[EvaluationCase, ...],
    annotations: tuple[AnswerAnnotation, ...],
    embedding_service: EmbeddingServiceProtocol,
    generation_service: GenerationServiceProtocol,
    embedding_model: str,
    top_k: int,
    threshold: float,
    dataset_path: str,
    annotations_path: str,
    run_name: str,
    model_label: str | None,
    prompt_version: str | None,
) -> AnswerEvaluationRun:
    """Run the production RagService pipeline over every case.

    Document embeddings are computed exactly once, every question is encoded
    exactly once, and the generation service is always closed in a
    ``finally`` block.  This mode intentionally uses the Memory
    KnowledgeIndex; pgvector evaluation is out of scope for V1.
    """
    from src.knowledge_index import KnowledgeIndex
    from src.prompt_builder import PromptBuilder
    from src.rag_service import RagService

    if not cases:
        raise AnswerEvaluationError("评估数据集不能为空")
    validate_dataset_against_corpus(cases, corpus.chunks)

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise AnswerEvaluationError("top_k 必须是正整数")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise AnswerEvaluationError("threshold 必须是 0 到 1 之间的有限数字")

    document_embeddings = embedding_service.encode_documents(
        [chunk.text for chunk in corpus.chunks]
    )
    index = KnowledgeIndex()
    index.replace(list(corpus.chunks), document_embeddings)

    rag_service = RagService(
        embedding_service=embedding_service,
        retriever=index,
        prompt_builder=PromptBuilder(),
        generation_service=generation_service,
        min_relevance_score=threshold,
    )

    responses: list[AnswerResponse] = []
    try:
        for case in cases:
            raw = rag_service.answer(case.question, top_k=top_k)
            responses.append(_to_answer_response(case.id, raw))
    finally:
        close = getattr(generation_service, "close", None)
        if callable(close):
            close()

    response_tuple = tuple(responses)
    reranker_applied_any, reranker_fallback_any = _reranker_any_flag(response_tuple)
    configuration = AnswerRunConfiguration(
        run_name=run_name,
        mode="live",
        model_label=model_label,
        prompt_version=prompt_version,
        embedding_model=embedding_model,
        top_k=top_k,
        relevance_threshold=float(threshold),
        dataset_path=dataset_path,
        annotations_path=annotations_path,
        responses_path=None,
        reranker_applied_any=reranker_applied_any,
        reranker_fallback_any=reranker_fallback_any,
    )
    return _build_run(
        cases=cases,
        annotations=annotations,
        responses=response_tuple,
        configuration=configuration,
    )


def write_responses_jsonl(
    responses: tuple[AnswerResponse, ...],
    output_dir: Path,
) -> Path:
    """Write the raw responses file for live runs.

    The file keeps answers, sources, timings, and reranker flags, but never
    the prompt, API keys, HTTP headers, full provider responses, or
    tracebacks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "responses.jsonl"
    lines = []
    for response in responses:
        payload = {
            "case_id": response.case_id,
            "answer_status": response.answer_status,
            "answer": response.answer,
            "sources": [
                {
                    "rank": source.rank,
                    "score": source.score,
                    "text": source.text,
                    "chunk_index": source.chunk_index,
                    "document_id": source.document_id,
                    "filename": source.filename,
                    "page_number": source.page_number,
                }
                for source in response.sources
            ],
            "max_relevance_score": response.max_relevance_score,
            "relevance_threshold": response.relevance_threshold,
            "retrieval_elapsed_ms": response.retrieval_elapsed_ms,
            "generation_elapsed_ms": response.generation_elapsed_ms,
            "total_elapsed_ms": response.total_elapsed_ms,
            "reranker_applied": response.reranker_applied,
            "reranker_fallback": response.reranker_fallback,
        }
        lines.append(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate_answers_offline(
    *,
    cases: tuple[EvaluationCase, ...],
    annotations: tuple[AnswerAnnotation, ...],
    responses: tuple[AnswerResponse, ...],
    dataset_path: str,
    annotations_path: str,
    responses_path: str,
    output_dir: Path,
    run_name: str,
    model_label: str | None,
    prompt_version: str | None,
) -> AnswerEvaluationRun:
    """Score already-loaded inputs and write reports in one call.

    The three path arguments must be safe display values (repository-relative
    paths or bare filenames) so reports never leak absolute local paths.
    Inputs are passed in already parsed to keep the CLI from loading them
    twice.
    """
    run = run_offline_answer_evaluation(
        cases=cases,
        annotations=annotations,
        responses=responses,
        dataset_path=dataset_path,
        annotations_path=annotations_path,
        responses_path=responses_path,
        run_name=run_name,
        model_label=model_label,
        prompt_version=prompt_version,
    )
    write_answer_reports(run, output_dir)
    return run
