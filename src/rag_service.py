"""Orchestrate retrieval, prompt construction, and answer generation."""

import math
import os
from time import perf_counter
from typing import Any

from src.exceptions import RagConfigurationError

DEFAULT_MIN_RELEVANCE_SCORE = 0.35
INSUFFICIENT_CONTEXT_ANSWER = (
    "当前课程资料中没有足够信息回答这个问题。请尝试换一种问法，"
    "或切换到语义检索查看最接近的课程原文。"
)
_THRESHOLD_ERROR = "RAG_MIN_RELEVANCE_SCORE 必须是 0 到 1 之间的有限数字"
_MAX_SCORE_ERROR = "检索结果的最高相关性分数无效"


def _validate_min_relevance_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RagConfigurationError(_THRESHOLD_ERROR)
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RagConfigurationError(_THRESHOLD_ERROR)
    return threshold


def has_sufficient_context(
    max_relevance_score: float | None,
    threshold: float,
) -> bool:
    """Decide whether the best retrieval score supports generating an answer.

    This is the single source of truth for the answer/refuse boundary so the
    offline evaluation tooling cannot drift away from production behaviour.
    A missing score means no usable retrieval result, which always refuses.

    The decision is always based on the highest **vector retrieval score**
    (cosine similarity), never on a reranker score.  This keeps the refusal
    semantics stable when an optional reranker is enabled or falls back.
    """
    if max_relevance_score is None:
        return False
    if isinstance(max_relevance_score, bool) or not isinstance(
        max_relevance_score, (int, float)
    ):
        raise ValueError(_MAX_SCORE_ERROR)
    score = float(max_relevance_score)
    if not math.isfinite(score):
        raise ValueError(_MAX_SCORE_ERROR)
    return score >= _validate_min_relevance_score(threshold)


def resolve_min_relevance_score(explicit_value: object | None = None) -> float:
    """Resolve and validate the relevance threshold without mutating state."""
    if explicit_value is not None:
        return _validate_min_relevance_score(explicit_value)

    raw_value = os.getenv("RAG_MIN_RELEVANCE_SCORE")
    if raw_value is None:
        return DEFAULT_MIN_RELEVANCE_SCORE

    try:
        parsed_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RagConfigurationError(_THRESHOLD_ERROR) from exc
    return _validate_min_relevance_score(parsed_value)


def _sources_to_dicts(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize source dicts so downstream consumers see stable keys.

    Uses ``.get()`` with defaults so test fakes that only provide a subset
    of keys (e.g. ``rank``, ``score``, ``text``, ``chunk_index``) still work.
    """
    return [
        {
            "rank": s.get("rank", 0),
            "score": s.get("score", 0.0),
            "text": s.get("text", ""),
            "chunk_index": s.get("chunk_index", 0),
            "document_id": s.get("document_id", ""),
            "filename": s.get("filename", ""),
            "page_number": s.get("page_number"),
        }
        for s in sources
    ]


class RagService:
    """Connect injected RAG components without creating their dependencies.

    When ``retrieval_service`` is provided, the service uses two-stage
    retrieval (vector candidate pool + optional reranking).  When it is
    ``None``, the service falls back to the original direct vector search,
    preserving full backward compatibility.
    """

    def __init__(
        self,
        embedding_service: Any,
        retriever: Any,
        prompt_builder: Any,
        generation_service: Any,
        min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
        retrieval_service: Any = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._generation_service = generation_service
        self._retrieval_service = retrieval_service
        self._min_relevance_score = _validate_min_relevance_score(
            min_relevance_score
        )

    @property
    def min_relevance_score(self) -> float:
        return self._min_relevance_score

    @property
    def retrieval_service(self) -> Any:
        return self._retrieval_service

    @staticmethod
    def _get_max_relevance_score(sources: list[dict[str, Any]]) -> float | None:
        if not sources:
            return None
        score = sources[0].get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(_MAX_SCORE_ERROR)
        parsed_score = float(score)
        if not math.isfinite(parsed_score) or not -1 <= parsed_score <= 1:
            raise ValueError(_MAX_SCORE_ERROR)
        return parsed_score

    def _retrieve(
        self,
        query_embedding: object,
        query_text: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], float | None, bool, bool]:
        """Run retrieval and return (sources, max_score, reranker_applied, fallback).

        When a retrieval service is available, it performs two-stage
        retrieval.  Otherwise the original direct vector search is used.
        """
        if self._retrieval_service is not None:
            outcome = self._retrieval_service.retrieve(
                query_embedding,
                query_text,
                final_top_k=top_k,
            )
            sources = [
                {
                    "rank": chunk.final_rank,
                    "score": chunk.retrieval_score,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "retrieval_rank": chunk.retrieval_rank,
                    "rerank_score": chunk.rerank_score,
                    "reranker_applied": outcome.reranker_applied,
                }
                for chunk in outcome.sources
            ]
            return (
                sources,
                outcome.max_retrieval_score,
                outcome.reranker_applied,
                outcome.reranker_fallback,
            )

        # Original direct vector search (backward compatible).
        raw_sources = self._retriever.search(query_embedding, top_k=top_k)
        # Pass through directly to preserve identity and key set for
        # existing tests and downstream consumers.
        return (
            raw_sources,
            self._get_max_relevance_score(raw_sources),
            False,
            False,
        )

    def answer(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict[str, object]:
        """Run the injected RAG pipeline and return its answer with timings."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        retrieval_started_at = perf_counter()
        query_embedding = self._embedding_service.encode_query(cleaned_question)
        sources, max_relevance_score, reranker_applied, reranker_fallback = (
            self._retrieve(query_embedding, cleaned_question, top_k)
        )
        retrieval_finished_at = perf_counter()

        if not has_sufficient_context(
            max_relevance_score,
            self._min_relevance_score,
        ):
            retrieval_elapsed_ms = round(
                (retrieval_finished_at - retrieval_started_at) * 1000,
                2,
            )
            return {
                "question": cleaned_question,
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "answer_status": "insufficient_context",
                "sources": sources,
                "max_relevance_score": max_relevance_score,
                "relevance_threshold": self._min_relevance_score,
                "retrieval_elapsed_ms": retrieval_elapsed_ms,
                "generation_elapsed_ms": 0.0,
                "total_elapsed_ms": retrieval_elapsed_ms,
                "reranker_applied": reranker_applied,
                "reranker_fallback": reranker_fallback,
            }

        prompt = self._prompt_builder.build(cleaned_question, sources)
        generation_started_at = perf_counter()
        answer = self._generation_service.generate(prompt)
        generation_finished_at = perf_counter()

        return {
            "question": cleaned_question,
            "answer": answer,
            "answer_status": "answered",
            "sources": sources,
            "max_relevance_score": max_relevance_score,
            "relevance_threshold": self._min_relevance_score,
            "retrieval_elapsed_ms": round(
                (retrieval_finished_at - retrieval_started_at) * 1000,
                2,
            ),
            "generation_elapsed_ms": round(
                (generation_finished_at - generation_started_at) * 1000,
                2,
            ),
            "total_elapsed_ms": round(
                (generation_finished_at - retrieval_started_at) * 1000,
                2,
            ),
            "reranker_applied": reranker_applied,
            "reranker_fallback": reranker_fallback,
        }
