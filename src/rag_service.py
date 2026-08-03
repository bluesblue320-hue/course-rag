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


def _validate_min_relevance_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RagConfigurationError(_THRESHOLD_ERROR)
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RagConfigurationError(_THRESHOLD_ERROR)
    return threshold


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


class RagService:
    """Connect injected RAG components without creating their dependencies."""

    def __init__(
        self,
        embedding_service: Any,
        retriever: Any,
        prompt_builder: Any,
        generation_service: Any,
        min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    ) -> None:
        self._embedding_service = embedding_service
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._generation_service = generation_service
        self._min_relevance_score = _validate_min_relevance_score(
            min_relevance_score
        )

    @property
    def min_relevance_score(self) -> float:
        return self._min_relevance_score

    @staticmethod
    def _get_max_relevance_score(sources: list[dict[str, Any]]) -> float | None:
        if not sources:
            return None
        score = sources[0].get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("检索结果的最高相关性分数无效")
        parsed_score = float(score)
        if not math.isfinite(parsed_score) or not -1 <= parsed_score <= 1:
            raise ValueError("检索结果的最高相关性分数无效")
        return parsed_score

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
        sources = self._retriever.search(query_embedding, top_k=top_k)
        retrieval_finished_at = perf_counter()

        max_relevance_score = self._get_max_relevance_score(sources)
        if (
            max_relevance_score is None
            or max_relevance_score < self._min_relevance_score
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
        }
