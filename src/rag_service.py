"""Orchestrate retrieval, prompt construction, and answer generation."""

from time import perf_counter
from typing import Any


class RagService:
    """Connect injected RAG components without creating their dependencies."""

    def __init__(
        self,
        embedding_service: Any,
        retriever: Any,
        prompt_builder: Any,
        generation_service: Any,
    ) -> None:
        self._embedding_service = embedding_service
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._generation_service = generation_service

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

        prompt = self._prompt_builder.build(cleaned_question, sources)
        generation_started_at = perf_counter()
        answer = self._generation_service.generate(prompt)
        generation_finished_at = perf_counter()

        return {
            "question": cleaned_question,
            "answer": answer,
            "sources": sources,
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
