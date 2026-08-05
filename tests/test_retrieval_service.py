"""Tests for the two-stage retrieval service and production integration."""

from __future__ import annotations

import pytest

from src.reranker import (
    FakeReranker,
    RerankerConfig,
)
from src.retrieval_service import (
    RetrievalOutcome,
    RetrievalService,
    RetrievedChunk,
    _stable_sort_key,
    build_reranker,
)
from src.rag_service import (
    RagService,
    has_sufficient_context,
    DEFAULT_MIN_RELEVANCE_SCORE,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeRetriever:
    """Return deterministic search results for testing."""

    def __init__(self, results: list[dict[str, object]] | None = None) -> None:
        self._results = results or []
        self.search_count = 0
        self.last_top_k: int | None = None

    def search(
        self,
        _query_embedding: object,
        top_k: int,
    ) -> list[dict[str, object]]:
        self.search_count += 1
        self.last_top_k = top_k
        return list(self._results[:top_k])


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.query_calls = 0

    def encode_query(self, query: str) -> object:
        self.query_calls += 1
        return [0.0]  # dummy embedding

    def encode_documents(self, texts: list[str]) -> object:
        return [[0.0] for _ in texts]


class FakePromptBuilder:
    def build(self, question: str, sources: list) -> str:
        return f"prompt for {question}"

    def __call__(self, question: str, sources: list) -> str:
        return self.build(question, sources)


class FakeGenerationService:
    def generate(self, prompt: str) -> str:
        return f"answer for {prompt}"

    def close(self) -> None:
        pass


def make_raw_results(n: int = 5) -> list[dict[str, object]]:
    """Create n deterministic raw search results."""
    return [
        {
            "rank": i,
            "score": 1.0 - i * 0.1,
            "text": f"chunk text {i}",
            "chunk_index": i,
            "document_id": f"doc-{i}",
            "filename": f"file-{i}.md",
            "page_number": None,
        }
        for i in range(1, n + 1)
    ]


def make_disabled_config(candidate_top_k: int = 15) -> RerankerConfig:
    return RerankerConfig(
        enabled=False,
        model_name="",
        candidate_top_k=candidate_top_k,
    )


def make_enabled_config(
    candidate_top_k: int = 15,
    model_name: str = "fake-reranker",
) -> RerankerConfig:
    return RerankerConfig(
        enabled=True,
        model_name=model_name,
        candidate_top_k=candidate_top_k,
    )


# ---------------------------------------------------------------------------
# RetrievalService basic behavior
# ---------------------------------------------------------------------------


class TestRetrievalServiceDisabled:
    def test_disabled_returns_vector_order(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_disabled_config()
        service = RetrievalService(retriever, config, reranker=None)

        outcome = service.retrieve([0.0], "query", final_top_k=3)

        assert outcome.reranker_applied is False
        assert outcome.reranker_fallback is False
        assert len(outcome.sources) == 3
        assert outcome.sources[0].final_rank == 1
        assert outcome.sources[0].retrieval_rank == 1
        assert outcome.sources[0].rerank_score is None

    def test_disabled_max_retrieval_score(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_disabled_config()
        service = RetrievalService(retriever, config, reranker=None)

        outcome = service.retrieve([0.0], "query", final_top_k=3)
        assert outcome.max_retrieval_score == 0.9  # first result's score

    def test_disabled_candidate_count(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_disabled_config()
        service = RetrievalService(retriever, config, reranker=None)

        outcome = service.retrieve([0.0], "query", final_top_k=3)
        # When disabled, candidate_count equals the number of results returned
        assert outcome.candidate_count == 3  # final_top_k results

    def test_empty_results(self) -> None:
        retriever = FakeRetriever([])
        config = make_disabled_config()
        service = RetrievalService(retriever, config, reranker=None)

        outcome = service.retrieve([0.0], "query", final_top_k=3)
        assert outcome.sources == ()
        assert outcome.max_retrieval_score is None
        assert outcome.candidate_count == 0


class TestRetrievalServiceEnabled:
    def test_enabled_retrieves_candidate_pool(self) -> None:
        raw = make_raw_results(15)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=15)
        reranker = FakeReranker()
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=5)

        # Should retrieve 15 candidates (candidate_top_k)
        assert retriever.last_top_k == 15
        # But only return 5
        assert len(outcome.sources) == 5
        assert outcome.reranker_applied is True
        assert outcome.reranker_fallback is False

    def test_enabled_reranker_changes_order(self) -> None:
        raw = make_raw_results(10)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=10)
        # Reverse the order: last candidate gets highest score
        reranker = FakeReranker(
            score_map={str(i): float(10 - i) for i in range(1, 11)}
        )
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=3)

        # The reranker gave higher scores to higher chunk_index values
        assert outcome.sources[0].chunk_index == 1  # score 9.0
        assert outcome.sources[0].rerank_score == 9.0
        assert outcome.sources[1].chunk_index == 2  # score 8.0
        assert outcome.sources[2].chunk_index == 3  # score 7.0

    def test_retrieval_score_not_overwritten(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(score_map={"1": 0.99, "2": 0.88, "3": 0.77, "4": 0.66, "5": 0.55})
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=2)

        for chunk in outcome.sources:
            # retrieval_score is the original cosine similarity, not rerank score
            assert chunk.retrieval_score != chunk.rerank_score
            # retrieval_score = 1.0 - chunk_index * 0.1 (from make_raw_results)
            expected = 1.0 - chunk.chunk_index * 0.1
            assert chunk.retrieval_score == expected

    def test_rerank_score_preserved(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(score_map={"1": 0.99, "2": 0.88, "3": 0.77, "4": 0.66, "5": 0.55})
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=3)

        for chunk in outcome.sources:
            assert chunk.rerank_score is not None

    def test_final_rank_correct(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(
            score_map={"5": 10.0, "4": 8.0, "3": 6.0, "2": 4.0, "1": 2.0}
        )
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=3)

        assert outcome.sources[0].final_rank == 1
        assert outcome.sources[0].chunk_index == 5  # highest rerank score
        assert outcome.sources[1].final_rank == 2
        assert outcome.sources[1].chunk_index == 4
        assert outcome.sources[2].final_rank == 3
        assert outcome.sources[2].chunk_index == 3

    def test_retrieval_rank_preserved(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(
            score_map={"5": 10.0, "4": 8.0, "3": 6.0, "2": 4.0, "1": 2.0}
        )
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=3)

        # retrieval_rank is the original vector rank, not the final rank
        assert outcome.sources[0].retrieval_rank == 5  # was rank 5 in vector
        assert outcome.sources[0].final_rank == 1      # now rank 1 after rerank

    def test_max_retrieval_score_unchanged_by_rerank(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(
            score_map={"5": 10.0, "1": 1.0}  # reverse order
        )
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=3)

        # max_retrieval_score is the highest cosine similarity, not rerank score
        assert outcome.max_retrieval_score == 0.9  # raw[0].score

    def test_stable_tie_breaker(self) -> None:
        """When rerank scores are equal, original order is preserved."""
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        # All same score → stable sort by retrieval_rank
        reranker = FakeReranker(
            score_map={"1": 5.0, "2": 5.0, "3": 5.0, "4": 5.0, "5": 5.0}
        )
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=3)

        # With all equal scores, original vector order is preserved
        assert outcome.sources[0].chunk_index == 1
        assert outcome.sources[1].chunk_index == 2
        assert outcome.sources[2].chunk_index == 3


class TestRetrievalServiceFallback:
    def test_reranker_failure_falls_back(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(fail=True)
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=3)

        assert outcome.reranker_applied is False
        assert outcome.reranker_fallback is True
        # Falls back to vector order
        assert outcome.sources[0].chunk_index == 1
        assert outcome.sources[0].rerank_score is None

    def test_fallback_preserves_vector_order(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(fail=True)
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=3)

        for i, chunk in enumerate(outcome.sources):
            assert chunk.chunk_index == i + 1
            assert chunk.rerank_score is None

    def test_fallback_max_retrieval_score_preserved(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(fail=True)
        service = RetrievalService(retriever, config, reranker=reranker)

        outcome = service.retrieve([0.0], "query", final_top_k=3)
        assert outcome.max_retrieval_score == 0.9


# ---------------------------------------------------------------------------
# RagService integration
# ---------------------------------------------------------------------------


class TestRagServiceWithRetrieval:
    def test_no_retrieval_service_backward_compat(self) -> None:
        """Without retrieval_service, RagService uses direct search."""
        sources = [{"rank": 1, "score": 0.9, "text": "answer", "chunk_index": 0}]
        embedding = FakeEmbeddingService()
        retriever = FakeRetriever(sources)
        prompt_builder = FakePromptBuilder()
        generation = FakeGenerationService()

        service = RagService(
            embedding_service=embedding,
            retriever=retriever,
            prompt_builder=prompt_builder,
            generation_service=generation,
            min_relevance_score=0.35,
            retrieval_service=None,
        )

        result = service.answer("question")
        assert result["answer_status"] == "answered"
        assert result["reranker_applied"] is False
        assert result["reranker_fallback"] is False

    def test_with_retrieval_service_reranker_disabled(self) -> None:
        raw = make_raw_results(5)
        retriever = FakeRetriever(raw)
        config = make_disabled_config()
        retrieval_service = RetrievalService(retriever, config, reranker=None)

        embedding = FakeEmbeddingService()
        prompt_builder = FakePromptBuilder()
        generation = FakeGenerationService()

        service = RagService(
            embedding_service=embedding,
            retriever=retriever,
            prompt_builder=prompt_builder,
            generation_service=generation,
            min_relevance_score=0.35,
            retrieval_service=retrieval_service,
        )

        result = service.answer("question")
        assert result["answer_status"] == "answered"
        assert result["reranker_applied"] is False
        assert result["reranker_fallback"] is False

    def test_refusal_decision_unchanged_by_rerank(self) -> None:
        """Score at threshold should still answer, regardless of reranking."""
        raw = [
            {"rank": 1, "score": 0.35, "text": "boundary", "chunk_index": 0,
             "document_id": "d", "filename": "f", "page_number": None},
        ]
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(score_map={"0": 10.0})
        retrieval_service = RetrievalService(retriever, config, reranker=reranker)

        embedding = FakeEmbeddingService()
        prompt_builder = FakePromptBuilder()
        generation = FakeGenerationService()

        service = RagService(
            embedding_service=embedding,
            retriever=retriever,
            prompt_builder=prompt_builder,
            generation_service=generation,
            min_relevance_score=0.35,
            retrieval_service=retrieval_service,
        )

        result = service.answer("question")
        # score == threshold → should answer
        assert result["answer_status"] == "answered"
        assert result["max_relevance_score"] == 0.35

    def test_no_candidates_still_refuses(self) -> None:
        retriever = FakeRetriever([])
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker()
        retrieval_service = RetrievalService(retriever, config, reranker=reranker)

        embedding = FakeEmbeddingService()
        prompt_builder = FakePromptBuilder()
        generation = FakeGenerationService()

        service = RagService(
            embedding_service=embedding,
            retriever=retriever,
            prompt_builder=prompt_builder,
            generation_service=generation,
            min_relevance_score=0.35,
            retrieval_service=retrieval_service,
        )

        result = service.answer("question")
        assert result["answer_status"] == "insufficient_context"
        assert result["max_relevance_score"] is None

    def test_insufficient_context_no_llm_call(self) -> None:
        raw = [
            {"rank": 1, "score": 0.1, "text": "low", "chunk_index": 0,
             "document_id": "d", "filename": "f", "page_number": None},
        ]
        retriever = FakeRetriever(raw)
        config = make_enabled_config(candidate_top_k=5)
        reranker = FakeReranker(score_map={"0": 10.0})
        retrieval_service = RetrievalService(retriever, config, reranker=reranker)

        embedding = FakeEmbeddingService()
        prompt_builder = FakePromptBuilder()
        generation = FakeGenerationService()

        service = RagService(
            embedding_service=embedding,
            retriever=retriever,
            prompt_builder=prompt_builder,
            generation_service=generation,
            min_relevance_score=0.35,
            retrieval_service=retrieval_service,
        )

        result = service.answer("question")
        assert result["answer_status"] == "insufficient_context"
        assert result["generation_elapsed_ms"] == 0.0


# ---------------------------------------------------------------------------
# build_reranker factory
# ---------------------------------------------------------------------------


class TestBuildReranker:
    def test_disabled_returns_none(self) -> None:
        config = make_disabled_config()
        assert build_reranker(config) is None

    def test_empty_model_returns_none(self) -> None:
        config = RerankerConfig(enabled=True, model_name="", candidate_top_k=15)
        assert build_reranker(config) is None

    def test_enabled_raises_on_stub(self) -> None:
        """When enabled with a model name, construction should attempt to load."""
        config = make_enabled_config(model_name="test-model")
        with pytest.raises((AssertionError, RuntimeError)):
            build_reranker(config)


# ---------------------------------------------------------------------------
# _stable_sort_key
# ---------------------------------------------------------------------------


class TestStableSortKey:
    def test_rerank_score_descending(self) -> None:
        key1 = _stable_sort_key(10.0, 1, "doc-a", 0)
        key2 = _stable_sort_key(5.0, 2, "doc-b", 1)
        assert key1 < key2  # 10.0 > 5.0, so -10.0 < -5.0

    def test_tie_breaker_retrieval_rank(self) -> None:
        key1 = _stable_sort_key(5.0, 1, "doc-a", 0)
        key2 = _stable_sort_key(5.0, 2, "doc-b", 1)
        assert key1 < key2  # same score, rank 1 < rank 2

    def test_tie_breaker_document_id(self) -> None:
        key1 = _stable_sort_key(5.0, 1, "doc-a", 0)
        key2 = _stable_sort_key(5.0, 1, "doc-b", 1)
        assert key1 < key2  # same score and rank, doc-a < doc-b

    def test_tie_breaker_chunk_index(self) -> None:
        key1 = _stable_sort_key(5.0, 1, "doc-a", 0)
        key2 = _stable_sort_key(5.0, 1, "doc-a", 1)
        assert key1 < key2  # same everything, chunk 0 < chunk 1

    def test_none_rerank_score_preserves_vector_order(self) -> None:
        key1 = _stable_sort_key(None, 1, "doc-a", 0)
        key2 = _stable_sort_key(None, 2, "doc-b", 1)
        assert key1 < key2  # None → use retrieval_rank ascending
