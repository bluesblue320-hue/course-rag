from copy import deepcopy
from typing import Any

import pytest

import src.rag_service as rag_service_module
from src.exceptions import GenerationError, RagConfigurationError
from src.rag_service import (
    DEFAULT_MIN_RELEVANCE_SCORE,
    INSUFFICIENT_CONTEXT_ANSWER,
    RagService,
    has_sufficient_context,
    resolve_min_relevance_score,
)


class FakeEmbeddingService:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.questions: list[str] = []

    def encode_query(self, question: str) -> str:
        self.calls.append("embedding")
        self.questions.append(question)
        return "query-vector"


class FakeRetriever:
    def __init__(
        self,
        calls: list[str],
        sources: list[dict[str, object]],
    ) -> None:
        self.calls = calls
        self.sources = sources
        self.requests: list[tuple[object, int]] = []

    def search(self, query_embedding: object, top_k: int) -> list[dict[str, object]]:
        self.calls.append("retriever")
        self.requests.append((query_embedding, top_k))
        return self.sources


class FakePromptBuilder:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.requests: list[tuple[str, list[dict[str, object]]]] = []

    def build(self, question: str, sources: Any) -> str:
        self.calls.append("prompt_builder")
        self.requests.append((question, sources))
        return "built prompt"


class FakeGenerationService:
    def __init__(
        self,
        calls: list[str],
        error: Exception | None = None,
    ) -> None:
        self.calls = calls
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append("generation")
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return "generated answer"


def make_pipeline(
    sources: list[dict[str, object]] | None = None,
    generation_error: Exception | None = None,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
) -> tuple[
    RagService,
    FakeEmbeddingService,
    FakeRetriever,
    FakePromptBuilder,
    FakeGenerationService,
    list[str],
]:
    calls: list[str] = []
    source_results = sources if sources is not None else [
        {
            "rank": 1,
            "score": 0.82,
            "text": "Service 层负责业务逻辑。",
            "chunk_index": 2,
        }
    ]
    embedding = FakeEmbeddingService(calls)
    retriever = FakeRetriever(calls, source_results)
    prompt_builder = FakePromptBuilder(calls)
    generation = FakeGenerationService(calls, generation_error)
    service = RagService(
        embedding,
        retriever,
        prompt_builder,
        generation,
        min_relevance_score=min_relevance_score,
    )
    return service, embedding, retriever, prompt_builder, generation, calls


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.0, 10.0105, 10.0106, 10.7308])
    monkeypatch.setattr(rag_service_module, "perf_counter", lambda: next(times))


def test_answer_returns_generated_answer_and_sources() -> None:
    service, _, _, _, _, _ = make_pipeline()

    result = service.answer("课程问题")

    assert result["answer"] == "generated answer"
    assert result["answer_status"] == "answered"
    assert result["max_relevance_score"] == 0.82
    assert result["relevance_threshold"] == 0.35
    assert result["sources"][0]["text"] == "Service 层负责业务逻辑。"


def test_question_is_cleaned() -> None:
    service, _, _, _, _, _ = make_pipeline()

    result = service.answer("  课程问题  ")

    assert result["question"] == "课程问题"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_is_rejected(question: str) -> None:
    service, _, _, _, _, _ = make_pipeline()

    with pytest.raises(ValueError, match="question 不能为空"):
        service.answer(question)


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_is_rejected(top_k: int) -> None:
    service, _, _, _, _, _ = make_pipeline()

    with pytest.raises(ValueError, match="top_k 必须大于 0"):
        service.answer("课程问题", top_k=top_k)


def test_top_k_is_passed_to_retriever() -> None:
    service, _, retriever, _, _, _ = make_pipeline()

    service.answer("课程问题", top_k=5)

    assert retriever.requests == [("query-vector", 5)]


def test_embedding_service_receives_cleaned_question() -> None:
    service, embedding, _, _, _, _ = make_pipeline()

    service.answer("  课程问题  ")

    assert embedding.questions == ["课程问题"]


def test_prompt_builder_receives_question_and_original_sources() -> None:
    sources = [{"rank": 1, "score": 0.9, "text": "资料", "chunk_index": 0}]
    service, _, _, prompt_builder, _, _ = make_pipeline(sources)

    service.answer("  课程问题  ")

    assert prompt_builder.requests == [("课程问题", sources)]
    assert prompt_builder.requests[0][1] is sources


def test_generation_service_receives_built_prompt() -> None:
    service, _, _, _, generation, _ = make_pipeline()

    service.answer("课程问题")

    assert generation.prompts == ["built prompt"]


def test_pipeline_calls_dependencies_in_order() -> None:
    service, _, _, _, _, calls = make_pipeline()

    service.answer("课程问题")

    assert calls == ["embedding", "retriever", "prompt_builder", "generation"]


def test_sources_content_and_order_are_not_modified() -> None:
    sources = [
        {"rank": 1, "score": 0.9, "text": "第一段", "chunk_index": 3},
        {"rank": 2, "score": 0.8, "text": "第二段", "chunk_index": 1},
    ]
    original_sources = deepcopy(sources)
    service, _, _, _, _, _ = make_pipeline(sources)

    result = service.answer("课程问题")

    assert sources == original_sources
    assert result["sources"] == original_sources


def test_answer_returns_rounded_timings() -> None:
    service, _, _, _, _, _ = make_pipeline()

    result = service.answer("课程问题")

    assert result["retrieval_elapsed_ms"] == 10.5
    assert result["generation_elapsed_ms"] == 720.2
    assert result["total_elapsed_ms"] == 730.8


def test_generation_error_is_propagated() -> None:
    error = GenerationError("LLM 服务调用失败")
    service, _, _, _, _, _ = make_pipeline(generation_error=error)

    with pytest.raises(GenerationError) as caught:
        service.answer("课程问题")

    assert caught.value is error


def test_empty_sources_return_insufficient_context_without_generation() -> None:
    service, _, _, prompt_builder, generation, calls = make_pipeline(sources=[])

    result = service.answer("课程问题")

    assert prompt_builder.requests == []
    assert generation.prompts == []
    assert calls == ["embedding", "retriever"]
    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert result["answer_status"] == "insufficient_context"
    assert result["sources"] == []
    assert result["max_relevance_score"] is None
    assert result["generation_elapsed_ms"] == 0.0


def test_score_equal_to_threshold_is_answered() -> None:
    sources = [{"rank": 1, "score": 0.35, "text": "资料", "chunk_index": 0}]
    service, _, _, prompt_builder, generation, _ = make_pipeline(sources)

    result = service.answer("课程问题")

    assert result["answer_status"] == "answered"
    assert prompt_builder.requests
    assert generation.prompts == ["built prompt"]


def test_low_score_returns_candidates_without_prompt_or_generation() -> None:
    sources = [{"rank": 1, "score": 0.34, "text": "候选资料", "chunk_index": 0}]
    service, _, _, prompt_builder, generation, calls = make_pipeline(sources)

    result = service.answer("课程问题")

    assert result["answer_status"] == "insufficient_context"
    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert result["sources"] is sources
    assert result["max_relevance_score"] == 0.34
    assert result["relevance_threshold"] == 0.35
    assert result["generation_elapsed_ms"] == 0.0
    assert prompt_builder.requests == []
    assert generation.prompts == []
    assert calls == ["embedding", "retriever"]


@pytest.mark.parametrize("score", [None, "0.9", True, -1.1, 1.1, float("nan"), float("inf")])
def test_invalid_top_score_is_rejected(score: object) -> None:
    sources = [{"rank": 1, "score": score, "text": "资料", "chunk_index": 0}]
    service, _, _, prompt_builder, generation, _ = make_pipeline(sources)

    with pytest.raises(ValueError, match="最高相关性分数无效"):
        service.answer("课程问题")

    assert prompt_builder.requests == []
    assert generation.prompts == []


def test_threshold_defaults_when_environment_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_MIN_RELEVANCE_SCORE", raising=False)
    assert resolve_min_relevance_score() == 0.35


def test_explicit_threshold_takes_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MIN_RELEVANCE_SCORE", "0.9")
    assert resolve_min_relevance_score(0.6) == 0.6


@pytest.mark.parametrize("value", [0, 1, 0.42])
def test_valid_threshold_values_are_accepted(value: float) -> None:
    assert resolve_min_relevance_score(value) == value


@pytest.mark.parametrize("value", ["bad", "nan", "inf", "-0.1", "1.1"])
def test_invalid_environment_threshold_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MIN_RELEVANCE_SCORE", value)
    with pytest.raises(RagConfigurationError):
        resolve_min_relevance_score()


@pytest.mark.parametrize("value", [True, False, -0.1, 1.1, float("nan")])
def test_constructor_rejects_invalid_threshold(value: object) -> None:
    with pytest.raises(RagConfigurationError):
        make_pipeline(min_relevance_score=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, False])
def test_explicit_bool_threshold_is_rejected(value: bool) -> None:
    with pytest.raises(RagConfigurationError):
        resolve_min_relevance_score(value)


def test_has_sufficient_context_answers_at_threshold_boundary() -> None:
    assert has_sufficient_context(0.35, 0.35) is True
    assert has_sufficient_context(0.3500001, 0.35) is True


def test_has_sufficient_context_refuses_below_threshold() -> None:
    assert has_sufficient_context(0.3499, 0.35) is False


def test_has_sufficient_context_refuses_missing_score() -> None:
    assert has_sufficient_context(None, 0.35) is False


def test_has_sufficient_context_refuses_non_finite_scores() -> None:
    assert has_sufficient_context(float("nan"), 0.35) is False
    assert has_sufficient_context(float("inf"), 0.35) is False
    assert has_sufficient_context(float("-inf"), 0.35) is False


def test_has_sufficient_context_matches_production_answer_behavior() -> None:
    service, _, _, prompt_builder, generation, _ = make_pipeline(
        sources=[{"rank": 1, "score": 0.6, "text": "资料", "chunk_index": 0}]
    )

    result = service.answer("课程问题")

    assert result["answer_status"] == "answered"
    assert prompt_builder.requests
    assert generation.prompts == ["built prompt"]
