from copy import deepcopy
from typing import Any

import pytest

import src.rag_service as rag_service_module
from src.exceptions import GenerationError
from src.rag_service import RagService


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
    service = RagService(embedding, retriever, prompt_builder, generation)
    return service, embedding, retriever, prompt_builder, generation, calls


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.0, 10.0105, 10.0106, 10.7308])
    monkeypatch.setattr(rag_service_module, "perf_counter", lambda: next(times))


def test_answer_returns_generated_answer_and_sources() -> None:
    service, _, _, _, _, _ = make_pipeline()

    result = service.answer("课程问题")

    assert result["answer"] == "generated answer"
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


def test_empty_sources_are_still_passed_to_prompt_builder() -> None:
    service, _, _, prompt_builder, _, _ = make_pipeline(sources=[])

    result = service.answer("课程问题")

    assert prompt_builder.requests == [("课程问题", [])]
    assert result["sources"] == []
