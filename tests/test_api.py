from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api as api_module
import src.rag_service as rag_service_module
from src.exceptions import GenerationConfigurationError, GenerationError


class FakeEmbeddingService:
    init_count = 0
    document_encode_count = 0
    queries: list[str] = []

    def __init__(self) -> None:
        type(self).init_count += 1
        self.model_name = "fake/test-model"

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        type(self).document_encode_count += 1
        return np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))

    def encode_query(self, query: str) -> np.ndarray:
        type(self).queries.append(query)
        return np.array([1.0, 0.0])


class FakeGenerationService:
    init_count = 0
    prompts: list[str] = []

    def __init__(self) -> None:
        type(self).init_count += 1
        self.model_name = "fake/llm-model"

    def generate(self, prompt: str) -> str:
        type(self).prompts.append(prompt)
        return "Service 层负责处理核心业务逻辑。[来源1]"


class FailingGenerationService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def generate(self, _prompt: str) -> str:
        raise self._error


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    FakeEmbeddingService.init_count = 0
    FakeEmbeddingService.document_encode_count = 0
    FakeEmbeddingService.queries = []
    FakeGenerationService.init_count = 0
    FakeGenerationService.prompts = []

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        FakeGenerationService,
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_app_builds_index_once_and_health_reports_chunk_count(
    client: TestClient,
) -> None:
    first_response = client.get("/health")
    second_response = client.get("/health")

    assert first_response.status_code == 200
    assert first_response.json() == {"status": "ok", "chunk_count": 4}
    assert second_response.status_code == 200
    assert FakeEmbeddingService.init_count == 1
    assert FakeEmbeddingService.document_encode_count == 1
    assert FakeGenerationService.init_count == 1


def test_search_uses_default_top_k_and_returns_typed_results(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([10.0, 10.125])
    monkeypatch.setattr(api_module, "perf_counter", lambda: next(times))

    response = client.post(
        "/search",
        json={"query": "  业务逻辑应该写在哪里？  "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "业务逻辑应该写在哪里？"
    assert body["elapsed_ms"] == 125.0
    assert body["indexed_chunks"] == 4
    assert body["model"] == "fake/test-model"
    assert len(body["results"]) == 3
    assert body["results"][0] == {
        "rank": 1,
        "score": 1.0,
        "text": "A" * 300,
        "chunk_index": 0,
    }
    assert FakeEmbeddingService.queries == ["业务逻辑应该写在哪里？"]


def test_search_honors_custom_top_k(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={"query": "repository", "top_k": 1},
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "valid", "top_k": 0},
        {"query": "valid", "top_k": -1},
    ],
)
def test_search_rejects_invalid_request_bodies(
    client: TestClient,
    payload: dict[str, object],
) -> None:
    response = client.post("/search", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("top_k", ["2", 2.0], ids=["string", "float"])
def test_search_rejects_non_integer_top_k(
    client: TestClient,
    top_k: object,
) -> None:
    response = client.post(
        "/search",
        json={"query": "valid", "top_k": top_k},
    )

    assert response.status_code == 422


def test_ask_returns_answer_sources_timings_and_model_names(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([10.0, 10.0105, 10.0106, 10.7308])
    monkeypatch.setattr(rag_service_module, "perf_counter", lambda: next(times))

    response = client.post(
        "/ask",
        json={"question": "  Service 层负责什么？  "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Service 层负责什么？"
    assert body["answer"] == "Service 层负责处理核心业务逻辑。[来源1]"
    assert len(body["sources"]) == 3
    assert body["sources"][0] == {
        "rank": 1,
        "score": 1.0,
        "text": "A" * 300,
        "chunk_index": 0,
    }
    assert body["retrieval_elapsed_ms"] == 10.5
    assert body["generation_elapsed_ms"] == 720.2
    assert body["total_elapsed_ms"] == 730.8
    assert body["embedding_model"] == "fake/test-model"
    assert body["llm_model"] == "fake/llm-model"
    assert FakeEmbeddingService.queries == ["Service 层负责什么？"]
    assert len(FakeGenerationService.prompts) == 1


def test_ask_uses_default_top_k(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "课程问题"})

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 3


def test_ask_honors_custom_top_k(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json={"question": "课程问题", "top_k": 1},
    )

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "valid", "top_k": 0},
        {"question": "valid", "top_k": 11},
        {"question": "valid", "top_k": "2"},
        {"question": "valid", "top_k": 2.0},
        {"question": "x" * 501},
    ],
)
def test_ask_rejects_invalid_request_bodies(
    client: TestClient,
    payload: dict[str, object],
) -> None:
    response = client.post("/ask", json=payload)

    assert response.status_code == 422
    assert FakeGenerationService.prompts == []


def test_ask_converts_generation_error_to_stable_502(
    client: TestClient,
) -> None:
    provider_message = "provider response containing secret data"
    client.app.state.rag_service._generation_service = FailingGenerationService(
        GenerationError(provider_message)
    )

    response = client.post("/ask", json={"question": "课程问题"})

    assert response.status_code == 502
    assert response.json() == {
        "code": "GENERATION_FAILED",
        "message": "回答生成失败，请稍后重试",
    }
    assert provider_message not in response.text


def test_ask_converts_configuration_error_to_stable_503(
    client: TestClient,
) -> None:
    client.app.state.rag_service._generation_service = FailingGenerationService(
        GenerationConfigurationError("missing secret configuration")
    )

    response = client.post("/ask", json={"question": "课程问题"})

    assert response.status_code == 503
    assert response.json() == {
        "code": "LLM_NOT_CONFIGURED",
        "message": "问答服务尚未完成配置",
    }
    assert "secret configuration" not in response.text
