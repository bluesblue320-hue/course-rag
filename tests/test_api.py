from collections.abc import Iterator
from contextlib import contextmanager
import os
import json
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
    close_count = 0
    prompts: list[str] = []

    def __init__(self) -> None:
        type(self).init_count += 1
        self.model_name = "fake/llm-model"

    def generate(self, prompt: str) -> str:
        type(self).prompts.append(prompt)
        return "Service 层负责处理核心业务逻辑。[来源1]"

    def close(self) -> None:
        type(self).close_count += 1


class MisconfiguredGenerationService:
    init_count = 0
    generate_count = 0

    def __init__(self) -> None:
        type(self).init_count += 1
        raise GenerationConfigurationError("缺少 LLM API Key 配置")

    def generate(self, _prompt: str) -> str:
        type(self).generate_count += 1
        return "不应生成答案"


class UnexpectedlyFailingGenerationService:
    def __init__(self) -> None:
        raise RuntimeError("unexpected generation bug")


class FailingGenerationService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def generate(self, _prompt: str) -> str:
        raise self._error

@pytest.fixture(autouse=True)
def clear_rag_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_MIN_RELEVANCE_SCORE", raising=False)



@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    FakeEmbeddingService.init_count = 0
    FakeEmbeddingService.document_encode_count = 0
    FakeEmbeddingService.queries = []
    FakeGenerationService.init_count = 0
    FakeGenerationService.close_count = 0
    FakeGenerationService.prompts = []

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
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
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_app_builds_index_once_and_health_reports_chunk_count(
    client: TestClient,
) -> None:
    first_response = client.get("/health")
    second_response = client.get("/health")

    assert first_response.status_code == 200
    assert first_response.json() == {
        "status": "ok",
        "chunk_count": 4,
        "retrieval_ready": True,
        "generation_ready": True,
        "rag_ready": True,
        "min_relevance_score": 0.35,
        "reranker_enabled": False,
        "reranker_ready": False,
        "reranker_model": None,
        "reranker_status": "disabled",
    }
    assert second_response.status_code == 200
    assert FakeEmbeddingService.init_count == 1
    assert FakeEmbeddingService.document_encode_count == 1
    assert FakeGenerationService.init_count == 1


def test_app_starts_without_llm_and_keeps_retrieval_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeEmbeddingService.init_count = 0
    FakeEmbeddingService.document_encode_count = 0
    FakeEmbeddingService.queries = []
    MisconfiguredGenerationService.init_count = 0
    MisconfiguredGenerationService.generate_count = 0

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        MisconfiguredGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as test_client:
        health_response = test_client.get("/health")
        search_response = test_client.post(
            "/search",
            json={"query": "课程问题"},
        )
        ask_response = test_client.post(
            "/ask",
            json={"question": "课程问题"},
        )

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "ok",
        "chunk_count": 4,
        "retrieval_ready": True,
        "generation_ready": False,
        "rag_ready": False,
        "min_relevance_score": None,
        "reranker_enabled": False,
        "reranker_ready": False,
        "reranker_model": None,
        "reranker_status": "disabled",
    }
    assert search_response.status_code == 200
    assert ask_response.status_code == 503
    assert ask_response.json() == {
        "code": "LLM_NOT_CONFIGURED",
        "message": "问答服务尚未完成配置",
    }
    assert "缺少 LLM API Key 配置" not in ask_response.text
    assert FakeEmbeddingService.init_count == 1
    assert FakeEmbeddingService.document_encode_count == 1
    assert MisconfiguredGenerationService.init_count == 1
    assert MisconfiguredGenerationService.generate_count == 0


def test_loads_project_dotenv_before_generation_without_overriding_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    dotenv_calls: list[tuple[Path, bool]] = []

    class EnvironmentRecordingGenerationService:
        observed_api_key = ""

        def __init__(self) -> None:
            events.append("generation")
            type(self).observed_api_key = os.environ["LLM_API_KEY"]
            self.model_name = "fake/llm-model"

        def close(self) -> None:
            return None

    def fake_load_dotenv(*, dotenv_path: Path, override: bool) -> bool:
        events.append("dotenv")
        dotenv_calls.append((dotenv_path, override))
        if override or "LLM_API_KEY" not in os.environ:
            os.environ["LLM_API_KEY"] = "dotenv-key"
        return True

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "process-key")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        EnvironmentRecordingGenerationService,
    )
    monkeypatch.setattr(api_module, "load_dotenv", fake_load_dotenv)

    with TestClient(api_module.app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert events == ["dotenv", "generation"]
    assert dotenv_calls == [(api_module.PROJECT_ROOT / ".env", False)]
    assert EnvironmentRecordingGenerationService.observed_api_key == (
        "process-key"
    )


def test_unexpected_generation_initialization_error_still_fails_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        UnexpectedlyFailingGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with pytest.raises(RuntimeError, match="unexpected generation bug"):
        with TestClient(api_module.app):
            pass


def test_lifespan_closes_configured_generation_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeGenerationService.close_count = 0
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
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
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as test_client:
        assert test_client.get("/health").status_code == 200
        assert FakeGenerationService.close_count == 0

    assert FakeGenerationService.close_count == 1


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
        "document_id": "builtin-knowledge",
        "filename": "knowledge.txt",
        "page_number": None,
        "retrieval_rank": None,
        "rerank_score": None,
        "reranker_applied": False,
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
    assert body["answer_status"] == "answered"
    assert body["max_relevance_score"] == 1.0
    assert body["relevance_threshold"] == 0.35
    assert len(body["sources"]) == 3
    assert body["sources"][0] == {
        "rank": 1,
        "score": 1.0,
        "text": "A" * 300,
        "chunk_index": 0,
        "document_id": "builtin-knowledge",
        "filename": "knowledge.txt",
        "page_number": None,
        "retrieval_rank": None,
        "rerank_score": None,
        "reranker_applied": False,
    }
    assert body["retrieval_elapsed_ms"] == 10.5
    assert body["generation_elapsed_ms"] == 720.2
    assert body["total_elapsed_ms"] == 730.8
    assert body["embedding_model"] == "fake/test-model"
    assert body["llm_model"] == "fake/llm-model"
    assert FakeEmbeddingService.queries == ["Service 层负责什么？"]
    assert len(FakeGenerationService.prompts) == 1


def test_ask_returns_structured_insufficient_context_without_generation(
    client: TestClient,
) -> None:
    candidate = {
        "rank": 1,
        "score": 0.1,
        "text": "最接近但仍不相关的课程资料",
        "chunk_index": 3,
        "document_id": "builtin-knowledge",
        "filename": "knowledge.txt",
        "page_number": None,
        "retrieval_rank": None,
        "rerank_score": None,
        "reranker_applied": False,
    }

    class LowScoreRetriever:
        def search(self, _embedding: object, top_k: int) -> list[dict[str, object]]:
            assert top_k == 3
            return [candidate]

    client.app.state.rag_service._retriever = LowScoreRetriever()

    response = client.post("/ask", json={"question": "今天天气怎么样？"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer_status"] == "insufficient_context"
    assert body["max_relevance_score"] == 0.1
    assert body["relevance_threshold"] == 0.35
    assert body["generation_elapsed_ms"] == 0.0
    assert body["sources"] == [candidate]
    assert "足够信息" in body["answer"]
    assert FakeGenerationService.prompts == []


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


def test_invalid_rag_configuration_keeps_search_available_and_closes_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeGenerationService.close_count = 0
    FakeGenerationService.prompts = []
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setenv("RAG_MIN_RELEVANCE_SCORE", "not-a-secret-value")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(api_module, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(api_module, "GenerationService", FakeGenerationService)
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as test_client:
        health_response = test_client.get("/health")
        search_response = test_client.post("/search", json={"query": "课程问题"})
        ask_response = test_client.post("/ask", json={"question": "课程问题"})

        assert health_response.json() == {
            "status": "ok",
            "chunk_count": 4,
            "retrieval_ready": True,
            "generation_ready": True,
            "rag_ready": False,
            "min_relevance_score": None,
            "reranker_enabled": False,
            "reranker_ready": False,
            "reranker_model": None,
            "reranker_status": "disabled",
        }
        assert search_response.status_code == 200
        assert ask_response.status_code == 503
        assert ask_response.json() == {
            "code": "RAG_NOT_CONFIGURED",
            "message": "问答相关性配置无效",
        }
        assert "not-a-secret-value" not in ask_response.text
        assert FakeGenerationService.close_count == 0

    assert FakeGenerationService.close_count == 1


def test_unexpected_rag_initialization_error_closes_generation_and_fails_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeGenerationService.close_count = 0

    class UnexpectedlyFailingRagService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("unexpected rag bug")

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(api_module, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(api_module, "GenerationService", FakeGenerationService)
    monkeypatch.setattr(api_module, "RagService", UnexpectedlyFailingRagService)
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with pytest.raises(RuntimeError, match="unexpected rag bug"):
        with TestClient(api_module.app):
            pass

    assert FakeGenerationService.close_count == 1


def test_dotenv_and_generation_are_initialized_before_threshold_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(api_module, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(api_module, "GenerationService", FakeGenerationService)
    monkeypatch.setattr(api_module, "load_dotenv", lambda **kwargs: events.append("dotenv"))
    monkeypatch.setattr(
        api_module,
        "resolve_min_relevance_score",
        lambda: events.append("threshold") or 0.35,
    )

    original_init = FakeGenerationService.__init__
    monkeypatch.setattr(FakeGenerationService, "__init__", lambda self: (events.append("generation"), original_init(self))[1])

    with TestClient(api_module.app):
        pass

    assert events == ["dotenv", "generation", "threshold"]


# ---------------------------------------------------------------------------
# Reranker status in /health (Fix 6)
# ---------------------------------------------------------------------------


@contextmanager
def _build_reranker_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    env: dict[str, str],
    build_reranker_override: object | None = None,
) -> "Iterator[TestClient]":  # type: ignore[name-defined]
    FakeEmbeddingService.init_count = 0
    FakeEmbeddingService.document_encode_count = 0
    FakeEmbeddingService.queries = []
    FakeGenerationService.init_count = 0
    FakeGenerationService.close_count = 0
    FakeGenerationService.prompts = []

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")

    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(api_module, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(api_module, "GenerationService", FakeGenerationService)
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    if build_reranker_override is not None:
        monkeypatch.setattr(
            api_module, "build_reranker", build_reranker_override
        )

    with TestClient(api_module.app) as test_client:
        yield test_client


class TestRerankerHealthStatus:
    def test_disabled_status(self, tmp_path, monkeypatch) -> None:
        with _build_reranker_client(
            tmp_path, monkeypatch, env={"RAG_RERANKER_ENABLED": "false"}
        ) as client:
            body = client.get("/health").json()
        assert body["reranker_enabled"] is False
        assert body["reranker_ready"] is False
        assert body["reranker_status"] == "disabled"
        assert body["reranker_model"] is None

    def test_enabled_ready_status(self, tmp_path, monkeypatch) -> None:
        from src.reranker import FakeReranker

        def fake_build(config):
            return FakeReranker()

        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": "local-zh-model",
            },
            build_reranker_override=fake_build,
        ) as client:
            body = client.get("/health").json()
            search = client.post("/search", json={"query": "课程问题"})
            ask = client.post("/ask", json={"question": "课程问题"})

        assert body["reranker_enabled"] is True
        assert body["reranker_ready"] is True
        assert body["reranker_status"] == "ready"
        assert body["reranker_model"] == "local-zh-model"
        assert search.status_code == 200
        assert ask.status_code == 200

    def test_enabled_load_failed_status(self, tmp_path, monkeypatch) -> None:
        def failing_build(config):
            raise RuntimeError(
                "model not found at C:\\secret\\cache\\local-zh-model"
            )

        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": "local-zh-model",
            },
            build_reranker_override=failing_build,
        ) as client:
            body = client.get("/health").json()
            search = client.post("/search", json={"query": "课程问题"})
            ask = client.post("/ask", json={"question": "课程问题"})

        # Enabled but model load failed -> load_failed, not disabled.
        assert body["reranker_enabled"] is True
        assert body["reranker_ready"] is False
        assert body["reranker_status"] == "load_failed"
        assert body["reranker_model"] == "local-zh-model"
        # No internal path / stack must leak into the health payload.
        assert "C:" not in json.dumps(body, ensure_ascii=False)
        assert "Traceback" not in json.dumps(body, ensure_ascii=False)
        # Search and ask must still work via vector-only.
        assert search.status_code == 200
        assert ask.status_code == 200

    def test_config_invalid_status(self, tmp_path, monkeypatch) -> None:
        # Enabled but empty model name -> config resolution fails.
        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": "",
            },
        ) as client:
            body = client.get("/health").json()

        # The user did request enablement, so enabled stays true; only the
        # model configuration is invalid.
        assert body["reranker_enabled"] is True
        assert body["reranker_status"] == "config_invalid"
        assert body["reranker_ready"] is False
        assert body["reranker_model"] is None

    def test_invalid_enable_flag_status(self, tmp_path, monkeypatch) -> None:
        # RAG_RERANKER_ENABLED=maybe cannot be parsed: the system cannot
        # confirm a request to enable, so enabled stays false with
        # config_invalid.
        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={"RAG_RERANKER_ENABLED": "maybe"},
        ) as client:
            body = client.get("/health").json()

        assert body["reranker_enabled"] is False
        assert body["reranker_ready"] is False
        assert body["reranker_status"] == "config_invalid"
        assert body["reranker_model"] is None

    def test_health_redacts_local_model_path(self, tmp_path, monkeypatch) -> None:
        local_path = "C:\\Users\\secret-user\\.cache\\models\\my-reranker"

        def fake_build(config):
            return FakeReranker()

        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": local_path,
            },
            build_reranker_override=fake_build,
        ) as client:
            body = client.get("/health").json()

        payload = json.dumps(body, ensure_ascii=False)
        assert body["reranker_model"] == "<local-model>"
        assert "secret-user" not in payload
        assert "C:" not in payload
        assert ".cache" not in payload

    def test_health_keeps_huggingface_model_id(self, tmp_path, monkeypatch) -> None:
        from src.reranker import FakeReranker

        def fake_build(config):
            return FakeReranker()

        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            },
            build_reranker_override=fake_build,
        ) as client:
            body = client.get("/health").json()

        assert body["reranker_model"] == "BAAI/bge-reranker-v2-m3"
        assert body["reranker_status"] == "ready"


class TestRequestTimeFallbackVisibility:
    """/search and /ask top-level reranker_applied / reranker_fallback."""

    def test_search_default_disabled(self, tmp_path, monkeypatch) -> None:
        with _build_reranker_client(
            tmp_path, monkeypatch, env={"RAG_RERANKER_ENABLED": "false"}
        ) as client:
            body = client.post("/search", json={"query": "课程问题"}).json()

        assert body["reranker_applied"] is False
        assert body["reranker_fallback"] is False

    def test_search_reranker_applied(self, tmp_path, monkeypatch) -> None:
        from src.reranker import FakeReranker

        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": "local-zh-model",
            },
            build_reranker_override=lambda config: FakeReranker(),
        ) as client:
            body = client.post("/search", json={"query": "课程问题"}).json()

        assert body["reranker_applied"] is True
        assert body["reranker_fallback"] is False
        assert body["results"]

    def test_search_request_time_fallback(self, tmp_path, monkeypatch) -> None:
        from src.reranker import FakeReranker

        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": "local-zh-model",
            },
            build_reranker_override=lambda config: FakeReranker(fail=True),
        ) as client:
            response = client.post("/search", json={"query": "课程问题"})
            body = response.json()

        assert response.status_code == 200
        assert body["reranker_applied"] is False
        assert body["reranker_fallback"] is True
        # Fallback preserves the original vector order and still returns results.
        assert body["results"]

    def test_ask_default_disabled(self, tmp_path, monkeypatch) -> None:
        with _build_reranker_client(
            tmp_path, monkeypatch, env={"RAG_RERANKER_ENABLED": "false"}
        ) as client:
            body = client.post("/ask", json={"question": "课程问题"}).json()

        assert body["reranker_applied"] is False
        assert body["reranker_fallback"] is False

    def test_ask_reranker_applied(self, tmp_path, monkeypatch) -> None:
        from src.reranker import FakeReranker

        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": "local-zh-model",
            },
            build_reranker_override=lambda config: FakeReranker(),
        ) as client:
            body = client.post("/ask", json={"question": "课程问题"}).json()

        assert body["reranker_applied"] is True
        assert body["reranker_fallback"] is False
        assert body["answer_status"] == "answered"

    def test_ask_request_time_fallback(self, tmp_path, monkeypatch) -> None:
        from src.reranker import FakeReranker

        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": "local-zh-model",
            },
            build_reranker_override=lambda config: FakeReranker(fail=True),
        ) as client:
            response = client.post("/ask", json={"question": "课程问题"})
            body = response.json()

        assert response.status_code == 200
        assert body["reranker_applied"] is False
        assert body["reranker_fallback"] is True
        # Fallback does not add extra LLM calls: still exactly one generation.
        assert body["answer_status"] == "answered"

    def test_applied_and_fallback_never_both_true(self, tmp_path, monkeypatch) -> None:
        from src.reranker import FakeReranker

        with _build_reranker_client(
            tmp_path,
            monkeypatch,
            env={
                "RAG_RERANKER_ENABLED": "true",
                "RAG_RERANKER_MODEL": "local-zh-model",
            },
            build_reranker_override=lambda config: FakeReranker(),
        ) as client:
            search_body = client.post(
                "/search", json={"query": "课程问题"}
            ).json()
            ask_body = client.post("/ask", json={"question": "课程问题"}).json()

        assert not (search_body["reranker_applied"] and search_body["reranker_fallback"])
        assert not (ask_body["reranker_applied"] and ask_body["reranker_fallback"])
