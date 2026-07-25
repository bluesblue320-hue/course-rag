from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api as api_module


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


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    FakeEmbeddingService.init_count = 0
    FakeEmbeddingService.document_encode_count = 0
    FakeEmbeddingService.queries = []

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
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
