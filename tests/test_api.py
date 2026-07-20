import importlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


def test_app_builds_index_once_and_health_reports_chunk_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_module = importlib.import_module("src.api")

    class FakeEmbeddingService:
        init_count = 0
        document_encode_count = 0

        def __init__(self) -> None:
            type(self).init_count += 1

        def encode_documents(self, texts: list[str]) -> np.ndarray:
            type(self).document_encode_count += 1
            return np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))

        def encode_query(self, query: str) -> np.ndarray:
            return np.array([1.0, 0.0])

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )

    with TestClient(api_module.app) as client:
        first_response = client.get("/health")
        second_response = client.get("/health")

    assert first_response.status_code == 200
    assert first_response.json() == {"status": "ok", "chunk_count": 4}
    assert second_response.status_code == 200
    assert FakeEmbeddingService.init_count == 1
    assert FakeEmbeddingService.document_encode_count == 1
