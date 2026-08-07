"""API integration tests against real PostgreSQL with fake models.

The application runs with ``VECTOR_STORE_BACKEND=pgvector`` and connects to
the real database.  FakeEmbeddingService and FakeGenerationService keep the
suite offline: no model is downloaded and no real LLM request is made.
"""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

import src.api as api_module
from conftest import deterministic_vector, integration_database_url


class FakeEmbeddingService:
    model_name = "fake/384"
    instances: list["FakeEmbeddingService"] = []

    def __init__(self) -> None:
        self.document_encode_count = 0
        type(self).instances.append(self)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_encode_count += 1
        return np.stack([deterministic_vector(text) for text in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return deterministic_vector(query)


class FakeGenerationService:
    def __init__(self) -> None:
        self.model_name = "fake/llm"

    def generate(self, prompt: str) -> str:
        return f"集成测试回答。[来源1] prompt={len(prompt)}"

    def close(self) -> None:
        return None


def _cleanup_uploads() -> None:
    """Remove non-builtin documents after each test."""
    engine = create_engine(integration_database_url())
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM documents "
                    "WHERE document_id <> 'builtin-knowledge'"
                )
            )
    finally:
        engine.dispose()


@pytest.fixture
def app_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    FakeEmbeddingService.instances = []
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "pgvector")
    monkeypatch.setenv("DATABASE_URL", integration_database_url())
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.delenv("RAG_RERANKER_ENABLED", raising=False)
    monkeypatch.delenv("RAG_MIN_RELEVANCE_SCORE", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_BYTES", raising=False)

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("内置知识库课程内容。" * 40, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(api_module, "EmbeddingService", FakeEmbeddingService)
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

    with TestClient(api_module.app) as client:
        yield client
    _cleanup_uploads()


UPLOAD_TEXT = "独特集成测试课程内容 alpha beta gamma"


class TestApiIntegration:
    def test_health_ready_and_builtin_present(
        self,
        app_client: TestClient,
    ) -> None:
        health = app_client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["retrieval_ready"] is True
        assert body["chunk_count"] > 0

        documents = app_client.get("/documents").json()
        ids = {record["document_id"] for record in documents["documents"]}
        assert "builtin-knowledge" in ids

    def test_upload_search_ask_delete_flow(
        self,
        app_client: TestClient,
    ) -> None:
        before = app_client.get("/documents").json()["document_count"]

        upload = app_client.post(
            "/documents",
            files={"file": ("it-upload.txt", UPLOAD_TEXT.encode("utf-8"), "text/plain")},
        )
        assert upload.status_code == 201
        uploaded = upload.json()
        assert uploaded["filename"] == "it-upload.txt"
        assert uploaded["is_builtin"] is False
        assert uploaded["index_status"] == "ready"

        after = app_client.get("/documents").json()
        assert after["document_count"] == before + 1
        assert any(
            record["document_id"] == uploaded["document_id"]
            for record in after["documents"]
        )

        search = app_client.post(
            "/search",
            json={"query": UPLOAD_TEXT, "top_k": 3},
        )
        assert search.status_code == 200
        search_body = search.json()
        assert any(
            result["document_id"] == uploaded["document_id"]
            for result in search_body["results"]
        )
        assert search_body["indexed_chunks"] == after["chunk_count"]

        ask = app_client.post(
            "/ask",
            json={"question": UPLOAD_TEXT, "top_k": 3},
        )
        assert ask.status_code == 200
        ask_body = ask.json()
        assert ask_body["answer_status"] == "answered"
        assert any(
            source["document_id"] == uploaded["document_id"]
            for source in ask_body["sources"]
        )
        assert "集成测试回答" in ask_body["answer"]

        delete = app_client.delete(f"/documents/{uploaded['document_id']}")
        assert delete.status_code == 200
        assert delete.json()["deleted"] is True

        after_delete = app_client.get("/documents").json()
        assert after_delete["document_count"] == before

        search_after = app_client.post(
            "/search",
            json={"query": UPLOAD_TEXT, "top_k": 5},
        ).json()
        assert not any(
            result["document_id"] == uploaded["document_id"]
            for result in search_after["results"]
        )
        # The built-in document must still exist.
        ids = {record["document_id"] for record in after_delete["documents"]}
        assert "builtin-knowledge" in ids

    def test_api_field_structure_matches_memory_contract(
        self,
        app_client: TestClient,
    ) -> None:
        app_client.post(
            "/documents",
            files={"file": ("it-structure.txt", UPLOAD_TEXT.encode("utf-8"), "text/plain")},
        )
        search = app_client.post(
            "/search",
            json={"query": UPLOAD_TEXT, "top_k": 3},
        ).json()

        assert set(search.keys()) == {
            "query",
            "elapsed_ms",
            "indexed_chunks",
            "model",
            "results",
            "reranker_applied",
            "reranker_fallback",
        }
        result = search["results"][0]
        assert set(result.keys()) == {
            "rank",
            "score",
            "text",
            "chunk_index",
            "document_id",
            "filename",
            "page_number",
            "retrieval_rank",
            "rerank_score",
            "reranker_applied",
        }
        assert result["document_id"]
        assert isinstance(result["score"], float)
        assert result["page_number"] is None

    def test_restart_does_not_reembed_uploaded_documents(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        FakeEmbeddingService.instances = []
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "pgvector")
        monkeypatch.setenv("DATABASE_URL", integration_database_url())
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("LLM_BASE_URL", "")
        monkeypatch.setenv("LLM_MODEL", "")
        monkeypatch.delenv("RAG_RERANKER_ENABLED", raising=False)
        monkeypatch.delenv("RAG_MIN_RELEVANCE_SCORE", raising=False)
        monkeypatch.delenv("MAX_UPLOAD_BYTES", raising=False)

        knowledge_file = tmp_path / "knowledge.txt"
        knowledge_file.write_text("内置知识库课程内容。" * 40, encoding="utf-8")
        monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
        monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
        monkeypatch.setattr(
            api_module,
            "METADATA_PATH",
            tmp_path / "documents.json",
        )
        monkeypatch.setattr(api_module, "EmbeddingService", FakeEmbeddingService)
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

        document_id: str | None = None
        with TestClient(api_module.app) as first_client:
            first_embedding = FakeEmbeddingService.instances[0]
            upload = first_client.post(
                "/documents",
                files={
                    "file": (
                        "it-restart.txt",
                        UPLOAD_TEXT.encode("utf-8"),
                        "text/plain",
                    )
                },
            )
            assert upload.status_code == 201
            document_id = upload.json()["document_id"]
            # The new upload is embedded exactly once during ingestion.
            assert first_embedding.document_encode_count >= 1

        # Simulate a full restart: a fresh app and a fresh embedding service.
        with TestClient(api_module.app) as second_client:
            second_embedding = FakeEmbeddingService.instances[1]
            documents = second_client.get("/documents").json()
            assert any(
                record["document_id"] == document_id
                for record in documents["documents"]
            )
            search = second_client.post(
                "/search",
                json={"query": UPLOAD_TEXT, "top_k": 3},
            )
            assert search.status_code == 200
            assert any(
                result["document_id"] == document_id
                for result in search.json()["results"]
            )
            # Persisted documents are never re-embedded on restart.
            assert second_embedding.document_encode_count == 0

    def test_builtin_not_reembedded_on_restart(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        FakeEmbeddingService.instances = []
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "pgvector")
        monkeypatch.setenv("DATABASE_URL", integration_database_url())
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("LLM_BASE_URL", "")
        monkeypatch.setenv("LLM_MODEL", "")
        monkeypatch.delenv("RAG_RERANKER_ENABLED", raising=False)
        monkeypatch.delenv("RAG_MIN_RELEVANCE_SCORE", raising=False)
        monkeypatch.delenv("MAX_UPLOAD_BYTES", raising=False)

        knowledge_file = tmp_path / "knowledge.txt"
        knowledge_file.write_text("内置知识库课程内容。" * 40, encoding="utf-8")
        monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
        monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
        monkeypatch.setattr(
            api_module,
            "METADATA_PATH",
            tmp_path / "documents.json",
        )
        monkeypatch.setattr(api_module, "EmbeddingService", FakeEmbeddingService)
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

        with TestClient(api_module.app) as _first:
            pass  # startup ensures the built-in document exists
        with TestClient(api_module.app) as second:
            second_embedding = FakeEmbeddingService.instances[-1]
            documents = second.get("/documents").json()
            assert any(
                record["document_id"] == "builtin-knowledge"
                for record in documents["documents"]
            )
            # The persisted built-in document is reused without re-embedding.
            assert second_embedding.document_encode_count == 0
