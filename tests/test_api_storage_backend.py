"""API-level storage backend tests using injected fake runtimes.

Each test configures its storage backend *before* entering the TestClient
context so the lifespan startup observes the intended environment.
"""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.documents import DocumentRecord, utc_now_iso
from src.exceptions import (
    DatabaseConnectionError,
    DatabaseSchemaError,
    GenerationConfigurationError,
)
from src.storage.runtime import StorageRuntime


class FakeEmbeddingService:
    model_name = "fake/test-model"

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.full((len(texts), 384), 1.0 / np.sqrt(384))

    def encode_query(self, _query: str) -> np.ndarray:
        return np.full(384, 1.0 / np.sqrt(384))


class MisconfiguredGenerationService:
    def __init__(self) -> None:
        raise GenerationConfigurationError("缺少 LLM API Key 配置")

    def generate(self, _prompt: str) -> str:
        return "不应生成答案"

    def close(self) -> None:
        return None


class FakeRetriever:
    def __init__(self) -> None:
        self.search_calls = 0

    @property
    def chunk_count(self) -> int:
        return 1

    def search(
        self,
        _query_embedding: object,
        top_k: int,
    ) -> list[dict[str, object]]:
        self.search_calls += 1
        return [
            {
                "rank": 1,
                "score": 0.9,
                "text": "fake hit",
                "chunk_index": 0,
                "document_id": "doc-1",
                "filename": "notes.txt",
                "page_number": None,
            }
            for _ in range(top_k)
        ]

    def close(self) -> None:
        return None


class FakeDocumentManager:
    def __init__(self, documents: list[DocumentRecord] | None = None) -> None:
        self._documents = list(documents or [])
        self.ingest_count = 0
        self.delete_count = 0

    @property
    def max_upload_bytes(self) -> int:
        return 1024 * 1024

    @property
    def chunk_count(self) -> int:
        return sum(record.chunk_count for record in self._documents)

    def list_documents(self) -> list[DocumentRecord]:
        return list(self._documents)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return next(
            (
                record
                for record in self._documents
                if record.document_id == document_id
            ),
            None,
        )

    def ingest(
        self,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> DocumentRecord:
        self.ingest_count += 1
        record = DocumentRecord(
            document_id=f"uploaded-{self.ingest_count}",
            original_filename=filename,
            stored_filename="a" * 32 + ".txt",
            content_type=content_type or "application/octet-stream",
            size_bytes=len(data),
            text_length=len(data),
            chunk_count=1,
            created_at=utc_now_iso(),
            is_builtin=False,
        )
        self._documents.append(record)
        return record

    def delete_document(self, document_id: str) -> DocumentRecord:
        self.delete_count += 1
        record = self.get_document(document_id)
        self._documents = [
            candidate
            for candidate in self._documents
            if candidate.document_id != document_id
        ]
        assert record is not None
        return record

    def close(self) -> None:
        return None


def make_record(document_id: str, is_builtin: bool = False) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename="knowledge.txt" if is_builtin else "notes.txt",
        stored_filename=None if is_builtin else "b" * 32 + ".txt",
        content_type="text/plain",
        size_bytes=10,
        text_length=10,
        chunk_count=1,
        created_at=utc_now_iso(),
        is_builtin=is_builtin,
    )


def fake_pgvector_runtime(
    manager: FakeDocumentManager | None = None,
    retriever: FakeRetriever | None = None,
) -> StorageRuntime:
    return StorageRuntime(
        backend="pgvector",
        document_manager=manager or FakeDocumentManager(),
        retriever=retriever or FakeRetriever(),
        close_callback=lambda: None,
    )


def setup_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Apply the shared offline fakes before TestClient starts."""
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
    monkeypatch.delenv("RAG_RERANKER_ENABLED", raising=False)
    monkeypatch.delenv("RAG_RERANKER_MODEL", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_BYTES", raising=False)
    monkeypatch.delenv("RAG_MIN_RELEVANCE_SCORE", raising=False)


def enable_pgvector(
    monkeypatch: pytest.MonkeyPatch,
    *,
    database_url: str = "postgresql+psycopg://user:pass@host:5432/db",
    runtime_builder: object | None = None,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "pgvector")
    monkeypatch.setenv("DATABASE_URL", database_url)
    if runtime_builder is not None:
        monkeypatch.setattr(api_module, "build_pgvector_runtime", runtime_builder)


class TestMemoryContract:
    def test_memory_is_default_and_healthy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)
        monkeypatch.delenv("VECTOR_STORE_BACKEND", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with TestClient(api_module.app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "ok"
            assert body["retrieval_ready"] is True
            assert client.app.state.storage_backend == "memory"
            assert client.app.state.knowledge_index is not None

    def test_memory_ignores_database_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)
        monkeypatch.delenv("VECTOR_STORE_BACKEND", raising=False)
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg://user:secret@host:5432/db",
        )

        with TestClient(api_module.app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            assert response.json()["status"] == "ok"
            assert client.app.state.storage_backend == "memory"


class TestPgvectorHealthy:
    def test_search_uses_generic_retriever(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)
        retriever = FakeRetriever()
        manager = FakeDocumentManager([make_record("doc-1")])
        enable_pgvector(
            monkeypatch,
            runtime_builder=lambda **kwargs: fake_pgvector_runtime(
                manager=manager,
                retriever=retriever,
            ),
        )

        with TestClient(api_module.app) as client:
            response = client.post(
                "/search",
                json={"query": "课程问题", "top_k": 2},
            )

            assert response.status_code == 200
            body = response.json()
            assert body["results"][0]["text"] == "fake hit"
            assert body["indexed_chunks"] == 1
            assert retriever.search_calls == 1
            assert client.app.state.retrieval_ready is True

    def test_documents_use_generic_manager(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)
        manager = FakeDocumentManager(
            [make_record("builtin-knowledge", is_builtin=True), make_record("doc-1")]
        )
        enable_pgvector(
            monkeypatch,
            runtime_builder=lambda **kwargs: fake_pgvector_runtime(manager=manager),
        )

        with TestClient(api_module.app) as client:
            response = client.get("/documents")

            assert response.status_code == 200
            body = response.json()
            assert body["document_count"] == 2
            assert body["chunk_count"] == 2
            assert body["documents"][0]["document_id"] == "builtin-knowledge"

    def test_upload_updates_chunk_count(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)
        manager = FakeDocumentManager()
        enable_pgvector(
            monkeypatch,
            runtime_builder=lambda **kwargs: fake_pgvector_runtime(manager=manager),
        )

        with TestClient(api_module.app) as client:
            response = client.post(
                "/documents",
                files={"file": ("notes.txt", b"hello world", "text/plain")},
            )

            assert response.status_code == 201
            assert manager.ingest_count == 1
            assert client.app.state.chunk_count == 1

    def test_delete_updates_chunk_count(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)
        manager = FakeDocumentManager(
            [make_record("doc-1"), make_record("doc-2")]
        )
        enable_pgvector(
            monkeypatch,
            runtime_builder=lambda **kwargs: fake_pgvector_runtime(manager=manager),
        )

        with TestClient(api_module.app) as client:
            response = client.delete("/documents/doc-1")

            assert response.status_code == 200
            assert manager.delete_count == 1
            body = response.json()
            assert body["chunk_count"] == 1
            assert client.app.state.chunk_count == 1

    def test_ask_uses_generic_retriever(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)
        retriever = FakeRetriever()
        enable_pgvector(
            monkeypatch,
            runtime_builder=lambda **kwargs: fake_pgvector_runtime(
                retriever=retriever,
            ),
        )

        with TestClient(api_module.app) as client:
            response = client.post("/ask", json={"question": "课程问题"})

            # Storage is ready but the fake generation service is
            # misconfigured, so the LLM error surface is unchanged.
            assert response.status_code == 503
            assert response.json()["code"] == "LLM_NOT_CONFIGURED"


class TestPgvectorDegraded:
    def test_missing_database_url_503(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)
        enable_pgvector(monkeypatch, database_url="")

        with TestClient(api_module.app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "degraded"
            assert health.json()["retrieval_ready"] is False
            assert health.json()["rag_ready"] is False
            assert health.json()["chunk_count"] == 0

            documents = client.get("/documents")
            assert documents.status_code == 503
            assert documents.json() == {
                "code": "STORAGE_NOT_CONFIGURED",
                "message": "存储服务配置无效",
            }

            search = client.post("/search", json={"query": "课程问题"})
            assert search.status_code == 503
            ask = client.post("/ask", json={"question": "课程问题"})
            assert ask.status_code == 503
            upload = client.post(
                "/documents",
                files={"file": ("notes.txt", b"data", "text/plain")},
            )
            assert upload.status_code == 503
            delete = client.delete("/documents/whatever")
            assert delete.status_code == 503

    def test_database_unavailable_503(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)

        def fail_builder(**kwargs: object) -> None:
            raise DatabaseConnectionError("数据库连接不可用")

        enable_pgvector(
            monkeypatch,
            database_url="postgresql+psycopg://user:hunter2secret@host:5432/db",
            runtime_builder=fail_builder,
        )

        with TestClient(api_module.app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "degraded"

            documents = client.get("/documents")
            assert documents.status_code == 503
            assert documents.json() == {
                "code": "STORAGE_UNAVAILABLE",
                "message": "存储服务暂时不可用",
            }
            # The connection password must never leak into the HTTP response.
            assert "hunter2secret" not in documents.text
            assert "postgresql" not in documents.text.lower()

    def test_schema_not_migrated_503(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)

        def fail_builder(**kwargs: object) -> None:
            raise DatabaseSchemaError("数据库结构尚未准备完成")

        enable_pgvector(
            monkeypatch,
            runtime_builder=fail_builder,
        )

        with TestClient(api_module.app) as client:
            health = client.get("/health")
            assert health.json()["status"] == "degraded"

            documents = client.get("/documents")
            assert documents.status_code == 503
            assert documents.json() == {
                "code": "STORAGE_SCHEMA_NOT_READY",
                "message": "数据库结构尚未准备完成",
            }

    def test_degraded_mode_keeps_health_available(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setup_common(monkeypatch, tmp_path)
        enable_pgvector(monkeypatch, database_url="")

        with TestClient(api_module.app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "degraded"
            assert body["retrieval_ready"] is False
            assert body["rag_ready"] is False
            assert body["chunk_count"] == 0
