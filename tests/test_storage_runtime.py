"""Unit tests for storage runtime assembly without a real database."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.storage.runtime as runtime_module
from src.exceptions import DatabaseOperationError, DatabaseSchemaError
from src.knowledge_index import KnowledgeIndex
from src.storage.runtime import build_memory_runtime, build_pgvector_runtime


class FakeEmbeddingService:
    model_name = "fake/384"

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.full((len(texts), 384), 1.0 / np.sqrt(384))

    def encode_query(self, _query: str) -> np.ndarray:
        return np.full(384, 1.0 / np.sqrt(384))


class FakeStore:
    def __init__(self, session_factory: object, **kwargs: object) -> None:
        del session_factory, kwargs

    @property
    def chunk_count(self) -> int:
        return 0

    def search(self, _query: object, top_k: int) -> list[dict[str, object]]:
        del top_k
        return []

    def list_documents(self) -> list:
        return []

    def get_document(self, _document_id: str) -> None:
        return None

    def count_document_chunks(self, _document_id: str) -> int:
        return 0

    def insert_document(self, *args: object) -> None:
        del args

    def delete_document(self, _document_id: str) -> None:
        raise AssertionError("unused")

    def close(self) -> None:
        return None


class FakeIngestionService:
    def __init__(self, **kwargs: object) -> None:
        del kwargs

    @property
    def max_upload_bytes(self) -> int:
        return 1024

    @property
    def chunk_count(self) -> int:
        return 0

    def list_documents(self) -> list:
        return []

    def get_document(self, _document_id: str) -> None:
        return None

    def ingest(self, *args: object) -> None:
        del args
        raise AssertionError("unused")

    def delete_document(self, _document_id: str) -> None:
        raise AssertionError("unused")

    def reconcile_tombstones(self) -> None:
        runtime_module.__dict__.setdefault("_fake_calls", []).append("reconcile")

    def ensure_builtin_document(self) -> None:
        runtime_module.__dict__.setdefault("_fake_calls", []).append("builtin")

    def close(self) -> None:
        return None


@pytest.fixture
def knowledge_path(tmp_path: Path) -> Path:
    path = tmp_path / "knowledge.txt"
    path.write_text("内置课程知识内容。" * 10, encoding="utf-8")
    return path


@pytest.fixture
def pgvector_mocks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    class RecordingEngine:
        def dispose(self) -> None:
            calls.append("dispose")

    engine = RecordingEngine()
    monkeypatch.setattr(
        runtime_module,
        "create_database_engine",
        lambda _url: engine,
    )
    monkeypatch.setattr(
        runtime_module,
        "check_database_connection",
        lambda _engine: calls.append("connection"),
    )
    monkeypatch.setattr(
        runtime_module,
        "check_database_schema",
        lambda _engine, **kwargs: calls.append("schema"),
    )
    monkeypatch.setattr(
        runtime_module,
        "create_session_factory",
        lambda _engine: SimpleNamespace(),
    )
    monkeypatch.setattr(
        runtime_module,
        "PgVectorStore",
        FakeStore,
    )
    monkeypatch.setattr(
        runtime_module,
        "PgVectorIngestionService",
        FakeIngestionService,
    )
    runtime_module.__dict__["_fake_calls"] = calls
    return calls


# ---------------------------------------------------------------------------
# memory runtime
# ---------------------------------------------------------------------------


class TestMemoryRuntime:
    def test_default_memory_uses_existing_components(
        self,
        tmp_path: Path,
        knowledge_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VECTOR_STORE_BACKEND", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)

        runtime = build_memory_runtime(
            embedding_service=FakeEmbeddingService(),
            knowledge_path=knowledge_path,
            upload_dir=tmp_path / "uploads",
            metadata_path=tmp_path / "documents.json",
        )

        assert runtime.backend == "memory"
        assert isinstance(runtime.retriever, KnowledgeIndex)
        assert hasattr(runtime.document_manager, "ingest")
        assert hasattr(runtime.document_manager, "delete_document")
        assert runtime.document_manager.chunk_count == runtime.retriever.chunk_count

    def test_memory_does_not_read_database_url(
        self,
        tmp_path: Path,
        knowledge_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg://user:supersecret@host:5432/db",
        )

        runtime = build_memory_runtime(
            embedding_service=FakeEmbeddingService(),
            knowledge_path=knowledge_path,
            upload_dir=tmp_path / "uploads",
            metadata_path=tmp_path / "documents.json",
        )

        assert runtime.backend == "memory"
        runtime.close()

    def test_memory_never_creates_an_engine(
        self,
        tmp_path: Path,
        knowledge_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_if_called(_url: str) -> None:
            raise AssertionError("memory 模式不应创建 engine")

        monkeypatch.setattr(runtime_module, "create_database_engine", fail_if_called)
        runtime = build_memory_runtime(
            embedding_service=FakeEmbeddingService(),
            knowledge_path=knowledge_path,
            upload_dir=tmp_path / "uploads",
            metadata_path=tmp_path / "documents.json",
        )
        assert runtime.backend == "memory"

    def test_close_is_idempotent(
        self,
        tmp_path: Path,
        knowledge_path: Path,
    ) -> None:
        runtime = build_memory_runtime(
            embedding_service=FakeEmbeddingService(),
            knowledge_path=knowledge_path,
            upload_dir=tmp_path / "uploads",
            metadata_path=tmp_path / "documents.json",
        )
        runtime.close()
        runtime.close()  # must not raise


# ---------------------------------------------------------------------------
# pgvector runtime
# ---------------------------------------------------------------------------


class TestPgvectorRuntime:
    def test_pgvector_creates_engine_checks_and_builtin(
        self,
        tmp_path: Path,
        knowledge_path: Path,
        pgvector_mocks: list[str],
    ) -> None:
        runtime = build_pgvector_runtime(
            embedding_service=FakeEmbeddingService(),
            database_url="postgresql+psycopg://u:p@h/db",
            knowledge_path=knowledge_path,
            upload_dir=tmp_path / "uploads",
            script_location=tmp_path / "migrations",
            alembic_config_path=tmp_path / "alembic.ini",
        )

        assert runtime.backend == "pgvector"
        assert "connection" in pgvector_mocks
        assert "schema" in pgvector_mocks
        assert "reconcile" in pgvector_mocks
        assert "builtin" in pgvector_mocks
        # The reconciliation must run before the built-in document write.
        assert pgvector_mocks.index("reconcile") < pgvector_mocks.index("builtin")

    def test_close_disposes_engine_once(
        self,
        tmp_path: Path,
        knowledge_path: Path,
        pgvector_mocks: list[str],
    ) -> None:
        runtime = build_pgvector_runtime(
            embedding_service=FakeEmbeddingService(),
            database_url="postgresql+psycopg://u:p@h/db",
            knowledge_path=knowledge_path,
            upload_dir=tmp_path / "uploads",
            script_location=tmp_path / "migrations",
            alembic_config_path=tmp_path / "alembic.ini",
        )

        runtime.close()
        runtime.close()

        assert pgvector_mocks.count("dispose") == 1

    def test_initialization_failure_disposes_engine(
        self,
        tmp_path: Path,
        knowledge_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        class RecordingEngine:
            def dispose(self) -> None:
                calls.append("dispose")

        engine = RecordingEngine()
        monkeypatch.setattr(
            runtime_module,
            "create_database_engine",
            lambda _url: engine,
        )
        monkeypatch.setattr(
            runtime_module,
            "check_database_connection",
            lambda _engine: None,
        )

        def fail_schema(_engine: object, **kwargs: object) -> None:
            raise DatabaseSchemaError("数据库结构尚未准备完成")

        monkeypatch.setattr(
            runtime_module,
            "check_database_schema",
            fail_schema,
        )

        with pytest.raises(DatabaseSchemaError):
            build_pgvector_runtime(
                embedding_service=FakeEmbeddingService(),
                database_url="postgresql+psycopg://u:p@h/db",
                knowledge_path=knowledge_path,
                upload_dir=tmp_path / "uploads",
            )

        assert calls == ["dispose"]

    def test_reconcile_database_error_blocks_startup_and_disposes_once(
        self,
        tmp_path: Path,
        knowledge_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A store-level DatabaseOperationError during reconciliation is not
        swallowed: the runtime build fails and the engine is disposed once."""
        calls: list[str] = []

        class RecordingEngine:
            def dispose(self) -> None:
                calls.append("dispose")

        engine = RecordingEngine()
        monkeypatch.setattr(
            runtime_module,
            "create_database_engine",
            lambda _url: engine,
        )
        monkeypatch.setattr(
            runtime_module,
            "check_database_connection",
            lambda _engine: None,
        )
        monkeypatch.setattr(
            runtime_module,
            "check_database_schema",
            lambda _engine, **kwargs: None,
        )
        monkeypatch.setattr(
            runtime_module,
            "create_session_factory",
            lambda _engine: SimpleNamespace(),
        )
        monkeypatch.setattr(
            runtime_module,
            "PgVectorStore",
            FakeStore,
        )

        class FailingReconcileService(FakeIngestionService):
            def reconcile_tombstones(self) -> None:
                raise DatabaseOperationError("数据库操作失败，请稍后重试")

        monkeypatch.setattr(
            runtime_module,
            "PgVectorIngestionService",
            FailingReconcileService,
        )

        with pytest.raises(DatabaseOperationError):
            build_pgvector_runtime(
                embedding_service=FakeEmbeddingService(),
                database_url="postgresql+psycopg://u:p@h/db",
                knowledge_path=knowledge_path,
                upload_dir=tmp_path / "uploads",
            )

        assert calls == ["dispose"]
