"""Unit tests for PgVectorStore without a real PostgreSQL database."""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError

from src.documents import ChunkRecord, DocumentRecord, utc_now_iso
from src.database.models import DocumentModel
from src.exceptions import DatabaseOperationError, DocumentNotFoundError
from src.storage.pgvector_store import (
    PgVectorStore,
    _cosine_score,
    _datetime_to_utc_iso,
    _document_model_to_record,
    _record_to_document_model,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class UnusedSessionFactory:
    """Fail hard when the store touches a session it must not need."""

    def __call__(self) -> None:
        raise AssertionError("测试不应连接数据库")

    def begin(self) -> None:
        raise AssertionError("测试不应连接数据库")


class RecordedSession:
    """Record executed statements and return canned database results."""

    def __init__(
        self,
        *,
        rows: list[tuple[object, float]] | None = None,
        models: list[DocumentModel] | None = None,
        scalars: int | list[int] | None = None,
        get_result: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self._rows = rows or []
        self._models = models or []
        self._scalars = scalars
        self._get_result = get_result
        self._error = error
        self.statements: list[object] = []
        self.added: list[object] = []
        self.deleted: list[object] = []

    def _record(self, statement: object) -> None:
        self.statements.append(statement)
        if self._error is not None:
            raise self._error

    def execute(self, statement: object) -> SimpleNamespace:
        self._record(statement)
        return SimpleNamespace(all=lambda: self._rows)

    def scalars(self, statement: object) -> SimpleNamespace:
        self._record(statement)
        return SimpleNamespace(all=lambda: self._models)

    def scalar(self, statement: object) -> int | None:
        self._record(statement)
        if isinstance(self._scalars, list):
            return self._scalars.pop(0)
        return self._scalars

    def get(self, _model: object, _document_id: str) -> object | None:
        self._record(SimpleNamespace(__tag__="get"))
        if self._error is not None:
            raise self._error
        return self._get_result

    def add(self, model: object) -> None:
        self.added.append(model)

    def add_all(self, models: list[object]) -> None:
        self.added.extend(models)

    def flush(self) -> None:
        return None

    def delete(self, model: object) -> None:
        self.deleted.append(model)

    def __enter__(self) -> "RecordedSession":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


class FakeSessionFactory:
    def __init__(self, session: RecordedSession) -> None:
        self._session = session

    def __call__(self) -> RecordedSession:
        return self._session

    def begin(self) -> RecordedSession:
        return self._session


def make_document(
    document_id: str = "doc-1",
    chunk_count: int = 1,
    is_builtin: bool = False,
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename="notes.txt",
        stored_filename=None if is_builtin else "0123456789abcdef0123456789abcdef.txt",
        content_type="text/plain",
        size_bytes=10,
        text_length=10,
        chunk_count=chunk_count,
        created_at=utc_now_iso(),
        is_builtin=is_builtin,
    )


def make_chunk(chunk_index: int = 0, document_id: str = "doc-1") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"chunk-{document_id}-{chunk_index}",
        document_id=document_id,
        filename="notes.txt",
        text=f"chunk text {chunk_index}",
        chunk_index=chunk_index,
        page_number=None,
    )


def make_chunk_model(
    chunk_index: int = 0,
    document_id: str = "doc-1",
    content: str = "chunk text",
    filename: str = "notes.txt",
    page_number: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        chunk_index=chunk_index,
        document_id=document_id,
        content=content,
        filename=filename,
        page_number=page_number,
    )


def compiled_sql(statement: object) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


# ---------------------------------------------------------------------------
# Query embedding validation
# ---------------------------------------------------------------------------


class TestQueryEmbeddingValidation:
    def test_top_k_zero_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        with pytest.raises(ValueError, match="top_k"):
            store.search(np.ones(384), top_k=0)

    def test_top_k_negative_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        with pytest.raises(ValueError, match="top_k"):
            store.search(np.ones(384), top_k=-1)

    def test_query_not_one_dimensional(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        with pytest.raises(ValueError):
            store.search(np.ones((2, 384)), top_k=1)

    def test_query_wrong_dimension(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        with pytest.raises(ValueError, match="384"):
            store.search(np.ones(3), top_k=1)

    def test_query_nan_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        with pytest.raises(ValueError):
            store.search(np.full(384, np.nan), top_k=1)

    def test_query_infinity_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        with pytest.raises(ValueError):
            store.search(np.full(384, np.inf), top_k=1)

    def test_query_zero_vector_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        with pytest.raises(ValueError):
            store.search(np.zeros(384), top_k=1)

    def test_error_messages_never_contain_vector_values(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        with pytest.raises(ValueError) as exc_info:
            store.search(np.full(384, np.nan), top_k=1)
        message = str(exc_info.value)
        assert "nan" not in message.lower()


# ---------------------------------------------------------------------------
# Document embedding validation
# ---------------------------------------------------------------------------


class TestDocumentEmbeddingValidation:
    def test_row_count_mismatch_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        document = make_document(chunk_count=2)
        chunks = [make_chunk(0), make_chunk(1)]
        with pytest.raises(ValueError, match="行数"):
            store.insert_document(document, chunks, np.ones((1, 384)))

    def test_not_two_dimensional_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        document = make_document(chunk_count=1)
        with pytest.raises(ValueError):
            store.insert_document(document, [make_chunk(0)], np.ones(384))

    def test_wrong_column_dimension_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        document = make_document(chunk_count=1)
        with pytest.raises(ValueError, match="384"):
            store.insert_document(document, [make_chunk(0)], np.ones((1, 3)))

    def test_nan_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        document = make_document(chunk_count=1)
        with pytest.raises(ValueError):
            store.insert_document(
                document,
                [make_chunk(0)],
                np.full((1, 384), np.nan),
            )

    def test_infinity_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        document = make_document(chunk_count=1)
        with pytest.raises(ValueError):
            store.insert_document(
                document,
                [make_chunk(0)],
                np.full((1, 384), np.inf),
            )

    def test_zero_vector_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        document = make_document(chunk_count=1)
        with pytest.raises(ValueError):
            store.insert_document(
                document,
                [make_chunk(0)],
                np.zeros((1, 384)),
            )

    def test_empty_chunks_rejected(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        document = make_document(chunk_count=0)
        with pytest.raises(ValueError):
            store.insert_document(document, [], np.empty((0, 384)))

    def test_caller_array_not_mutated(self) -> None:
        store = PgVectorStore(UnusedSessionFactory())
        document = make_document(chunk_count=1)
        original = np.full((1, 384), 0.5)
        snapshot = original.copy()
        with pytest.raises(ValueError):
            # wrong row count must not mutate the caller's array
            store.insert_document(document, [], original)
        assert np.array_equal(original, snapshot)


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------


class TestModelConversion:
    def test_record_to_model_ready_status(self) -> None:
        record = make_document(document_id="abc", is_builtin=False)
        model = _record_to_document_model(record)
        assert model.document_id == "abc"
        assert model.status == "ready"
        assert model.stored_filename == record.stored_filename
        assert model.is_builtin is False
        assert model.created_at.tzinfo is not None

    def test_model_to_record_utc_z_suffix(self) -> None:
        model = DocumentModel(
            document_id="d1",
            original_filename="f.txt",
            stored_filename=None,
            content_type="text/plain",
            size_bytes=1,
            text_length=1,
            chunk_count=1,
            is_builtin=True,
            status="ready",
            created_at=datetime(2026, 8, 6, 12, 30, 45, tzinfo=timezone.utc),
        )
        record = _document_model_to_record(model)
        assert record.created_at == "2026-08-06T12:30:45Z"
        assert record.is_builtin is True
        assert record.stored_filename is None
        assert record.document_id == "d1"

    def test_model_to_record_naive_datetime_is_utc(self) -> None:
        model = DocumentModel(
            document_id="d2",
            original_filename="f.txt",
            stored_filename="0123456789abcdef0123456789abcdef.txt",
            content_type="text/plain",
            size_bytes=1,
            text_length=1,
            chunk_count=2,
            is_builtin=False,
            status="ready",
            created_at=datetime(2026, 8, 6, 12, 30, 45),
        )
        record = _document_model_to_record(model)
        assert record.created_at == "2026-08-06T12:30:45Z"
        assert record.stored_filename == "0123456789abcdef0123456789abcdef.txt"

    def test_datetime_to_utc_iso(self) -> None:
        value = datetime(
            2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc
        )
        assert _datetime_to_utc_iso(value) == "2026-01-02T03:04:05Z"

    def test_round_trip_preserves_created_at(self) -> None:
        record = make_document()
        model = _record_to_document_model(record)
        restored = _document_model_to_record(model)
        assert restored.created_at == record.created_at


# ---------------------------------------------------------------------------
# Search behavior
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_result_shape_and_score(self) -> None:
        rows = [
            (make_chunk_model(chunk_index=0, document_id="a", content="alpha"), 0.2),
            (make_chunk_model(chunk_index=5, document_id="b", content="beta"), 0.35),
        ]
        session = RecordedSession(rows=rows)
        store = PgVectorStore(FakeSessionFactory(session))

        results = store.search(np.ones(384), top_k=2)

        assert len(results) == 2
        assert results[0]["rank"] == 1
        assert results[0]["score"] == pytest.approx(0.8)
        assert results[0]["text"] == "alpha"
        assert results[0]["chunk_index"] == 0
        assert results[0]["document_id"] == "a"
        assert results[0]["filename"] == "notes.txt"
        assert results[0]["page_number"] is None
        assert results[1]["rank"] == 2
        assert results[1]["score"] == pytest.approx(0.65)

    def test_search_sql_ready_filter_order_and_limit(self) -> None:
        session = RecordedSession(rows=[])
        store = PgVectorStore(FakeSessionFactory(session))

        store.search(np.ones(384), top_k=3)

        sql = compiled_sql(session.statements[0])
        assert "documents.status = 'ready'" in sql
        assert "ORDER BY" in sql
        assert "LIMIT 3" in sql
        assert "<=>" in sql  # cosine distance operator

    def test_search_database_error_converted(self) -> None:
        driver_error = OperationalError(
            "SELECT ...",
            {},
            Exception("connection refused host=db password=supersecret"),
        )
        session = RecordedSession(error=driver_error)
        store = PgVectorStore(FakeSessionFactory(session))

        with pytest.raises(DatabaseOperationError) as exc_info:
            store.search(np.ones(384), top_k=1)

        message = str(exc_info.value)
        assert "数据库操作失败" in message
        assert "password" not in message.lower()
        assert "supersecret" not in message
        assert "db" not in message
        assert isinstance(exc_info.value.__cause__, OperationalError)


# ---------------------------------------------------------------------------
# Read and write paths
# ---------------------------------------------------------------------------


class TestReadWritePaths:
    def test_list_documents_converts_models(self) -> None:
        model = DocumentModel(
            document_id="d1",
            original_filename="f.txt",
            stored_filename=None,
            content_type="text/plain",
            size_bytes=1,
            text_length=1,
            chunk_count=1,
            is_builtin=True,
            status="ready",
            created_at=datetime(2026, 8, 6, 12, 30, 45, tzinfo=timezone.utc),
        )
        session = RecordedSession(models=[model])
        store = PgVectorStore(FakeSessionFactory(session))

        records = store.list_documents()

        assert len(records) == 1
        assert records[0].document_id == "d1"
        assert records[0].is_builtin is True
        assert records[0].created_at == "2026-08-06T12:30:45Z"

    def test_list_documents_database_error_converted(self) -> None:
        session = RecordedSession(error=OperationalError("s", {}, Exception("boom")))
        store = PgVectorStore(FakeSessionFactory(session))
        with pytest.raises(DatabaseOperationError):
            store.list_documents()

    def test_get_document_missing_returns_none(self) -> None:
        session = RecordedSession()
        store = PgVectorStore(FakeSessionFactory(session))

        result = store.get_document("missing")

        assert result is None
        sql = compiled_sql(session.statements[0])
        assert "documents.status = 'ready'" in sql

    def test_chunk_count_uses_database_aggregate(self) -> None:
        session = RecordedSession(scalars=7)
        store = PgVectorStore(FakeSessionFactory(session))

        assert store.chunk_count == 7
        sql = compiled_sql(session.statements[0])
        assert "count(" in sql
        assert "documents.status = 'ready'" in sql

    def test_insert_document_commits_atomically(self) -> None:
        session = RecordedSession(scalars=[1, 2])  # one document, two chunks
        store = PgVectorStore(FakeSessionFactory(session))
        document = make_document(chunk_count=2)
        chunks = [make_chunk(0), make_chunk(1)]

        store.insert_document(document, chunks, np.ones((2, 384)))

        assert len(session.added) == 3  # one document + two chunks
        assert isinstance(session.added[0], DocumentModel)

    def test_insert_document_row_count_verification_fails(self) -> None:
        session = RecordedSession(scalars=[1, 0])  # chunks missing
        store = PgVectorStore(FakeSessionFactory(session))
        document = make_document(chunk_count=1)

        with pytest.raises(DatabaseOperationError):
            store.insert_document(document, [make_chunk(0)], np.ones((1, 384)))

    def test_insert_document_database_error_converted(self) -> None:
        driver_error = OperationalError(
            "INSERT",
            {},
            Exception("duplicate key password=hunter2"),
        )
        session = RecordedSession(error=driver_error)
        store = PgVectorStore(FakeSessionFactory(session))
        document = make_document(chunk_count=1)

        with pytest.raises(DatabaseOperationError) as exc_info:
            store.insert_document(document, [make_chunk(0)], np.ones((1, 384)))

        message = str(exc_info.value)
        assert "password" not in message.lower()
        assert "hunter2" not in message

    def test_delete_document_returns_record(self) -> None:
        model = DocumentModel(
            document_id="d1",
            original_filename="f.txt",
            stored_filename="0123456789abcdef0123456789abcdef.txt",
            content_type="text/plain",
            size_bytes=1,
            text_length=1,
            chunk_count=1,
            is_builtin=False,
            status="ready",
            created_at=datetime(2026, 8, 6, 12, 30, 45, tzinfo=timezone.utc),
        )
        session = RecordedSession(get_result=model)
        store = PgVectorStore(FakeSessionFactory(session))

        record = store.delete_document("d1")

        assert record.document_id == "d1"
        assert len(session.deleted) == 1

    def test_delete_document_missing_raises(self) -> None:
        session = RecordedSession(get_result=None)
        store = PgVectorStore(FakeSessionFactory(session))

        with pytest.raises(DocumentNotFoundError):
            store.delete_document("missing")

    def test_delete_document_database_error_converted(self) -> None:
        session = RecordedSession(error=OperationalError("D", {}, Exception("down")))
        store = PgVectorStore(FakeSessionFactory(session))

        with pytest.raises(DatabaseOperationError):
            store.delete_document("d1")

    def test_close_is_noop(self) -> None:
        store = PgVectorStore(FakeSessionFactory(RecordedSession()))
        store.close()
        store.close()


# ---------------------------------------------------------------------------
# Score conversion
# ---------------------------------------------------------------------------


class TestScoreConversion:
    def test_distance_to_score(self) -> None:
        assert _cosine_score(0.0) == pytest.approx(1.0)
        assert _cosine_score(0.25) == pytest.approx(0.75)
        assert _cosine_score(1.0) == pytest.approx(0.0)
        assert _cosine_score(2.0) == pytest.approx(-1.0)

    def test_epsilon_correction_at_boundaries(self) -> None:
        # 1.0 - (2.0 + 5e-10) = -1.0 - 5e-10 -> clamped to -1.0
        assert _cosine_score(2.0 + 5e-10) == -1.0
        # 1.0 - (-5e-10) = 1.0 + 5e-10 -> clamped to 1.0
        assert _cosine_score(-5e-10) == 1.0

    def test_severe_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            _cosine_score(3.0)  # score would be -2.0

    def test_non_finite_distance_raises(self) -> None:
        with pytest.raises(ValueError):
            _cosine_score(float("nan"))
        with pytest.raises(ValueError):
            _cosine_score(float("inf"))
