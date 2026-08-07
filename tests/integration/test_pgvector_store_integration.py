"""Integration tests for PgVectorStore against a real pgvector database."""

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text

from src.documents import ChunkRecord, DocumentRecord, utc_now_iso
from src.database.engine import create_session_factory
from src.exceptions import DatabaseOperationError

from conftest import (
    document_embeddings,
    integration_database_url,
    query_embedding,
)


def _document(document_id: str, chunk_count: int) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename="it-notes.txt",
        stored_filename="0123456789abcdef0123456789abcdef.txt",
        content_type="text/plain",
        size_bytes=100,
        text_length=100,
        chunk_count=chunk_count,
        created_at=utc_now_iso(),
        is_builtin=False,
    )


def _chunk(document_id: str, index: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{document_id}-{index}",
        document_id=document_id,
        filename="it-notes.txt",
        text=text,
        chunk_index=index,
        page_number=None,
    )


def _unique() -> str:
    return f"it-{uuid4().hex}"


class TestStoreIntegration:
    def test_insert_list_get_and_chunk_count(
        self,
        store,
        cleanup_uploads,
    ) -> None:
        first = _unique()
        second = _unique()
        store.insert_document(
            _document(first, 2),
            [_chunk(first, 0, "alpha 内容"), _chunk(first, 1, "beta 内容")],
            document_embeddings(
                [_chunk(first, 0, "alpha 内容"), _chunk(first, 1, "beta 内容")]
            ),
        )
        store.insert_document(
            _document(second, 1),
            [_chunk(second, 0, "gamma 内容")],
            document_embeddings([_chunk(second, 0, "gamma 内容")]),
        )

        documents = store.list_documents()
        ids = {record.document_id for record in documents}
        assert first in ids
        assert second in ids

        record = store.get_document(first)
        assert record is not None
        assert record.document_id == first
        assert record.chunk_count == 2
        assert store.get_document("missing-doc") is None

        assert store.count_document_chunks(first) == 2
        assert store.count_document_chunks(second) == 1

    def test_cosine_search_returns_matching_chunk(
        self,
        store,
        cleanup_uploads,
    ) -> None:
        document_id = _unique()
        chunks = [
            _chunk(document_id, 0, "alpha 检索目标内容"),
            _chunk(document_id, 1, "beta 完全不相关内容"),
        ]
        store.insert_document(
            _document(document_id, 2),
            chunks,
            document_embeddings(chunks),
        )

        results = store.search(query_embedding("alpha 检索目标内容"), top_k=2)

        assert len(results) == 2
        assert results[0]["document_id"] == document_id
        assert results[0]["chunk_index"] == 0
        assert results[0]["text"] == "alpha 检索目标内容"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)
        assert results[0]["rank"] == 1
        assert results[0]["filename"] == "it-notes.txt"
        assert results[0]["page_number"] is None

    def test_top_k_respected(
        self,
        store,
        cleanup_uploads,
    ) -> None:
        document_id = _unique()
        chunks = [_chunk(document_id, i, f"内容 {i}") for i in range(5)]
        store.insert_document(
            _document(document_id, 5),
            chunks,
            document_embeddings(chunks),
        )

        assert len(store.search(query_embedding("内容 2"), top_k=3)) == 3
        assert len(store.search(query_embedding("内容 2"), top_k=1)) == 1
        with pytest.raises(ValueError):
            store.search(query_embedding("内容 2"), top_k=0)

    def test_tie_break_is_stable_by_document_and_index(
        self,
        store,
        cleanup_uploads,
    ) -> None:
        first = _unique()
        second = _unique()
        shared_text = "完全相同的内容"
        store.insert_document(
            _document(first, 1),
            [_chunk(first, 0, shared_text)],
            document_embeddings([_chunk(first, 0, shared_text)]),
        )
        store.insert_document(
            _document(second, 1),
            [_chunk(second, 0, shared_text)],
            document_embeddings([_chunk(second, 0, shared_text)]),
        )

        results = store.search(query_embedding(shared_text), top_k=2)
        first_run = [r["document_id"] for r in results]
        second_run = [r["document_id"] for r in store.search(query_embedding(shared_text), top_k=2)]

        assert len(first_run) == 2
        assert first_run == sorted(first_run)  # document_id ascending tie-break
        assert first_run == second_run  # deterministic across runs

    def test_only_ready_documents_are_returned(
        self,
        store,
        session_factory,
        cleanup_uploads,
    ) -> None:
        ready_id = _unique()
        failed_id = _unique()
        store.insert_document(
            _document(ready_id, 1),
            [_chunk(ready_id, 0, "ready 内容")],
            document_embeddings([_chunk(ready_id, 0, "ready 内容")]),
        )
        store.insert_document(
            _document(failed_id, 1),
            [_chunk(failed_id, 0, "failed 内容")],
            document_embeddings([_chunk(failed_id, 0, "failed 内容")]),
        )
        with session_factory.begin() as session:
            session.execute(
                text("UPDATE documents SET status = 'failed' WHERE document_id = :id"),
                {"id": failed_id},
            )

        assert store.get_document(failed_id) is None
        ready_docs = store.list_documents()
        ids = {record.document_id for record in ready_docs}
        assert failed_id not in ids
        assert ready_id in ids
        # chunk_count counts only ready documents: it must equal the sum of
        # chunk counts over the ready documents (the failed one excluded).
        expected_count = sum(
            store.count_document_chunks(record.document_id)
            for record in ready_docs
        )
        assert store.chunk_count == expected_count

        results = store.search(query_embedding("failed 内容"), top_k=5)
        assert all(result["document_id"] != failed_id for result in results)

    def test_delete_cascades_chunks(
        self,
        store,
        session_factory,
        cleanup_uploads,
    ) -> None:
        document_id = _unique()
        chunks = [_chunk(document_id, i, f"内容 {i}") for i in range(3)]
        store.insert_document(
            _document(document_id, 3),
            chunks,
            document_embeddings(chunks),
        )

        record = store.delete_document(document_id)

        assert record.document_id == document_id
        assert store.get_document(document_id) is None
        assert store.count_document_chunks(document_id) == 0
        with session_factory() as session:
            remaining = session.scalar(
                text(
                    "SELECT count(*) FROM chunks WHERE document_id = :id"
                ),
                {"id": document_id},
            )
        assert remaining == 0  # ON DELETE CASCADE removed the chunks

    def test_engine_dispose_and_recreate_persists(
        self,
        session_factory,
        cleanup_uploads,
    ) -> None:
        document_id = _unique()
        from src.storage.pgvector_store import PgVectorStore

        store = PgVectorStore(session_factory)
        store.insert_document(
            _document(document_id, 1),
            [_chunk(document_id, 0, "持久化内容")],
            document_embeddings([_chunk(document_id, 0, "持久化内容")]),
        )

        # Simulate an application restart: dispose the engine and rebuild.
        engine = create_engine(integration_database_url())
        try:
            recreated_store = PgVectorStore(create_session_factory(engine))
            record = recreated_store.get_document(document_id)
            assert record is not None
            assert record.document_id == document_id
            results = recreated_store.search(
                query_embedding("持久化内容"),
                top_k=1,
            )
            assert len(results) == 1
            assert results[0]["document_id"] == document_id
        finally:
            engine.dispose()

    def test_duplicate_document_id_fails_without_partial_chunks(
        self,
        store,
        cleanup_uploads,
    ) -> None:
        document_id = _unique()
        original_chunks = [
            _chunk(document_id, 0, "第一块"),
            _chunk(document_id, 1, "第二块"),
        ]
        store.insert_document(
            _document(document_id, 2),
            original_chunks,
            document_embeddings(original_chunks),
        )

        conflicting = [
            _chunk(document_id, 0, "冲突块 A"),
            _chunk(document_id, 1, "冲突块 B"),
            _chunk(document_id, 2, "冲突块 C"),
        ]
        with pytest.raises(DatabaseOperationError):
            store.insert_document(
                _document(document_id, 3),
                conflicting,
                document_embeddings(conflicting),
            )

        # The failed transaction left no partial chunks behind.
        assert store.count_document_chunks(document_id) == 2
        record = store.get_document(document_id)
        assert record is not None
        assert record.chunk_count == 2

    def test_vector_column_is_384_dimensions(
        self,
        store,
        session_factory,
        cleanup_uploads,
    ) -> None:
        document_id = _unique()
        store.insert_document(
            _document(document_id, 1),
            [_chunk(document_id, 0, "384 维向量")],
            document_embeddings([_chunk(document_id, 0, "384 维向量")]),
        )
        with session_factory() as session:
            dimension = session.scalar(
                text("SELECT vector_dims(embedding) FROM chunks LIMIT 1")
            )
        assert dimension == 384
