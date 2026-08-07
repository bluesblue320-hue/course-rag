"""PostgreSQL + pgvector persistence and retrieval store.

This module owns the *database* half of the ``pgvector`` runtime:

* document metadata CRUD
* chunk text and embedding persistence
* pgvector cosine-distance retrieval
* chunk counting
* transaction boundaries
* ORM / domain-model conversion
* database error conversion

It deliberately does **not** handle file I/O, PDF/Markdown parsing, chunk
splitting, embedding model calls, or HTTP responses — those live in
:mod:`src.pgvector_ingestion_service` and the API layer.
"""

import math
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import Engine, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.database.base import DEFAULT_EMBEDDING_DIMENSION
from src.database.models import ChunkModel, DocumentModel
from src.documents import ChunkRecord, DocumentRecord
from src.exceptions import (
    DatabaseOperationError,
    DocumentNotFoundError,
)

#: Cosine distance computed by pgvector for two vectors whose norms are
#: exactly one is ``1 - dot``, so scores stay inside [-1, 1].  A tiny
#: tolerance absorbs float error at the boundary without clamping severe
#: anomalies.
_SCORE_EPSILON = 1e-9

_INVALID_VECTOR_SHAPE = "向量形状无效"
_INVALID_VECTOR_VALUE = "向量包含无效值"
_ZERO_VECTOR = "向量不能是零向量"
_OPERATION_FAILED = "数据库操作失败，请稍后重试"


def _validate_document_embeddings(
    embeddings: object,
    expected_rows: int,
    dimension: int,
) -> np.ndarray:
    """Validate a document embedding matrix and return a fresh float array.

    The caller's array is never mutated.  Error messages never contain
    vector values.
    """
    array = np.asarray(embeddings, dtype=float)
    if array.ndim != 2:
        raise ValueError(_INVALID_VECTOR_SHAPE)
    if array.shape[0] != expected_rows:
        raise ValueError("向量行数必须与 Chunk 数量一致")
    if array.shape[1] != dimension:
        raise ValueError(f"向量维度必须为 {dimension}")
    if not np.all(np.isfinite(array)):
        raise ValueError(_INVALID_VECTOR_VALUE)
    if np.any(np.linalg.norm(array, axis=1) == 0):
        raise ValueError(_ZERO_VECTOR)
    return array


def _validate_query_embedding(
    embedding: object,
    dimension: int,
) -> np.ndarray:
    """Validate one query embedding and return a fresh float array."""
    array = np.asarray(embedding, dtype=float)
    if array.ndim != 1:
        raise ValueError(_INVALID_VECTOR_SHAPE)
    if array.shape[0] != dimension:
        raise ValueError(f"向量维度必须为 {dimension}")
    if not np.all(np.isfinite(array)):
        raise ValueError(_INVALID_VECTOR_VALUE)
    if np.linalg.norm(array) == 0:
        raise ValueError(_ZERO_VECTOR)
    return array


def _datetime_to_utc_iso(value: datetime) -> str:
    """Convert one naive or aware datetime to a UTC ISO 8601 string."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _document_model_to_record(model: DocumentModel) -> DocumentRecord:
    """Convert one ready ORM document row to a domain record."""
    return DocumentRecord(
        document_id=model.document_id,
        original_filename=model.original_filename,
        stored_filename=model.stored_filename,
        content_type=model.content_type,
        size_bytes=model.size_bytes,
        text_length=model.text_length,
        chunk_count=model.chunk_count,
        created_at=_datetime_to_utc_iso(model.created_at),
        is_builtin=bool(model.is_builtin),
    )


def _record_to_document_model(record: DocumentRecord) -> DocumentModel:
    """Convert one domain record into a ready ORM document row."""
    created_at = datetime.fromisoformat(
        record.created_at.replace("Z", "+00:00")
    )
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return DocumentModel(
        document_id=record.document_id,
        original_filename=record.original_filename,
        stored_filename=record.stored_filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        text_length=record.text_length,
        chunk_count=record.chunk_count,
        is_builtin=record.is_builtin,
        status="ready",
        created_at=created_at,
    )


def _cosine_score(distance_value: object) -> float:
    """Convert a pgvector cosine distance into a stable cosine score."""
    score = 1.0 - float(distance_value)
    if not math.isfinite(score):
        raise ValueError("检索结果的分数不是有限数")
    if -1.0 - _SCORE_EPSILON < score < -1.0:
        score = -1.0
    elif 1.0 < score < 1.0 + _SCORE_EPSILON:
        score = 1.0
    if not -1.0 <= score <= 1.0:
        raise ValueError("检索结果的分数超出有效范围")
    return score


class PgVectorStore:
    """Persist documents and chunks and search them inside PostgreSQL."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        engine: Engine | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_dimension = embedding_dimension
        # Kept for optional diagnostics; the store never disposes it.
        self._engine = engine

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        """Count ready chunks with a database aggregate query."""
        statement = (
            select(func.count(ChunkModel.chunk_id))
            .join(
                DocumentModel,
                ChunkModel.document_id == DocumentModel.document_id,
            )
            .where(DocumentModel.status == "ready")
        )
        try:
            with self._session_factory() as session:
                count = session.scalar(statement)
        except SQLAlchemyError as exc:
            raise DatabaseOperationError(_OPERATION_FAILED) from exc
        return int(count or 0)

    def list_documents(self) -> list[DocumentRecord]:
        """Return all ready documents in stable display order."""
        statement = (
            select(DocumentModel)
            .where(DocumentModel.status == "ready")
            .order_by(
                DocumentModel.is_builtin.desc(),
                DocumentModel.created_at.desc(),
                DocumentModel.document_id.asc(),
            )
        )
        try:
            with self._session_factory() as session:
                models = session.scalars(statement).all()
        except SQLAlchemyError as exc:
            raise DatabaseOperationError(_OPERATION_FAILED) from exc
        return [_document_model_to_record(model) for model in models]

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one ready document by id or None when absent."""
        statement = select(DocumentModel).where(
            DocumentModel.document_id == document_id,
            DocumentModel.status == "ready",
        )
        try:
            with self._session_factory() as session:
                model = session.scalar(statement)
        except SQLAlchemyError as exc:
            raise DatabaseOperationError(_OPERATION_FAILED) from exc
        if model is None:
            return None
        return _document_model_to_record(model)

    def count_document_chunks(self, document_id: str) -> int:
        """Count the persisted chunks belonging to one document."""
        statement = select(func.count(ChunkModel.chunk_id)).where(
            ChunkModel.document_id == document_id
        )
        try:
            with self._session_factory() as session:
                count = session.scalar(statement)
        except SQLAlchemyError as exc:
            raise DatabaseOperationError(_OPERATION_FAILED) from exc
        return int(count or 0)

    # ------------------------------------------------------------------
    # Write paths
    # ------------------------------------------------------------------

    def insert_document(
        self,
        document: DocumentRecord,
        chunks: list[ChunkRecord],
        embeddings: object,
    ) -> None:
        """Insert one document and all of its chunks in a single transaction.

        The whole batch commits atomically: a failure leaves no orphan
        document and no partial chunks behind.
        """
        embeddings_array = _validate_document_embeddings(
            embeddings,
            len(chunks),
            self._embedding_dimension,
        )
        if not chunks:
            raise ValueError("Chunk 列表不能为空")

        document_model = _record_to_document_model(document)
        chunk_models = [
            self._make_chunk_model(chunk, vector)
            for chunk, vector in zip(chunks, embeddings_array)
        ]
        try:
            with self._session_factory.begin() as session:
                session.add(document_model)
                session.add_all(chunk_models)
                session.flush()
                self._verify_inserted_document(session, document.document_id, len(chunks))
        except DatabaseOperationError:
            raise
        except SQLAlchemyError as exc:
            raise DatabaseOperationError(_OPERATION_FAILED) from exc

    def _make_chunk_model(
        self,
        chunk: ChunkRecord,
        vector: np.ndarray,
    ) -> ChunkModel:
        """Create one chunk ORM row; the embedding is never logged or repr'd."""
        return ChunkModel(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            filename=chunk.filename,
            content=chunk.text,
            page_number=chunk.page_number,
            chunk_metadata={},
            embedding=vector.tolist(),
        )

    def _verify_inserted_document(
        self,
        session: Session,
        document_id: str,
        expected_chunks: int,
    ) -> None:
        """Verify the committed row counts match expectations."""
        stored_documents = session.scalar(
            select(func.count())
            .select_from(DocumentModel)
            .where(DocumentModel.document_id == document_id)
        )
        stored_chunks = session.scalar(
            select(func.count())
            .select_from(ChunkModel)
            .where(ChunkModel.document_id == document_id)
        )
        if int(stored_documents or 0) != 1 or int(stored_chunks or 0) != expected_chunks:
            raise DatabaseOperationError(_OPERATION_FAILED)

    def delete_document(self, document_id: str) -> DocumentRecord:
        """Delete one document; stored chunks are removed by ON DELETE CASCADE."""
        try:
            with self._session_factory.begin() as session:
                model = session.get(DocumentModel, document_id)
                if model is None:
                    raise DocumentNotFoundError("文档不存在")
                record = _document_model_to_record(model)
                session.delete(model)
        except DocumentNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise DatabaseOperationError(_OPERATION_FAILED) from exc
        return record

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: object,
        top_k: int,
    ) -> list[dict[str, object]]:
        """Run a real pgvector cosine-distance query inside PostgreSQL."""
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        validated_query = _validate_query_embedding(
            query_embedding,
            self._embedding_dimension,
        )
        query_vector = validated_query.tolist()

        distance = ChunkModel.embedding.cosine_distance(query_vector).label(
            "distance"
        )
        statement = (
            select(
                ChunkModel,
                distance,
            )
            .join(
                DocumentModel,
                ChunkModel.document_id == DocumentModel.document_id,
            )
            .where(DocumentModel.status == "ready")
            .order_by(
                distance.asc(),
                ChunkModel.document_id.asc(),
                ChunkModel.chunk_index.asc(),
            )
            .limit(top_k)
        )
        try:
            with self._session_factory() as session:
                rows = session.execute(statement).all()
        except SQLAlchemyError as exc:
            raise DatabaseOperationError(_OPERATION_FAILED) from exc

        results: list[dict[str, object]] = []
        for rank, (model, distance_value) in enumerate(rows, start=1):
            results.append(
                {
                    "rank": rank,
                    "score": _cosine_score(distance_value),
                    "text": model.content,
                    "chunk_index": model.chunk_index,
                    "document_id": model.document_id,
                    "filename": model.filename,
                    "page_number": model.page_number,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """No-op: the engine is owned and disposed by StorageRuntime."""
