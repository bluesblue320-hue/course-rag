"""Manage a replaceable in-memory search index over source metadata."""

import threading

import numpy as np

from src.documents import ChunkRecord
from src.retriever import SemanticRetriever


class KnowledgeIndex:
    """Own the active retriever and replace it atomically after building."""

    def __init__(
        self,
        chunks: list[ChunkRecord] | None = None,
        embeddings: object | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._retriever: SemanticRetriever | None = None
        self._chunks: list[ChunkRecord] = []
        if chunks:
            self.replace(chunks, embeddings)

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the current complete snapshot."""
        with self._lock:
            return len(self._chunks)

    def close(self) -> None:
        """No-op for the storage protocol; the index owns no resources."""

    def search(
        self,
        query_embedding: object,
        top_k: int,
    ) -> list[dict[str, object]]:
        """Search the current snapshot and attach document source metadata."""
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        with self._lock:
            retriever = self._retriever
            chunks = self._chunks

        if retriever is None:
            return []

        results = retriever.search(query_embedding, top_k=top_k)
        return [
            {
                "rank": result["rank"],
                "score": result["score"],
                "text": result["text"],
                "chunk_index": result["chunk_index"],
                "document_id": chunks[result["chunk_index"]].document_id,
                "filename": chunks[result["chunk_index"]].filename,
                "page_number": chunks[result["chunk_index"]].page_number,
            }
            for result in results
        ]

    def replace(
        self,
        chunks: list[ChunkRecord],
        embeddings: object,
    ) -> None:
        """Swap the whole index at once only after the new retriever builds."""
        retriever: SemanticRetriever | None = None
        if chunks:
            retriever = SemanticRetriever(
                [chunk.text for chunk in chunks],
                np.asarray(embeddings, dtype=float),
            )

        with self._lock:
            self._retriever = retriever
            self._chunks = list(chunks)
