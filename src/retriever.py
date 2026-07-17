"""Rank text chunks by cosine similarity using NumPy only."""

import numpy as np

SearchResult = dict[str, int | float | str]


class SemanticRetriever:
    """Store normalized chunk vectors and search them by cosine score."""

    def __init__(self, chunks: list[str], embeddings: np.ndarray) -> None:
        if not chunks:
            raise ValueError("Chunk 列表不能为空")

        array = np.asarray(embeddings, dtype=float)
        if array.ndim != 2:
            raise ValueError("文档向量必须是二维数组")
        if len(chunks) != array.shape[0]:
            raise ValueError("Chunk 数量必须与向量数量一致")

        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("文档向量不能是零向量")

        self._chunks = list(chunks)
        self._embeddings = array / norms

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks in descending similarity order."""
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query = np.asarray(query_embedding, dtype=float)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1:
            raise ValueError("查询向量必须是一维数组或单行二维数组")
        if query.shape[0] != self._embeddings.shape[1]:
            raise ValueError("查询向量维度必须与文档向量一致")

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("查询向量不能是零向量")

        normalized_query = query / query_norm
        scores = self._embeddings @ normalized_query
        result_count = min(top_k, len(self._chunks))
        indices = np.argsort(-scores, kind="stable")[:result_count]

        return [
            {
                "rank": rank,
                "score": float(scores[index]),
                "text": self._chunks[index],
                "chunk_index": int(index),
            }
            for rank, index in enumerate(indices, start=1)
        ]
