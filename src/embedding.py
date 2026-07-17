"""Create normalized text embeddings with sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Return a two-dimensional array with unit-length rows."""
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2:
        raise ValueError("Embedding 模型必须返回二维数组")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding 模型返回了零向量")
    return array / norms


class EmbeddingService:
    """Load one embedding model and expose document/query operations."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode a non-empty document batch as normalized NumPy rows."""
        cleaned_texts = [text.strip() for text in texts]
        if not cleaned_texts or any(not text for text in cleaned_texts):
            raise ValueError("文档文本不能为空")

        vectors = self._model.encode(
            cleaned_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return _normalize_rows(vectors)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one non-empty query as a normalized NumPy vector."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("查询文本不能为空")

        vectors = self._model.encode(
            [cleaned_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return _normalize_rows(vectors)[0]
