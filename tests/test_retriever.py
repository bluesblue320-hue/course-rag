import numpy as np
import pytest

from src.retriever import SemanticRetriever


@pytest.fixture
def retriever() -> SemanticRetriever:
    return SemanticRetriever(
        chunks=["service", "repository", "embedding"],
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
            ]
        ),
    )


def test_search_returns_results_in_descending_score_order(
    retriever: SemanticRetriever,
) -> None:
    results = retriever.search(np.array([1.0, 0.0]), top_k=3)

    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)
    assert [result["rank"] for result in results] == [1, 2, 3]


def test_search_returns_requested_top_k(retriever: SemanticRetriever) -> None:
    results = retriever.search(np.array([1.0, 0.0]), top_k=2)

    assert len(results) == 2


def test_search_caps_top_k_at_chunk_count(retriever: SemanticRetriever) -> None:
    results = retriever.search(np.array([1.0, 0.0]), top_k=10)

    assert len(results) == 3


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_non_positive_top_k(
    retriever: SemanticRetriever,
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="top_k 必须大于 0"):
        retriever.search(np.array([1.0, 0.0]), top_k=top_k)


def test_constructor_rejects_chunk_embedding_count_mismatch() -> None:
    with pytest.raises(ValueError, match="Chunk 数量必须与向量数量一致"):
        SemanticRetriever(
            chunks=["one", "two"],
            embeddings=np.array([[1.0, 0.0]]),
        )


def test_most_relevant_vector_is_ranked_first() -> None:
    retriever = SemanticRetriever(
        chunks=["业务逻辑", "数据库访问"],
        embeddings=np.array([[1.0, 0.0], [0.0, 2.0]]),
    )

    results = retriever.search(np.array([0.0, 3.0]), top_k=1)

    assert results == [
        {
            "rank": 1,
            "score": pytest.approx(1.0),
            "text": "数据库访问",
            "chunk_index": 1,
        }
    ]


def test_constructor_rejects_zero_document_vector() -> None:
    with pytest.raises(ValueError, match="文档向量不能是零向量"):
        SemanticRetriever(
            chunks=["zero"],
            embeddings=np.array([[0.0, 0.0]]),
        )


def test_search_rejects_wrong_query_dimension(retriever: SemanticRetriever) -> None:
    with pytest.raises(ValueError, match="查询向量维度必须与文档向量一致"):
        retriever.search(np.array([1.0, 0.0, 0.0]))


def test_search_rejects_zero_query_vector(retriever: SemanticRetriever) -> None:
    with pytest.raises(ValueError, match="查询向量不能是零向量"):
        retriever.search(np.array([0.0, 0.0]))
