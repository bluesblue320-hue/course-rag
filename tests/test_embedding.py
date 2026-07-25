from collections.abc import Callable

import numpy as np
import pytest

import src.embedding as embedding_module
from src.embedding import EmbeddingService


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.calls.append(
            {
                "texts": texts,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return np.tile(np.array([[3.0, 4.0]]), (len(texts), 1))


@pytest.fixture
def fake_model_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[FakeSentenceTransformer], Callable[[str], FakeSentenceTransformer]]:
    created_models: list[FakeSentenceTransformer] = []

    def factory(model_name: str) -> FakeSentenceTransformer:
        model = FakeSentenceTransformer(model_name)
        created_models.append(model)
        return model

    monkeypatch.setattr(embedding_module, "SentenceTransformer", factory)
    return created_models, factory


def test_embedding_service_loads_model_once_and_batches_documents(
    fake_model_factory: tuple[
        list[FakeSentenceTransformer],
        Callable[[str], FakeSentenceTransformer],
    ],
) -> None:
    created_models, _ = fake_model_factory
    service = EmbeddingService("example/model")

    result = service.encode_documents(["第一段", "第二段"])

    assert len(created_models) == 1
    assert service.model_name == "example/model"
    assert created_models[0].model_name == "example/model"
    assert created_models[0].calls == [
        {
            "texts": ["第一段", "第二段"],
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
    ]
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.6, 0.8]])


def test_encode_query_returns_one_normalized_numpy_vector(
    fake_model_factory: tuple[
        list[FakeSentenceTransformer],
        Callable[[str], FakeSentenceTransformer],
    ],
) -> None:
    created_models, _ = fake_model_factory
    service = EmbeddingService()

    result = service.encode_query("  如何转换为向量？  ")

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [0.6, 0.8])
    assert created_models[0].calls[0]["texts"] == ["如何转换为向量？"]


@pytest.mark.parametrize("texts", [[], ["有效文本", "   "]])
def test_encode_documents_rejects_empty_input(
    texts: list[str],
    fake_model_factory: tuple[
        list[FakeSentenceTransformer],
        Callable[[str], FakeSentenceTransformer],
    ],
) -> None:
    service = EmbeddingService()

    with pytest.raises(ValueError, match="文档文本不能为空"):
        service.encode_documents(texts)


@pytest.mark.parametrize("query", ["", "   "])
def test_encode_query_rejects_empty_input(
    query: str,
    fake_model_factory: tuple[
        list[FakeSentenceTransformer],
        Callable[[str], FakeSentenceTransformer],
    ],
) -> None:
    service = EmbeddingService()

    with pytest.raises(ValueError, match="查询文本不能为空"):
        service.encode_query(query)
