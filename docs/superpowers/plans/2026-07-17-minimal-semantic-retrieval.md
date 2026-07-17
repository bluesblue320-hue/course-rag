# Minimal Semantic Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a beginner-friendly Python command-line project that retrieves the three most semantically similar local text chunks without generating an LLM answer.

**Architecture:** Keep file loading, fixed-window chunking, embedding, NumPy retrieval, and CLI orchestration in five focused modules. Encode document chunks once at startup, normalize all vectors, and answer each query by a manual NumPy dot product followed by stable descending sorting.

**Tech Stack:** Python 3.11+ (current environment: 3.14.6), `sentence-transformers`, `numpy`, `pytest`, and the Python standard library.

## Global Constraints

- Use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` as the default Chinese-capable embedding model.
- Do not use LangChain, LangGraph, FastAPI, PostgreSQL, pgvector, FAISS, Chroma, Redis, Docker, frontend frameworks, LLM APIs, Agent, or Tool Calling.
- Use `pathlib.Path` for filesystem paths and never hardcode an absolute path in application code.
- Every source file has one primary responsibility; keep the implementation direct and beginner-readable.
- Unit tests must not download or load the real embedding model.
- Git commits require repository-local `user.name` and `user.email`; until the user supplies them, run every verification step but leave commit steps pending.

---

### Task 1: Project Foundation and UTF-8 Loader

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/test_loader.py`
- Create: `src/loader.py`

**Interfaces:**
- Consumes: A filesystem path supplied as `str`.
- Produces: `load_text(file_path: str) -> str`, used later by `src.main`.

- [ ] **Step 1: Create the package, dependency, ignore, and failing loader test files**

Create an empty `src/__init__.py`.

Create `requirements.txt`:

```text
numpy
pytest
sentence-transformers
```

Create `.gitignore`:

```gitignore
# Virtual environments
.venv/
venv/
env/

# Python caches and build outputs
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.pytest_cache/
.coverage
htmlcov/

# Model and application caches
.cache/
.huggingface/
models/

# Local configuration and secrets
.env
.env.*
!.env.example

# Editors and operating systems
.idea/
.vscode/
.DS_Store
Thumbs.db
```

Create `tests/test_loader.py`:

```python
from pathlib import Path

import pytest

from src.loader import load_text


def test_load_text_reads_utf8_and_strips_outer_whitespace(tmp_path: Path) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("  业务逻辑属于 service 层。\n", encoding="utf-8")

    result = load_text(str(knowledge_file))

    assert result == "业务逻辑属于 service 层。"


def test_load_text_raises_when_file_does_not_exist(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="文本文件不存在"):
        load_text(str(missing_file))


def test_load_text_raises_when_file_is_empty(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match="文本文件为空"):
        load_text(str(empty_file))
```

- [ ] **Step 2: Run the loader tests and verify the expected RED state**

Run:

```powershell
pytest tests/test_loader.py -v
```

Expected: collection fails because `src.loader` does not exist. This confirms the requested loader behavior has no implementation yet.

- [ ] **Step 3: Implement the minimal loader**

Create `src/loader.py`:

```python
"""Load plain-text knowledge files."""

from pathlib import Path


def load_text(file_path: str) -> str:
    """Read a non-empty UTF-8 text file and trim outer whitespace."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"文本文件不存在：{path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"文本文件为空：{path}")

    return text
```

- [ ] **Step 4: Run the loader tests and verify GREEN**

Run:

```powershell
pytest tests/test_loader.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the loader foundation when Git identity is available**

```powershell
git add .gitignore requirements.txt src/__init__.py src/loader.py tests/test_loader.py
git commit -m "feat: add UTF-8 knowledge loader"
```

---

### Task 2: Fixed-Length Overlapping Chunker

**Files:**
- Create: `tests/test_chunker.py`
- Create: `src/chunker.py`

**Interfaces:**
- Consumes: `text: str`, `chunk_size: int = 300`, and `chunk_overlap: int = 50`.
- Produces: `split_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]`, used later by `src.main`.

- [ ] **Step 1: Write the failing chunker tests**

Create `tests/test_chunker.py`:

```python
import pytest

from src.chunker import split_text


def test_split_text_creates_fixed_length_chunks() -> None:
    result = split_text("abcdefghij", chunk_size=4, chunk_overlap=0)

    assert result == ["abcd", "efgh", "ij"]


def test_split_text_preserves_the_requested_overlap() -> None:
    result = split_text("abcdefghij", chunk_size=4, chunk_overlap=1)

    assert result == ["abcd", "defg", "ghij"]
    assert result[0][-1:] == result[1][:1]
    assert result[1][-1:] == result[2][:1]


def test_split_text_returns_one_chunk_for_short_text() -> None:
    assert split_text("短文本", chunk_size=10, chunk_overlap=2) == ["短文本"]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_split_text_rejects_non_positive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size 必须大于 0"):
        split_text("text", chunk_size=chunk_size, chunk_overlap=0)


def test_split_text_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap 不能小于 0"):
        split_text("text", chunk_size=4, chunk_overlap=-1)


@pytest.mark.parametrize("chunk_overlap", [4, 5])
def test_split_text_rejects_overlap_not_smaller_than_chunk_size(
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError, match="chunk_overlap 必须小于 chunk_size"):
        split_text("text", chunk_size=4, chunk_overlap=chunk_overlap)


def test_split_text_skips_whitespace_only_chunks() -> None:
    result = split_text("abc   xyz", chunk_size=3, chunk_overlap=0)

    assert result == ["abc", "xyz"]


def test_split_text_returns_empty_list_for_whitespace_only_text() -> None:
    assert split_text("   ", chunk_size=2, chunk_overlap=1) == []
```

- [ ] **Step 2: Run the chunker tests and verify the expected RED state**

Run:

```powershell
pytest tests/test_chunker.py -v
```

Expected: collection fails because `src.chunker` does not exist.

- [ ] **Step 3: Implement the sliding-window chunker**

Create `src/chunker.py`:

```python
"""Split text into simple fixed-length overlapping chunks."""


def split_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """Return non-blank chunks produced by a fixed-size sliding window."""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Advancing by less than chunk_size keeps context shared by neighbors.
        start += step

    return chunks
```

- [ ] **Step 4: Run loader and chunker tests and verify GREEN**

Run:

```powershell
pytest tests/test_loader.py tests/test_chunker.py -v
```

Expected: `13 passed`.

- [ ] **Step 5: Commit the chunker when Git identity is available**

```powershell
git add src/chunker.py tests/test_chunker.py
git commit -m "feat: add overlapping text chunker"
```

---

### Task 3: Embedding Service Without Networked Unit Tests

**Files:**
- Create: `tests/test_embedding.py`
- Create: `src/embedding.py`

**Interfaces:**
- Consumes: model name, a non-empty list of non-blank documents, or one non-blank query.
- Produces: `EmbeddingService`, `encode_documents(texts: list[str]) -> np.ndarray`, and `encode_query(query: str) -> np.ndarray`.

- [ ] **Step 1: Write embedding tests with a lightweight fake model**

Create `tests/test_embedding.py`:

```python
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
```

- [ ] **Step 2: Run the embedding tests and verify the expected RED state**

Run:

```powershell
pytest tests/test_embedding.py -v
```

Expected: collection fails because `src.embedding` does not exist.

- [ ] **Step 3: Implement the one-model embedding wrapper**

Create `src/embedding.py`:

```python
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
```

- [ ] **Step 4: Run embedding tests without network access and verify GREEN**

Run:

```powershell
pytest tests/test_embedding.py -v
```

Expected: `6 passed`; no model download occurs because the model constructor is monkeypatched before service construction.

- [ ] **Step 5: Run all current unit tests**

Run:

```powershell
pytest tests/test_loader.py tests/test_chunker.py tests/test_embedding.py -v
```

Expected: `19 passed`.

- [ ] **Step 6: Commit the embedding service when Git identity is available**

```powershell
git add src/embedding.py tests/test_embedding.py
git commit -m "feat: add normalized embedding service"
```

---

### Task 4: Manual NumPy Semantic Retriever

**Files:**
- Create: `tests/test_retriever.py`
- Create: `src/retriever.py`

**Interfaces:**
- Consumes: `chunks: list[str]`, `embeddings: np.ndarray`, a query vector, and `top_k`.
- Produces: `SemanticRetriever.search(query_embedding: np.ndarray, top_k: int = 3) -> list[dict[str, int | float | str]]`.

- [ ] **Step 1: Write failing retriever tests using only hand-built vectors**

Create `tests/test_retriever.py`:

```python
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
```

- [ ] **Step 2: Run the retriever tests and verify the expected RED state**

Run:

```powershell
pytest tests/test_retriever.py -v
```

Expected: collection fails because `src.retriever` does not exist.

- [ ] **Step 3: Implement validation, normalization, dot product, and sorting**

Create `src/retriever.py`:

```python
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
```

- [ ] **Step 4: Run retriever tests and verify GREEN**

Run:

```powershell
pytest tests/test_retriever.py -v
```

Expected: `10 passed`.

- [ ] **Step 5: Run every unit test created so far**

Run:

```powershell
pytest -v
```

Expected: `29 passed`.

- [ ] **Step 6: Commit the retriever when Git identity is available**

```powershell
git add src/retriever.py tests/test_retriever.py
git commit -m "feat: add NumPy semantic retriever"
```

---

### Task 5: CLI Pipeline and Chinese Knowledge Base

**Files:**
- Create: `tests/test_main.py`
- Create: `data/knowledge.txt`
- Create: `src/main.py`

**Interfaces:**
- Consumes: `data/knowledge.txt` and repeated standard-input questions.
- Produces: `main() -> None`; each query prints Top 3 rank, four-decimal score, zero-based chunk index, and source text.

- [ ] **Step 1: Write failing offline CLI tests**

Create `tests/test_main.py`:

```python
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import src.main as main_module


class FakeEmbeddingService:
    query_count = 0

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))

    def encode_query(self, query: str) -> np.ndarray:
        type(self).query_count += 1
        return np.array([1.0, 0.0])


def _set_inputs(
    monkeypatch: pytest.MonkeyPatch,
    values: list[str],
) -> None:
    inputs: Iterator[str] = iter(values)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))


def test_main_loads_chunks_searches_and_formats_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("service 层负责业务逻辑。", encoding="utf-8")
    monkeypatch.setattr(main_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(main_module, "EmbeddingService", FakeEmbeddingService)
    _set_inputs(monkeypatch, ["业务逻辑在哪里？", "exit"])

    main_module.main()

    output = capsys.readouterr().out
    assert "成功加载 1 个 Chunk" in output
    assert "Top 1" in output
    assert "相似度：1.0000" in output
    assert "Chunk编号：0" in output
    assert "原文：service 层负责业务逻辑。" in output
    assert "程序已退出。" in output


@pytest.mark.parametrize("exit_value", ["", "exit", "QUIT"])
def test_main_exits_without_encoding_a_query(
    exit_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("repository 层访问数据库。", encoding="utf-8")
    monkeypatch.setattr(main_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(main_module, "EmbeddingService", FakeEmbeddingService)
    FakeEmbeddingService.query_count = 0
    _set_inputs(monkeypatch, [exit_value])

    main_module.main()

    assert FakeEmbeddingService.query_count == 0
```

- [ ] **Step 2: Run the CLI tests and verify the expected RED state**

Run:

```powershell
pytest tests/test_main.py -v
```

Expected: collection fails because `src.main` does not exist.

- [ ] **Step 3: Create the Chinese example knowledge base**

Create `data/knowledge.txt`:

```text
FastAPI 是一个用于构建 Python Web API 的框架。一个清晰的后端项目通常会把接收请求、处理业务规则和访问数据拆分到不同层，这样代码更容易阅读和测试。

router 层负责接收 HTTP 请求、读取路径参数或请求体，并调用 service 层。router 应保持轻量，不应该堆放复杂业务规则。

service 层负责业务逻辑，例如校验订单状态、计算价格、决定工作流程以及协调多个数据操作。如果有人问“业务逻辑应该写在哪里”，通常答案是 service 层。

repository 层负责数据访问。它封装查询、新增、修改和删除操作，使 service 层不需要了解具体 SQL。需要访问数据库时，应由 repository 模块承担这项职责。

PostgreSQL 是关系型数据库，可以持久化用户、课程、订单等结构化数据。在分层项目中，repository 层会与 PostgreSQL 交互，而 router 层不会直接执行数据库查询。

Embedding 是把文本转换成一组数字向量的过程。语义相近的句子会得到方向相近的向量，因此即使用词不同，也可以通过向量相似度找到相关内容。

余弦相似度衡量两个向量方向有多接近。向量先归一化后，可以直接使用点积得到余弦相似度。分数越高，通常表示两段文本的语义越相关。

RAG 是检索增强生成。完整流程会先把资料切分成 Chunk 并生成 Embedding，再根据问题检索 Top K 相关 Chunk，最后把检索结果交给大语言模型生成答案。本项目只实现检索，不执行最后的生成步骤。
```

- [ ] **Step 4: Implement the interactive CLI**

Create `src/main.py`:

```python
"""Run the minimal semantic retrieval command-line program."""

from pathlib import Path

from src.chunker import split_text
from src.embedding import EmbeddingService
from src.loader import load_text
from src.retriever import SearchResult, SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge.txt"


def _print_results(results: list[SearchResult]) -> None:
    """Print ranked retrieval results in a beginner-readable format."""
    for result in results:
        print(f"\nTop {result['rank']}")
        print(f"相似度：{float(result['score']):.4f}")
        print(f"Chunk编号：{result['chunk_index']}")
        print(f"原文：{result['text']}")


def main() -> None:
    """Build the document index once, then search until the user exits."""
    text = load_text(str(KNOWLEDGE_PATH))
    chunks = split_text(text)
    embedding_service = EmbeddingService()
    document_embeddings = embedding_service.encode_documents(chunks)
    retriever = SemanticRetriever(chunks, document_embeddings)

    print(f"成功加载 {len(chunks)} 个 Chunk。")

    while True:
        query = input("\n请输入问题（输入 exit/quit 或直接回车退出）：").strip()
        if not query or query.lower() in {"exit", "quit"}:
            print("程序已退出。")
            break

        query_embedding = embedding_service.encode_query(query)
        results = retriever.search(query_embedding, top_k=3)
        _print_results(results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run offline CLI tests and verify GREEN**

Run:

```powershell
pytest tests/test_main.py -v
```

Expected: `4 passed` and no real model download.

- [ ] **Step 6: Run the complete offline suite**

Run:

```powershell
pytest -v
```

Expected: `33 passed`.

- [ ] **Step 7: Commit the CLI and example data when Git identity is available**

```powershell
git add data/knowledge.txt src/main.py tests/test_main.py
git commit -m "feat: add semantic retrieval CLI"
```

---

### Task 6: Chinese Learning Guide and End-to-End Verification

**Files:**
- Create: `README.md`
- Modify only if verification exposes a defect: source or test file responsible for that defect

**Interfaces:**
- Consumes: the completed project and local Python interpreters.
- Produces: beginner setup documentation, a passing full suite, and captured real-model smoke-test output for the three required questions.

- [ ] **Step 1: Write the complete Chinese README**

Create `README.md` with these exact sections and commands:

```markdown
# 最小语义检索项目

这是一个面向初学者的最小 RAG 检索项目。它读取本地中文资料，把资料切分成 Chunk，使用 Embedding 模型把文本转换为向量，再通过余弦相似度找出与用户问题最相关的三个 Chunk。

## 当前实现

- 从 `data/knowledge.txt` 读取 UTF-8 文本
- 固定长度、带重叠地切分文本
- 批量生成并归一化文档向量
- 在命令行循环接收问题
- 使用 NumPy 手动计算余弦相似度
- 输出 Top 3、四位小数分数、Chunk 编号和原文
- 使用 pytest 离线测试核心逻辑

## 当前没有实现

本阶段没有调用大语言模型生成答案，也没有 Web API、数据库、向量数据库、LangChain、LangGraph、Agent 或前端。程序只负责检索原文。

## 项目目录

```text
course-rag/
├── data/knowledge.txt        # 中文示例知识库
├── src/
│   ├── loader.py             # 读取文本
│   ├── chunker.py            # 切分文本
│   ├── embedding.py          # 生成向量
│   ├── retriever.py          # 计算相似度并排序
│   └── main.py               # 命令行入口
├── tests/                    # 离线单元测试
├── requirements.txt          # Python 依赖
└── README.md                 # 学习说明
```

## 创建虚拟环境

需要 Python 3.11 或兼容的较新 Python 3.x。在 Windows PowerShell 中进入项目根目录：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，可以不激活，直接使用 `.\.venv\Scripts\python.exe` 运行后续命令；也可以只为当前窗口调整策略：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 安装依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

第一次实际运行会从 Hugging Face 下载默认模型，需要网络连接和一定磁盘空间。后续运行通常会使用本地缓存。

## 运行程序

请在项目根目录执行：

```powershell
python -m src.main
```

输入 `exit`、`quit` 或直接按回车即可退出。

## 运行测试

```powershell
pytest -v
```

测试通过假模型和手工 NumPy 向量验证逻辑，不会下载真实模型。

## 示例问题

- 业务逻辑应该写在哪一层？
- 哪个模块负责访问数据库？
- 如何将文本表示成向量？

## RAG 数据流

```text
knowledge.txt
    ↓ 读取
完整文本
    ↓ Chunk 切分
多个文本块
    ↓ Embedding
文档向量矩阵

用户问题 → 问题向量
              ↓
文档向量与问题向量计算余弦相似度
              ↓
按分数降序排列并返回 Top K 原文
```

完整 RAG 还会把 Top K 原文和问题一起交给大语言模型。本项目故意停在检索结果处，便于先理解基础数据流。

## 四个核心概念

### Chunk

Chunk 是从长文本中切出来的小段。本项目使用固定字符数切分，并让相邻 Chunk 重叠一部分内容，避免重要语义恰好被边界切断。

### Embedding

Embedding 是文本对应的数字向量。语义相近的文本通常会在向量空间中靠得更近。本项目默认使用支持中文的 `paraphrase-multilingual-MiniLM-L12-v2`。

### 余弦相似度

余弦相似度比较两个向量的方向。向量归一化后，NumPy 点积就是余弦相似度。分数越高，表示问题与 Chunk 通常越相关。

### Top K

Top K 表示只保留分数最高的 K 条结果。本项目默认取 Top 3；如果知识库不足三个 Chunk，就返回全部 Chunk。

## 常见错误

### 找不到文本文件

确认 `data/knowledge.txt` 存在，并从项目根目录使用 `python -m src.main` 启动。

### 文本文件为空

在 `knowledge.txt` 中写入非空 UTF-8 文本。只有空格或换行也会被视为空文件。

### 无法下载模型

检查网络、代理、防火墙和 Hugging Face 是否可访问。模型下载错误不会被程序静默忽略。

### Python 或 PyTorch 版本不兼容

`sentence-transformers` 依赖 PyTorch。如果最新 Python 暂时没有可用的 PyTorch 安装包，请安装一个受支持的 Python 3.x（例如 Python 3.11、3.12 或 3.13），用它重新创建 `.venv`。

### PowerShell 无法激活虚拟环境

可以使用前文的进程级执行策略命令，或直接运行 `.\.venv\Scripts\python.exe`，无需激活。

## 下一阶段

理解当前项目后，可以增加一个生成模块：把用户问题和检索出的 Top K 原文组装成提示词，再交给大语言模型生成有依据的回答。检索器、Chunk 和 Embedding 逻辑仍可复用。加入生成前，应先理解为什么检索结果会影响答案质量。
```

- [ ] **Step 2: Detect installed interpreters and create a compatible virtual environment**

Run:

```powershell
py -0p
python -m venv .venv
```

Expected: interpreter list is printed and `.venv` is created. If Python 3.14 cannot install the ML stack, remove only the newly created `.venv` after verifying its resolved path is inside the project, then recreate it with an installed compatible interpreter such as `py -3.13 -m venv .venv` or `py -3.12 -m venv .venv`.

- [ ] **Step 3: Install dependencies inside the virtual environment**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: both commands exit 0 and install `numpy`, `pytest`, `sentence-transformers`, and their compatible dependencies. If installation fails, capture the exact resolver error before choosing a different installed Python 3.x.

- [ ] **Step 4: Run the required fresh full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected: `33 passed`, zero failures, zero errors.

- [ ] **Step 5: Run the three-question real-model smoke test**

Run from the project root:

```powershell
@(
    "业务逻辑应该写在哪一层？",
    "哪个模块负责访问数据库？",
    "如何将文本表示成向量？",
    "exit"
) | .\.venv\Scripts\python.exe -m src.main
```

Expected: the model loads, the CLI reports the Chunk count, each question produces up to three results with four-decimal scores and source text, and the program exits normally. Preserve the actual output for final reporting rather than replacing it with an invented example.

- [ ] **Step 6: Re-run the full suite after any smoke-test fix**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected: `33 passed`, zero failures, zero errors. If a defect was found, first add or adjust a test that fails for that defect, verify RED, apply the minimal fix, and then verify GREEN.

- [ ] **Step 7: Verify repository scope and commit documentation when identity is available**

Run:

```powershell
git status --short
git diff --check
git add README.md docs/superpowers/plans/2026-07-17-minimal-semantic-retrieval.md
git commit -m "docs: add semantic retrieval learning guide"
```

Expected: only project files are listed; `git diff --check` reports no whitespace errors; the commit succeeds after repository-local identity is configured.
