# FastAPI 检索接口实施计划

> **供智能代理执行：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项实施本计划。所有步骤使用复选框（`- [ ]`）跟踪进度。

**目标：** 在保留现有命令行程序的同时，增加一个最小 FastAPI 后端；服务只构建一次语义索引，并提供健康检查与 Top-K 检索接口。

**架构：** 保持 `src/main.py` 不变，新增 `src/api.py` 作为独立的 FastAPI 入口。应用在生命周期启动阶段加载知识文件、切分文本、创建一个 Embedding 服务和一个检索器，并将这些对象及 Chunk 数量保存到 `app.state`；后续请求复用这些状态。

**技术栈：** Python 3.11+、FastAPI、Pydantic、Uvicorn、HTTPX/TestClient、NumPy、sentence-transformers、pytest

## 全局约束

- 第一版 API 只返回 Top-K 原文，不调用大语言模型生成答案。
- 不增加数据库、向量数据库、CORS、用户系统、LangChain、LangGraph、Agent、前端或其他无关子系统。
- 保留 `src/main.py` 作为现有命令行入口，其行为不得改变。
- 应用启动时只初始化一次文档索引，后续请求必须复用该索引。
- 知识文件或 Embedding 模型初始化失败时，保留原始启动异常。
- API 测试不得访问网络、下载模型或实例化真实的 Sentence Transformer 模型。
- 保留无关的已暂存和未跟踪内容；提交时只能指定本任务涉及的文件路径。

---

### Task 1：FastAPI 应用启动与健康检查接口

**文件：**
- 修改：`requirements.txt:1-3`
- 新建：`src/api.py`
- 新建：`tests/test_api.py`

**接口：**
- 使用：`load_text(file_path: str) -> str`、`split_text(text: str) -> list[str]`、`EmbeddingService.encode_documents(texts: list[str]) -> np.ndarray`、`SemanticRetriever(chunks, embeddings)`
- 产出：`src.api.app: FastAPI`，应用状态字段 `embedding_service`、`retriever`、`chunk_count`，以及 `GET /health -> {"status": "ok", "chunk_count": int}`

- [ ] **步骤 1：添加 Web 与 API 测试依赖**

将 `requirements.txt` 替换为：

```text
numpy
pytest
sentence-transformers
fastapi
uvicorn
httpx
```

- [ ] **步骤 2：在项目虚拟环境中安装依赖**

运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

预期：退出码为 `0`，成功安装 FastAPI、Uvicorn 和 HTTPX。沙箱环境下载软件包时可能需要明确授权。

- [ ] **步骤 3：先编写失败的启动与健康检查测试**

新建 `tests/test_api.py`：

```python
import importlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


def test_app_builds_index_once_and_health_reports_chunk_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_module = importlib.import_module("src.api")

    class FakeEmbeddingService:
        init_count = 0
        document_encode_count = 0

        def __init__(self) -> None:
            type(self).init_count += 1

        def encode_documents(self, texts: list[str]) -> np.ndarray:
            type(self).document_encode_count += 1
            return np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))

        def encode_query(self, query: str) -> np.ndarray:
            return np.array([1.0, 0.0])

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )

    with TestClient(api_module.app) as client:
        first_response = client.get("/health")
        second_response = client.get("/health")

    assert first_response.status_code == 200
    assert first_response.json() == {"status": "ok", "chunk_count": 4}
    assert second_response.status_code == 200
    assert FakeEmbeddingService.init_count == 1
    assert FakeEmbeddingService.document_encode_count == 1
```

- [ ] **步骤 4：运行测试并确认功能尚不存在**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py::test_app_builds_index_once_and_health_reports_chunk_count -v
```

预期：测试内部显示 `FAILED`，错误为 `ModuleNotFoundError: No module named 'src.api'`，因为 API 模块尚未创建。

- [ ] **步骤 5：实现启动初始化与健康检查接口**

新建 `src/api.py`：

```python
"""Expose semantic retrieval through a minimal FastAPI application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.chunker import split_text
from src.embedding import EmbeddingService
from src.loader import load_text
from src.retriever import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge.txt"


def initialize_search(app: FastAPI) -> None:
    """Build the semantic index and store reusable objects on the app."""
    text = load_text(str(KNOWLEDGE_PATH))
    chunks = split_text(text)
    embedding_service = EmbeddingService()
    document_embeddings = embedding_service.encode_documents(chunks)
    retriever = SemanticRetriever(chunks, document_embeddings)

    app.state.embedding_service = embedding_service
    app.state.retriever = retriever
    app.state.chunk_count = len(chunks)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the search index before accepting requests."""
    initialize_search(app)
    yield


class HealthResponse(BaseModel):
    """Report that the API and its semantic index are ready."""

    status: str
    chunk_count: int


app = FastAPI(
    title="Course RAG Retrieval API",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return readiness and the number of indexed chunks."""
    return HealthResponse(
        status="ok",
        chunk_count=request.app.state.chunk_count,
    )
```

- [ ] **步骤 6：运行健康检查测试和现有完整测试套件**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py::test_app_builds_index_once_and_health_reports_chunk_count -v
.\.venv\Scripts\python.exe -m pytest -v
```

预期：目标测试通过；现有完整测试套件也通过，且不会下载模型。

- [ ] **步骤 7：只提交任务 1 涉及的文件**

运行：

```powershell
git add requirements.txt src/api.py tests/test_api.py
git commit --only -m "feat: initialize FastAPI retrieval service" -- requirements.txt src/api.py tests/test_api.py
```

预期：提交中只包含上述三个路径；无关的已暂存和未跟踪文件保持不变。


---

### Task 2：成功检索与类型化响应

**文件：**
- 修改：`tests/test_api.py`
- 修改：`src/api.py`

**接口：**
- 使用：应用状态中的 `embedding_service` 和 `retriever`、`EmbeddingService.encode_query(query: str) -> np.ndarray`、`SemanticRetriever.search(query_embedding, top_k: int) -> list[SearchResult]`
- 产出：`SearchRequest(query: str, top_k: int = 3)`、`SearchResultResponse`、`SearchResponse` 和 `POST /search`

- [ ] **步骤 1：把健康检查测试的准备逻辑重构为可复用的离线 API fixture，并添加成功检索测试**

将 `tests/test_api.py` 替换为：

```python
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api as api_module


class FakeEmbeddingService:
    init_count = 0
    document_encode_count = 0
    queries: list[str] = []

    def __init__(self) -> None:
        type(self).init_count += 1

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        type(self).document_encode_count += 1
        return np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))

    def encode_query(self, query: str) -> np.ndarray:
        type(self).queries.append(query)
        return np.array([1.0, 0.0])


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    FakeEmbeddingService.init_count = 0
    FakeEmbeddingService.document_encode_count = 0
    FakeEmbeddingService.queries = []

    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_app_builds_index_once_and_health_reports_chunk_count(
    client: TestClient,
) -> None:
    first_response = client.get("/health")
    second_response = client.get("/health")

    assert first_response.status_code == 200
    assert first_response.json() == {"status": "ok", "chunk_count": 4}
    assert second_response.status_code == 200
    assert FakeEmbeddingService.init_count == 1
    assert FakeEmbeddingService.document_encode_count == 1


def test_search_uses_default_top_k_and_returns_typed_results(
    client: TestClient,
) -> None:
    response = client.post(
        "/search",
        json={"query": "  业务逻辑应该写在哪里？  "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "业务逻辑应该写在哪里？"
    assert len(body["results"]) == 3
    assert body["results"][0] == {
        "rank": 1,
        "score": 1.0,
        "text": "A" * 300,
        "chunk_index": 0,
    }
    assert FakeEmbeddingService.queries == ["业务逻辑应该写在哪里？"]


def test_search_honors_custom_top_k(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={"query": "repository", "top_k": 1},
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
```

- [ ] **步骤 2：运行成功检索测试并确认其失败**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py::test_search_uses_default_top_k_and_returns_typed_results tests/test_api.py::test_search_honors_custom_top_k -v
```

预期：两个测试都失败，因为 `POST /search` 此时返回 `404 Not Found`。

- [ ] **步骤 3：添加请求模型、响应模型和检索处理函数**

将 `src/api.py` 替换为：

```python
"""Expose semantic retrieval through a minimal FastAPI application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.chunker import split_text
from src.embedding import EmbeddingService
from src.loader import load_text
from src.retriever import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge.txt"


def initialize_search(app: FastAPI) -> None:
    """Build the semantic index and store reusable objects on the app."""
    text = load_text(str(KNOWLEDGE_PATH))
    chunks = split_text(text)
    embedding_service = EmbeddingService()
    document_embeddings = embedding_service.encode_documents(chunks)
    retriever = SemanticRetriever(chunks, document_embeddings)

    app.state.embedding_service = embedding_service
    app.state.retriever = retriever
    app.state.chunk_count = len(chunks)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the search index before accepting requests."""
    initialize_search(app)
    yield


class HealthResponse(BaseModel):
    """Report that the API and its semantic index are ready."""

    status: str
    chunk_count: int


class SearchRequest(BaseModel):
    """Describe one semantic retrieval request."""

    query: str
    top_k: int = 3


class SearchResultResponse(BaseModel):
    """Describe one ranked source chunk."""

    rank: int
    score: float
    text: str
    chunk_index: int


class SearchResponse(BaseModel):
    """Return the cleaned query and ranked source chunks."""

    query: str
    results: list[SearchResultResponse]


app = FastAPI(
    title="Course RAG Retrieval API",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return readiness and the number of indexed chunks."""
    return HealthResponse(
        status="ok",
        chunk_count=request.app.state.chunk_count,
    )


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Encode one query and return its most relevant source chunks."""
    cleaned_query = payload.query.strip()
    query_embedding = request.app.state.embedding_service.encode_query(
        cleaned_query
    )
    results = request.app.state.retriever.search(
        query_embedding,
        top_k=payload.top_k,
    )
    return SearchResponse(
        query=cleaned_query,
        results=[SearchResultResponse(**result) for result in results],
    )
```

- [ ] **步骤 4：运行 API 测试并确认成功检索测试通过**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

预期：三个测试全部通过，并且不会加载真实模型。

- [ ] **步骤 5：只提交检索接口相关改动**

运行：

```powershell
git add src/api.py tests/test_api.py
git commit --only -m "feat: add semantic search endpoint" -- src/api.py tests/test_api.py
```

预期：提交中只包含 `src/api.py` 和 `tests/test_api.py`。


---

### Task 3：请求参数校验

**文件：**
- 修改：`tests/test_api.py`
- 修改：`src/api.py`

**接口：**
- 使用：`SearchRequest`
- 产出：清理首尾空白且非空的 `query`、正整数 `top_k`，以及非法请求体对应的 FastAPI `422` 响应

- [ ] **步骤 1：添加失败的请求校验测试**

在 `tests/test_api.py` 末尾添加：

```python
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "valid", "top_k": 0},
        {"query": "valid", "top_k": -1},
    ],
)
def test_search_rejects_invalid_request_bodies(
    client: TestClient,
    payload: dict[str, object],
) -> None:
    response = client.post("/search", json=payload)

    assert response.status_code == 422
```

- [ ] **步骤 2：运行校验测试并确认尚未支持的用例失败**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py::test_search_rejects_invalid_request_bodies -v
```

预期：缺少 `query` 的用例已经返回 `422`；空查询仍错误地返回 `200`，非正数 `top_k` 也会继续进入检索器而不是返回 `422`，因此参数化测试中的这些未支持用例失败。

- [ ] **步骤 3：添加 Pydantic 校验并直接使用已校验的查询文本**

在 `src/api.py` 中，将 Pydantic 导入替换为：

```python
from pydantic import BaseModel, Field, field_validator
```

将 `SearchRequest` 替换为：

```python
class SearchRequest(BaseModel):
    """Describe one validated semantic retrieval request."""

    query: str
    top_k: int = Field(default=3, gt=0)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        """Trim the query and reject blank text."""
        cleaned_query = value.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空")
        return cleaned_query
```

在 `search` 处理函数中，将：

```python
    cleaned_query = payload.query.strip()
    query_embedding = request.app.state.embedding_service.encode_query(
        cleaned_query
    )
```

替换为：

```python
    query_embedding = request.app.state.embedding_service.encode_query(
        payload.query
    )
```

再将：

```python
        query=cleaned_query,
```

替换为：

```python
        query=payload.query,
```

- [ ] **步骤 4：运行参数校验、API 和完整回归测试**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py::test_search_rejects_invalid_request_bodies -v
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v
.\.venv\Scripts\python.exe -m pytest -v
```

预期：所有非法请求用例都以 `422` 通过；全部 API 测试和原有测试均通过，且不会下载模型。

- [ ] **步骤 5：只提交参数校验相关改动**

运行：

```powershell
git add src/api.py tests/test_api.py
git commit --only -m "feat: validate retrieval requests" -- src/api.py tests/test_api.py
```

预期：提交中只包含参数校验及其测试改动。


---

### Task 4：API 文档与最终验证

**文件：**
- 修改：`README.md:3-76`

**接口：**
- 使用：`src.api:app`、`GET /health`、`POST /search`
- 产出：面向初学者的依赖安装、服务启动、Swagger、curl 和功能范围说明

- [ ] **步骤 1：确认 README 尚未包含 API 文档**

运行：

```powershell
rg -n "uvicorn src\.api:app|GET /health|POST /search|127\.0\.0\.1:8000/docs" README.md
```

预期：没有匹配内容，命令以非零退出码结束。

- [ ] **步骤 2：更新“当前实现”和“当前没有实现”范围说明**

在 `README.md` 的 `## 当前实现` 下添加：

```markdown
- 通过 FastAPI 提供健康检查和 Top-K 检索接口
- 在服务启动时构建一次索引，并在请求之间复用
```

将 `## 当前没有实现` 下的段落替换为：

```markdown
本阶段没有调用大语言模型生成答案，也没有数据库、向量数据库、LangChain、LangGraph、Agent 或前端。命令行和 Web API 都只负责返回检索原文。
```

在项目目录树的 `src/` 下添加：

```text
│   ├── api.py                # FastAPI Web 接口
```

- [ ] **步骤 3：添加 API 启动与请求示例**

在现有 `## 运行程序` 命令行章节之后添加：

````markdown
## 启动 FastAPI

在项目根目录执行：

```powershell
uvicorn src.api:app --reload
```

服务启动后可以打开 Swagger UI：`http://127.0.0.1:8000/docs`。

健康检查：

```powershell
curl.exe http://127.0.0.1:8000/health
```

检索请求：

```powershell
curl.exe -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query":"业务逻辑应该写在哪里？","top_k":3}'
```

`query` 不能为空，`top_k` 默认是 `3` 且必须大于 `0`。接口返回按相似度降序排列的原文、分数、排名和 Chunk 编号。
````

- [ ] **步骤 4：验证文档、OpenAPI 路由和完整测试套件**

运行：

```powershell
rg -n "uvicorn src\.api:app|/health|/search|127\.0\.0\.1:8000/docs" README.md
.\.venv\Scripts\python.exe -c "from src.api import app; print(sorted(route.path for route in app.routes))"
.\.venv\Scripts\python.exe -m pytest -v
```

预期：

- README 匹配结果包含启动命令、两个接口和 Swagger 地址。
- 路由列表包含 `/docs`、`/health`、`/openapi.json`、`/redoc` 和 `/search`。
- 完整测试套件通过，不下载模型，也不引入新的错误或警告。

- [ ] **步骤 5：只提交文档改动**

运行：

```powershell
git add README.md
git commit --only -m "docs: explain FastAPI retrieval service" -- README.md
```

预期：提交中只包含 `README.md`；无关的已暂存和未跟踪文件保持不变。
