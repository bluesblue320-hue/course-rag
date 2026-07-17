# FastAPI 检索接口设计

## 1. 目标

在现有最小语义检索项目上增加一个独立的 FastAPI Web API，同时保留原有命令行入口。第一版只公开 Top-K 原文检索能力，主要供 Swagger UI 和 curl 学习、调试使用。

本阶段不接入大语言模型生成答案，不增加数据库、向量数据库、CORS、用户系统、LangChain、LangGraph、Agent 或前端。

## 2. 架构

保留 `src/main.py` 作为命令行入口，新增 `src/api.py` 作为 Web API 入口。API 直接复用现有模块：

- `src.loader.load_text`：读取本地知识文件。
- `src.chunker.split_text`：生成文本块。
- `src.embedding.EmbeddingService`：生成文档和查询向量。
- `src.retriever.SemanticRetriever`：返回按相似度降序排列的 Top-K 原文。

FastAPI 应用在启动阶段读取 `data/knowledge.txt`、切块、创建一个 `EmbeddingService`、批量生成文档向量并创建一个 `SemanticRetriever`。应用状态明确保存 `embedding_service`、`retriever` 和 `chunk_count`，在所有请求之间复用，避免每次请求重复加载模型或重建索引。

`src/main.py` 的命令行行为不变。API 不复制检索算法，也不把命令行输入输出逻辑带入 Web 层。

## 3. API 接口

### `GET /health`

用途：确认应用已经完成模型和检索索引初始化。

成功响应状态码为 `200`：

```json
{
  "status": "ok",
  "chunk_count": 3
}
```

`chunk_count` 是当前检索器中的文本块数量。应用只有在启动初始化成功后才接受请求，因此该接口不返回“未就绪”的成功响应。

### `POST /search`

请求体：

```json
{
  "query": "业务逻辑应该写在哪里？",
  "top_k": 3
}
```

字段规则：

- `query` 是必填字符串，清理首尾空白后必须非空。
- `top_k` 是可选整数，默认值为 `3`，必须大于 `0`。
- 当 `top_k` 大于文本块总数时，返回全部文本块，与现有检索器行为一致。

成功响应状态码为 `200`：

```json
{
  "query": "业务逻辑应该写在哪里？",
  "results": [
    {
      "rank": 1,
      "score": 0.82,
      "text": "相关原文",
      "chunk_index": 0
    }
  ]
}
```

响应中的 `query` 使用清理首尾空白后的值。`results` 按 `score` 降序排列；`rank` 从 `1` 开始，`chunk_index` 保持现有检索器从 `0` 开始的索引规则。`score` 保留原始浮点值，不在 API 层格式化成字符串。

## 4. 数据流

启动阶段：

```text
data/knowledge.txt
    → load_text
    → split_text
    → EmbeddingService.encode_documents
    → SemanticRetriever
    → 保存 embedding_service、retriever、chunk_count 到应用状态
```

请求阶段：

```text
POST /search
    → 清理并校验 query、top_k
    → EmbeddingService.encode_query
    → SemanticRetriever.search
    → JSON 响应
```

## 5. 错误处理

- 缺少 `query`、`query` 为空或仅含空白、`top_k` 类型错误、`top_k <= 0`：由请求模型校验并返回 `422 Unprocessable Entity`。
- 知识文件缺失、文件为空、模型无法加载或文档向量无法生成：保留原始异常，使应用启动失败；不启动一个无法检索但仍报告健康的服务。
- 查询编码或检索过程中发生未预期异常：由 FastAPI 返回 `500 Internal Server Error`，不在第一版中把任意内部异常转换成伪造的成功结果。

## 6. 依赖与运行方式

在 `requirements.txt` 中增加：

- `fastapi`
- `uvicorn`
- `httpx`，供 FastAPI/Starlette 测试客户端使用

从项目根目录启动开发服务：

```powershell
uvicorn src.api:app --reload
```

Swagger UI 地址为 `http://127.0.0.1:8000/docs`。

## 7. 测试策略

API 测试使用 FastAPI `TestClient`，并替换真实 `EmbeddingService`，确保测试不会联网、下载模型或依赖本机模型缓存。测试必须覆盖：

- 应用启动时只构建一次文档索引。
- `GET /health` 返回 `status` 和真实文本块数量。
- `POST /search` 使用默认 `top_k=3` 并返回规定的响应结构。
- 自定义 `top_k` 被传给现有检索器。
- 查询字符串在编码和响应前清理首尾空白。
- 缺失查询、空查询、纯空白查询和非正 `top_k` 返回 `422`。
- 测试期间不实例化或调用真实 Sentence Transformer 模型。

现有 Loader、Chunker、Embedding、Retriever 和命令行测试继续保留。最终验证运行完整测试套件：

```powershell
pytest -v
```

## 8. 文档更新

README 的“当前没有实现”部分改为准确说明已经提供 Web API，但仍没有大模型生成、数据库、向量数据库或前端。README 增加依赖安装、API 启动、Swagger UI、健康检查和检索请求示例，同时保留原有命令行使用说明。
