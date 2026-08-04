# 课程 RAG 问答项目

这是一个面向初学者的最小 RAG 项目。它读取本地中文资料，把资料切分成 Chunk，使用 Embedding 模型和余弦相似度检索相关原文，并通过 OpenAI-compatible Chat Completions API 生成有来源标注的回答。除了内置知识库，用户还可以上传 TXT、Markdown 和文本型 PDF，上传成功后内容会立即参与语义检索和问答。

## 当前实现

- 从 `data/knowledge.txt` 读取 UTF-8 文本
- 固定长度、带重叠地切分文本
- 批量生成并归一化文档向量
- 在命令行循环接收问题
- 使用 NumPy 手动计算余弦相似度
- 输出 Top 3、四位小数分数、Chunk 编号和原文
- 使用 pytest 离线测试核心逻辑
- 通过 FastAPI 提供健康检查和 Top-K 检索接口
- 在服务启动时构建一次索引，并在请求之间复用
- 使用 PromptBuilder 组装回答规则、检索来源和用户问题
- 使用 GenerationService 调用可配置的 OpenAI-compatible LLM
- 使用 RagService 编排 Embedding、检索、Prompt 和答案生成
- 使用可配置的相关性阈值判断检索资料是否足以支撑回答
- 资料不足时返回结构化拒答和检索候选，不构建 Prompt、不调用 LLM
- 通过 FastAPI `/ask` 返回答案、来源、模型名和各阶段耗时
- 通过 Vue 3 页面在“智能问答”和“语义检索”之间切换
- 智能问答模式调用 `/api/ask`，展示最终答案、引用来源、模型和耗时
- 语义检索模式继续调用 `/api/search`，展示原始 Top-K 检索结果
- 对问答请求的 422、502、503 和网络错误提供稳定、安全的页面提示
- 通过 Vite `/api` 代理连接浏览器与 FastAPI
- 返回检索耗时、模型名和已索引 Chunk 数量
- 支持上传 TXT、Markdown 和文本型 PDF 文档，上传时同步完成解析与索引构建
- 上传成功后内容立即参与语义检索和问答，无需重启应用
- `/search` 和 `/ask` 的来源包含文件名、文档 ID 和可用的 PDF 页码
- 通过 `POST /documents`、`GET /documents`、`DELETE /documents/{id}` 管理文档
- 上传和删除使用原子索引替换，构建失败时旧索引继续可用
- 内置文档 `knowledge.txt` 不可删除
- 上传文件使用 UUID 存储名，运行时数据目录不进入 Git
- 支持上传大小限制 `MAX_UPLOAD_BYTES`，默认 10 MB
- 通过 Vue 3 页面提供“知识库管理”模式，支持上传、列表和删除文档

## 当前没有实现

本项目仍不支持扫描 PDF 的 OCR、图片识别、Word、PowerPoint、Excel、网页抓取、URL 导入、数据库、向量数据库、对象存储、用户登录与多用户隔离、后台任务队列、流式输出、多轮记忆、LangChain、LangGraph、Reranker、BM25、混合检索或自动评估。本地 JSON 和文件系统只是当前实现，不代表生产级存储方案。

## 项目目录

```text
course-rag/
├── data/
│   ├── knowledge.txt        # 中文示例知识库
│   └── runtime/             # 运行时上传文件与元数据（不进入 Git）
│       ├── uploads/         # UUID 命名的上传文件
│       └── documents.json   # 文档元数据
├── src/
│   ├── api.py                # FastAPI Web 接口
│   ├── loader.py             # 读取文本
│   ├── chunker.py            # 切分文本
│   ├── embedding.py          # 生成向量
│   ├── retriever.py          # 计算相似度并排序
│   ├── knowledge_index.py    # 原子替换的内存索引与来源元数据
│   ├── documents.py          # 文档、页面和 Chunk 领域模型
│   ├── document_loaders.py   # TXT / Markdown / PDF 加载器
│   ├── document_chunker.py   # 带元数据的 Chunk 生成
│   ├── document_repository.py# 本地 JSON 文档元数据仓库
│   ├── ingestion_service.py  # 上传、删除与索引重建编排
│   ├── prompt_builder.py      # 组装课程问答 Prompt
│   ├── generation.py         # 调用 OpenAI-compatible LLM
│   ├── rag_service.py        # 编排完整 RAG 调用链
│   ├── exceptions.py         # RAG 与文档领域异常
│   └── main.py               # 命令行入口
├── tests/                    # 离线单元测试
├── frontend/                 # Vue 3 + Vite + TypeScript 前端
│   ├── src/services/         # Search/Ask/Document HTTP 服务与可选模拟服务
│   ├── src/composables/      # 三种模式的独立状态管理
│   ├── src/components/       # 检索、答案、文档管理与状态组件
│   └── README.md             # 前端运行与学习说明
├── requirements.txt          # Python 依赖
└── README.md                 # 学习说明
```

## 创建虚拟环境

需要 Python 3.11 或兼容的较新 Python 3.x。本项目已在 Python 3.14.6、NumPy 2.5.1、sentence-transformers 5.6.0 和 PyTorch 2.13.0 上实际验证。在 Windows PowerShell 中进入项目根目录：

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

## 配置 LLM

在项目根目录复制 `.env.example` 为本地 `.env`：

```powershell
Copy-Item .env.example .env
```

然后填写实际配置：

```env
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
LLM_TIMEOUT_SECONDS=30
RAG_MIN_RELEVANCE_SCORE=0.35
MAX_UPLOAD_BYTES=10485760
```

FastAPI 启动时会通过 `python-dotenv` 自动读取项目根目录的 `.env`，且不会覆盖已有的进程环境变量。`LLM_API_KEY` 和 `LLM_MODEL` 必填；`LLM_BASE_URL` 默认是 `https://api.openai.com/v1`，`LLM_TIMEOUT_SECONDS` 默认是 `30` 且必须大于 `0`。`.env` 包含密钥，绝不能提交到 Git；`.env.example` 只能保留不含真实值的字段模板。

`MAX_UPLOAD_BYTES` 默认是 `10485760`（10 MB），必须是正整数，可以通过构造参数注入，也可以在环境变量或 `.env` 中配置。配置无效时文档上传返回 503，但语义检索不受影响。

`RAG_MIN_RELEVANCE_SCORE` 默认是 `0.35`，必须是 `[0, 1]` 内的有限数字。最高检索分数大于或等于阈值时才生成答案；低于阈值或没有检索结果时，`/ask` 返回 HTTP 200、`answer_status=insufficient_context`、固定的资料不足说明和原始检索候选，并将生成耗时记为 `0`。`/search` 不受问答阈值影响，始终保留原始 Top-K 结果。`0.35` 只是当前阶段的初始启发式设置，尚未通过评估集优化，后续需要结合评估数据调整。

也可以在当前 PowerShell 会话中设置进程环境变量；这些值优先于 `.env`：

```powershell
$env:LLM_API_KEY="your-local-api-key"
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_MODEL="your-model-name"
$env:LLM_TIMEOUT_SECONDS="30"
```

## 运行程序

请在项目根目录执行：

```powershell
python -m src.main
```

输入 `exit`、`quit` 或直接按回车即可退出。输出中的 `Chunk编号` 从 0 开始，与 Python 列表索引一致。

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

响应会分别报告检索和生成能力是否就绪。例如只配置好检索、尚未配置 LLM 时：

```json
{
  "status": "ok",
  "chunk_count": 8,
  "retrieval_ready": true,
  "generation_ready": false,
  "rag_ready": false,
  "min_relevance_score": null
}
```

检索索引成功初始化后，`status` 保持为 `"ok"`。`generation_ready` 表示 LLM 客户端可用，`rag_ready` 表示 LLM 和相关性阈值都已正确配置；阈值无效不会阻止 `/search`，但 `/ask` 会返回 `503 RAG_NOT_CONFIGURED`。

检索请求：

```powershell
curl.exe -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query":"业务逻辑应该写在哪里？","top_k":3}'
```

`query` 不能为空，`top_k` 默认是 `3` 且必须大于 `0`。接口返回清理后的问题、检索耗时、模型名、已索引 Chunk 数量，以及按相似度降序排列的原文、分数、排名和 Chunk 编号。

问答请求：

```powershell
curl.exe -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"Service 层负责什么？","top_k":3}'
```

`question` 清理后长度必须为 1 到 500，`top_k` 默认是 `3` 且范围为 1 到 10。`/ask` 始终以 HTTP 200 返回 `answered` 或 `insufficient_context` 两种业务状态，并包含 `max_relevance_score`、`relevance_threshold`、检索候选和耗时。只有 `answered` 会调用 LLM；LLM 调用失败返回统一的 `502`，LLM 或 RAG 配置不可用返回安全的 `503`，不会暴露内部配置值或上游完整错误。

`/search` 始终只返回原始检索结果；`/ask` 才会执行 Prompt 构造和 LLM 回答生成。两者的来源结构一致，都包含 `rank`、`score`、`text`、`chunk_index`、`document_id`、`filename` 和可为 `null` 的 `page_number`；PDF 页码从 1 开始，TXT 和 Markdown 的 `page_number` 为 `null`。

### 文档上传与管理接口

`POST /documents` 以 `multipart/form-data` 上传字段 `file`，支持 `.txt`、`.md` 和文本型 `.pdf`，默认最大 10 MB。上传成功返回 201 和文档记录（`document_id`、`filename`、`content_type`、`size_bytes`、`text_length`、`chunk_count`、`created_at`、`is_builtin`、`index_status`），内容同步完成解析、Embedding 和原子索引替换，立即参与 `/search` 和 `/ask`。错误映射：

```text
413 UPLOAD_TOO_LARGE
415 UNSUPPORTED_DOCUMENT_TYPE
422 EMPTY_DOCUMENT
422 DOCUMENT_PARSE_FAILED
500 DOCUMENT_INGESTION_FAILED
503 UPLOAD_NOT_CONFIGURED（MAX_UPLOAD_BYTES 无效）
```

`GET /documents` 返回统一文档列表（内置文档优先，其余按上传时间降序）、`document_count` 和当前索引 `chunk_count`。`DELETE /documents/{document_id}` 删除上传文档并重建索引；内置文档返回 `409 BUILTIN_DOCUMENT_CANNOT_BE_DELETED`，不存在的 ID 返回 `404 DOCUMENT_NOT_FOUND`。删除或上传在任何一步失败时，旧索引和原文档列表保持不变，临时文件会被清理。

文件校验同时使用扩展名、Content-Type 和解析结果，实际磁盘文件名由 UUID 生成，原始文件名只用于展示，不会用于拼接保存路径。上传文件保存在 `data/runtime/uploads/`，元数据保存在 `data/runtime/documents.json`，该目录已加入 `.gitignore`，不会进入 Git。LLM 未配置或相关性阈值无效时，文档上传和语义检索仍可正常使用。

上传端点最多把 `MAX_UPLOAD_BYTES + 1` 字节读入内存，超过限制立即返回 413，不依赖客户端提供的 Content-Length。持久化的 `stored_filename` 在读取元数据时按 UUID 文件名格式校验（32 位十六进制 + `.txt`/`.md`/`.pdf`），路径解析被限制在上传目录内，非法记录不会被读取或删除到目录之外。

应用启动时会验证持久化上传记录：缺失、损坏或无法解析的上传记录不会进入活动索引，也不会继续显示为已就绪文档；无效元数据会被原子清理（相关无效文件按 best-effort 删除，失败不影响启动）。`documents.json` 自身损坏仍会明确报错。Embedding、索引构建等意外内部错误统一返回稳定的 `500 DOCUMENT_INGESTION_FAILED`，不会泄露内部异常或服务器路径。

### 缺少 LLM 配置时

LLM 未配置不会阻止 FastAPI 启动。只要检索依赖初始化成功，接口行为如下：

```text
GET  /health   → 200，generation_ready 为 false
POST /search   → 200，正常语义检索
POST /ask      → 503 LLM_NOT_CONFIGURED
```

配置完整时，`/health` 的 `generation_ready` 为 `true`，`/search` 仍执行原有语义检索，`/ask` 正常生成回答。

## 启动完整 Web 应用

先在项目根目录启动 FastAPI：

```powershell
uvicorn src.api:app --reload
```

再打开另一个 PowerShell，进入 `frontend/` 并启动 Vite：

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

浏览器打开 `http://127.0.0.1:5173`。智能问答模式请求 `/api/ask`，语义检索模式请求 `/api/search`；Vite 会移除 `/api` 前缀并代理到 `http://127.0.0.1:8000`，因此开发环境不需要额外配置 CORS。

配置好根目录 `.env` 中的后端 LLM 环境变量后，智能问答可以调用真实模型。未配置 LLM 时，智能问答会展示 503 提示，但不依赖 LLM 的语义检索和文档上传仍可正常使用。真实 `.env` 不得提交到 Git。

如果只想演示界面、不启动 Python 后端，可以在启动 Vite 前设置：

```powershell
$env:VITE_USE_MOCK_SEARCH="true"
$env:VITE_USE_MOCK_ASK="true"
$env:VITE_USE_MOCK_DOCUMENTS="true"
npm.cmd run dev
```

## 运行测试

```powershell
python -m pytest -v
```

测试通过假模型和手工 NumPy 向量验证逻辑，不会下载真实模型。

前端验证在 `frontend/` 中执行：

```powershell
npm.cmd run test
npm.cmd run type-check
npm.cmd run build
```

## 示例问题

- 业务逻辑应该写在哪一层？
- 哪个模块负责访问数据库？
- 如何将文本表示成向量？

## RAG 数据流

```text
内置 knowledge.txt + 用户上传的 TXT / Markdown / 文本型 PDF
    ↓ 文件校验与文本解析
多个带来源元数据的 Chunk（文档 ID、文件名、页码）
    ↓ Embedding
文档向量矩阵
    ↓ KnowledgeIndex 原子替换（RLock 内一次性交换）
当前索引快照

用户问题
    ↓ Embedding
问题向量
    ↓ 与文档向量计算余弦相似度
Top K 来源（rank/score/text/chunk_index/document_id/filename/page_number）
    ├── `/search`：直接返回检索结果
    ├── `/ask`：比较最高分与相关性阈值
    │     ├── 达到阈值：PromptBuilder → GenerationService → answered
    │     └── 低于阈值/无结果：固定拒答 → insufficient_context
    └── `/documents`：上传/删除 → 构建完整新索引 → 成功后原子替换
```

上传和删除都在内存中完整构建新索引，全部验证成功后一次性替换当前索引；解析、Embedding、元数据写入或索引构建任何一步失败，旧索引和原文档列表继续可用。Vue 前端通过模式切换分别使用 `/ask`、`/search` 和 `/documents` 三个分支；问答和检索模式共用问题输入，知识库管理独立保留自己的状态。

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

## 当前范围之外

当前 Web 应用已经支持单轮、带来源的课程知识问答、可配置相关性阈值、资料不足拒答、独立语义检索，以及 TXT、Markdown、文本型 PDF 的上传、列表、删除和即时索引更新。扫描 PDF 的 OCR、图片识别、Word、PowerPoint、Excel、数据库、向量数据库、pgvector、对象存储、用户登录与多用户隔离、后台任务队列、评估集、多轮记忆、流式输出、LangChain、LangGraph 与 Agent 仍未实现。本地 JSON 与文件系统只是当前阶段的存储实现，没有声称支持生产级并发和扩展。
