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
- 提供离线检索与拒答评估基准：受控语料、60 题标注数据集和可复现的评估脚本
- 通过 `python -m scripts.evaluate_rag` 输出 Hit@K、Recall@K、MRR 与回答/拒答混淆矩阵
- 在 calibration split 上扫描相似度阈值并给出确定性的推荐值，不自动改写生产配置
- 提供基于人工标注事实与引用的确定性回答质量评估：`python -m scripts.evaluate_answers` 离线评分已有回答，或显式 `--live` 调用生产 RAG 管线
- 回答评估覆盖标注事实覆盖率、来源支持引用覆盖率、`[来源N]` 引用合法率、已知矛盾短语检测与严格标注通过率；真实 LLM 评估必须显式使用 `--live`，CI 只使用确定性 Fixture
- 提供 FastAPI、Vue 构建、Nginx 反向代理、具名卷和健康检查组成的 Docker Compose 一键启动方案
- 提供 PostgreSQL + pgvector 存储后端：`VECTOR_STORE_BACKEND=pgvector` 时由 PostgreSQL 持久化文档元数据、Chunk 文本与 `VECTOR(384)` Embedding，检索在数据库内执行余弦相似度查询
- `VECTOR_STORE_BACKEND=memory` 仍是默认后端：现有 `documents.json` 与内存 `KnowledgeIndex` 继续负责上传、删除、检索和问答，完全不连接 PostgreSQL
- 提供 production-oriented single-host 部署：Docker Compose 三文件组合、PostgreSQL + pgvector、Caddy 自动 HTTPS、仅暴露 80/443、具名卷持久化、一次性 Alembic 迁移服务，详见 [docs/deployment.md](docs/deployment.md)

## 当前没有实现

本项目仍不支持扫描 PDF 的 OCR、图片识别、Word、PowerPoint、Excel、网页抓取、URL 导入、对象存储、用户登录与多用户隔离、后台任务队列、流式输出、多轮记忆、LangChain、LangGraph、BM25 或混合检索。PostgreSQL + pgvector 后端已经可用，但默认仍使用内存索引和 JSON 元数据；`pgvector` 模式不会自动迁移 `documents.json` 中的历史数据，也不会创建 HNSW / IVFFlat 等 ANN 索引。现有评估覆盖检索质量、拒答决策，以及基于人工标注事实与引用的回答质量；回答评估是标注级确定性检查，不是完整忠实度检测，不使用 LLM Judge，不会自动修改 Prompt 或阈值，真实 LLM 评估必须显式使用 `--live`，CI 只运行确定性 Fixture。本地 JSON 和文件系统只是默认实现，不代表生产级存储方案。

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
│   ├── ingestion_service.py  # 上传、删除与索引重建编排（memory 运行时）
│   ├── pgvector_ingestion_service.py  # pgvector 运行时：文件、切分、Embedding 与事务编排
│   ├── storage/              # 存储抽象：协议、运行时装配、PgVectorStore
│   ├── prompt_builder.py      # 组装课程问答 Prompt
│   ├── generation.py         # 调用 OpenAI-compatible LLM
│   ├── rag_service.py        # 编排完整 RAG 调用链
│   ├── exceptions.py         # RAG 与文档领域异常
│   ├── database/             # PostgreSQL + pgvector 基础设施（配置、engine、ORM 模型）
│   ├── evaluation/           # 离线评估库（数据集、语料、指标、阈值、报告、回答质量评估）
│   └── main.py               # 命令行入口
├── eval/                     # 评估基准：受控语料、清单与标注数据集
│   ├── README.md             # 评估设计、指标定义与使用说明
│   ├── corpus_manifest.json  # 语料清单
│   ├── corpus/               # 3 份原创 Markdown 语料
│   ├── dataset.jsonl         # 60 道带标注问题
│   └── answer_annotations.jsonl  # 60 题回答质量标注（事实、引用、矛盾短语）
├── scripts/
│   ├── evaluate_rag.py       # 检索与拒答评估命令行入口
│   └── evaluate_answers.py   # 回答质量评估命令行入口
├── reports/
│   ├── baseline/             # 已提交的基线报告快照
│   └── generated/            # 本地评估输出（不进入 Git）
├── tests/                    # 离线单元测试
├── frontend/                 # Vue 3 + Vite + TypeScript 前端
│   ├── src/services/         # Search/Ask/Document HTTP 服务与可选模拟服务
│   ├── src/composables/      # 三种模式的独立状态管理
│   ├── src/components/       # 检索、答案、文档管理与状态组件
│   └── README.md             # 前端运行与学习说明
├── docs/docker.md            # Docker Compose 使用、持久化测试与排障
├── docs/database.md          # PostgreSQL + pgvector 数据库基础设施指南
├── docs/answer-evaluation.md # 回答质量评估框架：Schema、指标、命令、限制
├── docs/deployment.md        # Production 单机部署：Caddy HTTPS、迁移、备份与排障
├── Dockerfile                # FastAPI CPU 运行镜像
├── compose.yaml              # 本地完整应用编排和具名卷
├── compose.pgvector.yaml     # pgvector 运行时 override
├── compose.prod.yaml         # production 加固 + Caddy + 迁移服务 override
├── deploy/Caddyfile          # Production Caddy 反向代理配置（自动 HTTPS）
├── .env.production.example   # Production 环境变量模板（真实配置用 .env.production）
├── alembic.ini               # Alembic 迁移配置
├── migrations/               # Alembic 迁移环境与 schema 版本
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

## 使用 Docker Compose

在仓库根目录可以一键构建并启动 FastAPI、编译后的 Vue 前端和 Nginx：

```powershell
docker compose up --build -d
```

默认访问 `http://127.0.0.1:8080`，FastAPI 继续通过 `/api/*` 访问。上传文件与文档元数据保存在 `course-rag-runtime` 具名卷，Hugging Face 模型缓存在 `course-rag-huggingface-cache` 具名卷。

完整配置、健康检查、停止/重启、数据卷说明和自动化重建持久化测试见 [Docker Compose 使用指南](docs/docker.md)。

## Production Deployment

面向单台 Linux VPS 的 production-oriented single-host 部署：

- Docker Compose 三文件组合：`compose.yaml` + `compose.pgvector.yaml` + `compose.prod.yaml`
- 存储后端固定为 **PostgreSQL + pgvector**（生产不使用 memory backend）
- **Caddy 自动 HTTPS**：自动签发与续期证书、HTTP 自动跳转 HTTPS，是唯一公网入口
- 仅暴露公网端口 `80` / `443`；PostgreSQL `5432`、FastAPI `8000`、前端 `8080` 一律不发布
- 具名卷持久化：PostgreSQL 数据、上传原始文档（RAG runtime）、Hugging Face 模型缓存、Caddy 证书与 ACME 状态
- 一次性 Alembic 迁移服务，迁移成功前 backend 不会启动
- 完整部署指南（DNS、防火墙、备份、升级、排障）见 [docs/deployment.md](docs/deployment.md)

快速开始：

```bash
cp .env.production.example .env.production
# 编辑 DOMAIN / POSTGRES_PASSWORD / DATABASE_URL / LLM 配置
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  up -d --build
```

注意：这是 single-host production-style 部署，不是 multi-tenant SaaS；应用当前没有用户认证、访问限流与多用户隔离，公网演示请使用低额度 API key（详见部署文档的安全警告）。

## PostgreSQL 与 pgvector 数据库

本项目提供两种可选存储后端，通过 `VECTOR_STORE_BACKEND` 选择：

- `VECTOR_STORE_BACKEND=memory` 是默认后端：现有 `DocumentRepository`（`documents.json`）和 `KnowledgeIndex`（NumPy 余弦相似度）继续负责上传、删除、检索和问答；**不读取 `DATABASE_URL`、不创建 engine、不连接 PostgreSQL**，即使 `db` 服务未启动也照常工作
- `VECTOR_STORE_BACKEND=pgvector` 是已实现的 PostgreSQL 后端：文档元数据写入 `documents` 表，Chunk 文本与来源信息、`VECTOR(384)` Embedding 写入 `chunks` 表，检索在 PostgreSQL 内执行 pgvector 余弦相似度查询，不把全部向量加载进 Python 内存

pgvector 模式的行为要点：

- 上传时只计算**新文档**的 Embedding，不重新切分、不重新编码历史文档
- 应用重启后直接复用数据库里已保存的 Embedding，不重新计算
- 原始上传文件仍保存在本地 `data/runtime/uploads/`（volume）
- 删除文档时数据库行与 Chunk 通过 `ON DELETE CASCADE` 级联删除；删除使用**补偿式一致性机制**（内部 tombstone 文件），数据库与文件系统无法构成真正的 ACID 事务，失败时会在下次启动自动恢复或清理
- `documents.json` 中的数据**不会**被自动迁移；memory 与 pgvector 的数据集彼此独立，切回 memory 即恢复原有 JSON 数据
- 不创建 HNSW / IVFFlat 等 ANN 索引，不使用 BM25、Redis 或对象存储

`pgvector` 模式启动时会检查数据库可连接、Alembic revision 与本地 head 一致、`vector` 扩展与 `documents` / `chunks` 表存在；本地 Alembic 配置或 migration 目录异常同样会被转换为稳定的 schema 就绪失败。**应用启动不会自动执行迁移**。数据库未就绪或 schema 未迁移时，应用仍能启动并响应 `/health`（`status="degraded"`），但 `/search`、`/ask` 与文档管理接口返回稳定的 503，不会泄露连接串、密码、文件路径或异常堆栈。

启动与迁移流程（详见 [docs/database.md](docs/database.md)）：

```powershell
# 启动数据库（仅 Compose 网络内可见，不映射宿主机端口）
docker compose up -d db

# 重建 backend 镜像以包含依赖与迁移文件，然后显式执行迁移
docker compose build backend
docker compose run --rm backend alembic upgrade head

# 用 pgvector 模式启动 backend 与 frontend
docker compose -f compose.yaml -f compose.pgvector.yaml up -d backend frontend
```

`compose.pgvector.yaml` 只负责把 `VECTOR_STORE_BACKEND` 切换为 `pgvector` 并等待数据库健康，**不会覆盖** `DATABASE_URL`：连接串始终由基础 `compose.yaml` 从环境变量解析（默认 `postgresql+psycopg://course_rag:course_rag@db:5432/course_rag`）。如果自定义了 `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB`，请同步设置 `DATABASE_URL`。

移除数据库并保留具名卷：

```powershell
docker compose down          # 不删除任何具名卷
docker compose down --volumes  # 显式删除全部数据卷（含 postgres-data）时使用
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

## 离线评估

`eval/` 是一份离线评估基准：3 份受控语料、60 道带标注问题，用来回答四个问题——正确证据有没有进入 Top-K、资料内问题有多少被错误拒答、资料外问题有多少被错误放行、仓库默认对比阈值 0.35 在这份受控基准上是否合理。

```powershell
python -m scripts.evaluate_rag
```

报告写入 `reports/generated/`：`summary.json`（机器可读，用于回归对比）、`cases.csv`（每题明细，Excel 可直接打开）、`report.md`（人读报告）。已提交的基线快照在 `reports/baseline/`。

评估复用生产管线（`DocumentLoader` → `chunk_document` → `EmbeddingService` → `KnowledgeIndex`），并复用生产的拒答判定函数 `has_sufficient_context`，因此评估结论不会和线上行为漂移。评估**不调用 LLM**。

当前基线（`paraphrase-multilingual-MiniLM-L12-v2`，chunk 300/50，Top-5）：

| 指标 | 值 |
| --- | ---: |
| Hit@1 / Hit@3 / Hit@5 | 0.5556 / 0.7778 / 0.9167 |
| MRR | 0.6894 |
| test 决策准确率 @ 0.35（仓库默认对比值） | 0.7778 |
| test 错误放行率 @ 0.35 | 0.5714 |
| 推荐阈值（仅由 calibration split 决定） | 0.49 |
| test 决策准确率 @ 0.49 | 0.8333 |

推荐阈值**不会**自动写入生产配置，是否采纳是独立决策。指标定义、标注规则、阈值权重的业务取舍和已知限制都写在 [`eval/README.md`](eval/README.md)。

补充两点：`0.35` 只是仓库代码里的拒答默认值，仅作报告对比基准，不一定等于部署环境实际生效的阈值（部署值可能由 `RAG_MIN_RELEVANCE_SCORE` 覆盖）；test split 的结果已随本仓库公开，之后更适合作为回归基准，不再是未来完全未见的最终 holdout，正式横评需要另取一个新留出集。

### 回答质量评估

除了检索与拒答评估，本仓库还提供基于人工标注事实和引用的**确定性回答质量评估**。
它检查最终答案是否覆盖标注的必要事实、引用编号是否合法、被引用来源是否支持标注事实、
答案是否包含已知矛盾短语，并给出严格标注通过率。`reference_answer` 只用于人工审计，
不参与语义相似度评分；不使用 BLEU/ROUGE、不使用 LLM Judge；这是标注级检查，不是完整
忠实度检测。

离线评分已有回答结果文件（不加载模型、不调用 LLM、完全确定性）：

```powershell
python -m scripts.evaluate_answers `
  --dataset eval/dataset.jsonl `
  --annotations eval/answer_annotations.jsonl `
  --responses <path\to\responses.jsonl> `
  --output-dir reports/generated/answer-quality `
  --run-name my-run --model-label my-model
```

显式 live 生成（真实 Embedding + LLM，必须显式 `--live`，不会无意触发）：

```powershell
python -m scripts.evaluate_answers --live `
  --manifest eval/corpus_manifest.json `
  --dataset eval/dataset.jsonl `
  --annotations eval/answer_annotations.jsonl `
  --output-dir reports/generated/answer-quality/manual-live `
  --run-name manual-live --model-label local-configured-model `
  --top-k 3 --threshold 0.35
```

CI 只使用 `tests/fixtures/answer_evaluation/` 的确定性 Fixture 运行离线 smoke test，
**不会**调用真实 LLM、不会下载 Embedding 模型、不会加载真实 Reranker。

回答评估的解析规则要点：严格引用格式只有 `[来源N]`；普通 Markdown 方括号（如
`[FastAPI]`、`[Python]`、`[1]`）不是引用，不会被当作 malformed citation；引用标记不参与
标注事实与矛盾短语的匹配；`sources` 的 rank 必须从 1 开始连续；启用 Reranker 时
`max_relevance_score`（拒答依据）允许高于最终返回来源的最高检索分数；离线报告只记录
仓库相对路径或文件名，不包含本地绝对路径。

指标定义、标注 Schema、strict_pass 规则与限制详见 [`docs/answer-evaluation.md`](docs/answer-evaluation.md)。

## 可选 Reranker

系统支持可选的两阶段检索（向量候选池 + CrossEncoder Reranker）。Reranker 默认关闭，启用时在向量检索之后对更大候选池（默认 Top-15）进行精排，返回最终 Top-K。

在受控的 60 题离线基准（固定语料 `eval/corpus/` 与固定问题集 `eval/dataset.jsonl`）中，启用 Reranker 后 Hit@1 从 0.5556 提升到 0.8611，MRR 从 0.6894 提升到 0.9259。当前本地 CPU 环境下 Reranker 中位延迟约 5.6 秒，因此默认没有启用。注意：这些是固定语料和固定问题集上的受控离线结果，不能直接代表任意知识库或线上环境表现。Reranker 模型加载或执行失败时会安全回退到向量排序，不影响 `/search` 与 `/ask` 可用性。

**配置**（`.env`）：

```env
RAG_RERANKER_ENABLED=false
RAG_RERANKER_MODEL=<本地缓存的中文或多语言 CrossEncoder 模型>
RAG_RERANKER_CANDIDATE_TOP_K=15
```

- `RAG_RERANKER_ENABLED=false` 时行为与原来完全一致，不加载额外模型
- 拒答决策仍基于 `retrieval_score`（向量相似度），不使用 `rerank_score`
- Reranker 失败时安全回退到原始向量排序
- **Reranker 模型必须与语料语言匹配**：本项目语料为中文，应使用中文或多语言 CrossEncoder 模型，**不要默认使用英文 MS MARCO 模型**（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`）；模型必须提前下载并存在于本地缓存，以 `local_files_only` 方式加载，不会自动联网下载
- `/search` 与 `/ask` 顶层响应暴露 `reranker_applied`（本次请求实际使用 Reranker）与 `reranker_fallback`（本次请求尝试 Reranker 但失败回退）；两者不会同时为 true
- `/health` 的 `reranker_model` 会对本地路径脱敏（显示为 `<local-model>`）；`config_invalid` 状态下若启用标志本身无法解析，`reranker_enabled` 保持 false
- 每个可回答问题必须且只能属于 improved / regressed / unchanged 之一，**两个分支都未命中的题目归入 unchanged**
- **只有真实 CrossEncoder 完整运行（`real_model_run=true`）才可能产生生产启用建议**；FakeReranker 或测试替身（`real_model_run=false`）永远不会

**A/B 评估**：

```powershell
python -m scripts.evaluate_reranker `
  --manifest eval/corpus_manifest.json `
  --dataset eval/dataset.jsonl `
  --candidate-top-k 15 `
  --final-top-k 5 `
  --reranker-model "<本地缓存的中文或多语言 CrossEncoder 模型>" `
  --output-dir reports/generated/reranking
```

报告对比 vector-only 与 reranked 两种模式的 Hit@K、Recall@K、MRR，并按 category / difficulty / split 分组。详见 [`docs/reranking.md`](docs/reranking.md)。

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

当前 Web 应用已经支持单轮、带来源的课程知识问答、可配置相关性阈值、资料不足拒答、独立语义检索，以及 TXT、Markdown、文本型 PDF 的上传、列表、删除和即时索引更新，另有一份离线的检索与拒答评估基准。存储层支持两种后端：默认 `memory`（`documents.json` + `KnowledgeIndex`，不连接数据库）与 `pgvector`（PostgreSQL 持久化文档、Chunk 与 `VECTOR(384)` Embedding，数据库内余弦检索）；`pgvector` 模式不自动迁移 `documents.json`，不创建 HNSW / IVFFlat 索引，应用启动不自动执行 Alembic。扫描 PDF 的 OCR、图片识别、Word、PowerPoint、Excel、对象存储、用户登录与多用户隔离、后台任务队列、多轮记忆、流式输出、LangChain、LangGraph 与 Agent 仍未实现。评估只覆盖检索质量与拒答决策，生成答案质量与答案忠实度的评估也未实现。本地 JSON 与文件系统只是默认存储实现，没有声称支持生产级并发和扩展。
