# 课程 RAG 问答项目

这是一个面向初学者的最小 RAG 项目。它读取本地中文资料，把资料切分成 Chunk，使用 Embedding 模型和余弦相似度检索相关原文，并通过 OpenAI-compatible Chat Completions API 生成有来源标注的回答。

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
- 通过 FastAPI `/ask` 返回答案、来源、模型名和各阶段耗时
- 通过 Vue 3 页面在“智能问答”和“语义检索”之间切换
- 智能问答模式调用 `/api/ask`，展示最终答案、引用来源、模型和耗时
- 语义检索模式继续调用 `/api/search`，展示原始 Top-K 检索结果
- 对问答请求的 422、502、503 和网络错误提供稳定、安全的页面提示
- 通过 Vite `/api` 代理连接浏览器与 FastAPI
- 返回检索耗时、模型名和已索引 Chunk 数量

## 当前没有实现

后端和 Vue 前端已经打通从语义检索到 LLM 回答展示的 RAG 闭环。本项目仍未支持 PDF、文件上传、数据库、向量数据库、评估集、多轮记忆、LangChain、LangGraph 或 Agent。

## 项目目录

```text
course-rag/
├── data/knowledge.txt        # 中文示例知识库
├── src/
│   ├── api.py                # FastAPI Web 接口
│   ├── loader.py             # 读取文本
│   ├── chunker.py            # 切分文本
│   ├── embedding.py          # 生成向量
│   ├── retriever.py          # 计算相似度并排序
│   ├── prompt_builder.py      # 组装课程问答 Prompt
│   ├── generation.py         # 调用 OpenAI-compatible LLM
│   ├── rag_service.py        # 编排完整 RAG 调用链
│   ├── exceptions.py         # RAG 领域异常
│   └── main.py               # 命令行入口
├── tests/                    # 离线单元测试
├── frontend/                 # Vue 3 + Vite + TypeScript 前端
│   ├── src/services/         # Search/Ask HTTP 服务与可选模拟服务
│   ├── src/composables/      # 两种模式的独立状态管理
│   ├── src/components/       # 检索、答案、来源与状态组件
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
```

FastAPI 启动时会通过 `python-dotenv` 自动读取项目根目录的 `.env`，且不会覆盖已有的进程环境变量。`LLM_API_KEY` 和 `LLM_MODEL` 必填；`LLM_BASE_URL` 默认是 `https://api.openai.com/v1`，`LLM_TIMEOUT_SECONDS` 默认是 `30` 且必须大于 `0`。`.env` 包含密钥，绝不能提交到 Git；`.env.example` 只能保留不含真实值的字段模板。

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
  "generation_ready": false
}
```

检索索引成功初始化后，`status` 保持为 `"ok"`；`generation_ready` 仅在完整的问答服务可用时为 `true`。

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

`question` 清理后长度必须为 1 到 500，`top_k` 默认是 `3` 且范围为 1 到 10。`/ask` 返回生成答案、Top-K 来源、Embedding 与 LLM 模型名，以及检索、生成和总耗时。LLM 调用失败返回统一的 `502` 响应，配置缺失返回统一的 `503` 响应，不会向客户端暴露上游完整错误。

`/search` 始终只返回原始检索结果；`/ask` 才会执行 Prompt 构造和 LLM 回答生成。

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

配置好根目录 `.env` 中的后端 LLM 环境变量后，智能问答可以调用真实模型。未配置 LLM 时，智能问答会展示 503 提示，但不依赖 LLM 的语义检索仍可正常使用。真实 `.env` 不得提交到 Git。

如果只想演示界面、不启动 Python 后端，可以在启动 Vite 前设置：

```powershell
$env:VITE_USE_MOCK_SEARCH="true"
$env:VITE_USE_MOCK_ASK="true"
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
knowledge.txt
    ↓ 读取
完整文本
    ↓ Chunk 切分
多个文本块
    ↓ Embedding
文档向量矩阵

用户问题
    ↓ Embedding
问题向量
    ↓ 与文档向量计算余弦相似度
Top K 原文
    ├── `/search`：直接返回检索结果
    └── `/ask`：PromptBuilder → GenerationService → 答案和来源
```

Vue 前端通过模式切换分别使用 `/ask` 和 `/search` 两个分支；两种模式共用问题输入，但分别保留最近的请求状态和结果。

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

当前 Web 应用已经支持单轮、带来源的课程知识问答和独立语义检索。PDF、文件上传、数据库、向量数据库、相似度阈值、评估集、多轮记忆、LangChain、LangGraph 与 Agent 仍未实现。
