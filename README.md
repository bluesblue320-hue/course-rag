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
- 通过 FastAPI 提供健康检查和 Top-K 检索接口
- 在服务启动时构建一次索引，并在请求之间复用
- 通过 Vue 3 页面提交问题并展示真实检索结果
- 通过 Vite `/api` 代理连接浏览器与 FastAPI
- 返回检索耗时、模型名和已索引 Chunk 数量

## 当前没有实现

本阶段没有调用大语言模型生成答案，也没有数据库、向量数据库、LangChain、LangGraph 或 Agent。命令行、Web API 和 Vue 前端都只负责检索并展示相关原文。

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
│   └── main.py               # 命令行入口
├── tests/                    # 离线单元测试
├── frontend/                 # Vue 3 + Vite + TypeScript 前端
│   ├── src/services/         # 真实 HTTP 服务与可选模拟服务
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

检索请求：

```powershell
curl.exe -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query":"业务逻辑应该写在哪里？","top_k":3}'
```

`query` 不能为空，`top_k` 默认是 `3` 且必须大于 `0`。接口返回清理后的问题、检索耗时、模型名、已索引 Chunk 数量，以及按相似度降序排列的原文、分数、排名和 Chunk 编号。

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

浏览器打开 `http://127.0.0.1:5173`。前端默认请求 `/api/search`，Vite 会把 `/api` 代理到 `http://127.0.0.1:8000`，因此开发环境不需要额外配置 CORS。

如果只想演示界面、不启动 Python 后端，可以在启动 Vite 前设置：

```powershell
$env:VITE_USE_MOCK_SEARCH="true"
npm.cmd run dev
```

## 运行测试

```powershell
pytest -v
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

浏览器 → Vue → Vite /api 代理 → FastAPI
                                ↓
用户问题                      问题向量
                                ↓
                  文档向量与问题向量计算余弦相似度
                                ↓
                  按分数降序返回 Top K 原文
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

当前浏览器到真实检索器的链路已经打通。下一阶段可以增加一个生成模块：把用户问题和检索出的 Top K 原文组装成提示词，再交给大语言模型生成有依据的回答。检索器、Chunk、Embedding、FastAPI 和前端状态机仍可复用。
