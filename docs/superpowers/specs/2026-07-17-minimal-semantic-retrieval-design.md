# 最小语义检索项目设计

## 1. 目标与范围

本项目面向第一次学习 RAG 的 Python 初学者，只实现 RAG 中的“检索”阶段：读取本地中文资料、按固定长度切分、生成向量、计算问题与文本块的余弦相似度，并在命令行展示最相关的三个结果。

本阶段不生成自然语言答案，也不引入大语言模型 API、Agent、Tool Calling、Web 服务、数据库、向量数据库、LangChain 或 LangGraph。

## 2. 环境与目录决策

- 项目目录：`course-rag/`
- 当前解释器：Python 3.14.6
- 依赖：`sentence-transformers`、`numpy`、`pytest`
- 路径处理：只使用 `pathlib.Path`，不硬编码绝对路径
- 当前父目录为空且不是 Git 仓库，没有虚拟环境或依赖文件，因此不存在文件覆盖或模块复用冲突

项目结构如下：

```text
course-rag/
├── data/
│   └── knowledge.txt
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embedding.py
│   ├── retriever.py
│   └── main.py
├── tests/
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_embedding.py
│   └── test_retriever.py
├── docs/superpowers/specs/
│   └── 2026-07-17-minimal-semantic-retrieval-design.md
├── .gitignore
├── requirements.txt
└── README.md
```

`test_embedding.py` 是在用户给出的最小结构上增加的测试文件，用轻量假模型验证封装逻辑，不下载真实模型。

## 3. 模块职责与接口

### `src/loader.py`

提供 `load_text(file_path: str) -> str`。函数先用 `Path.is_file()` 检查文件，再以 UTF-8 读取，去掉全文首尾空白并返回。文件不存在时抛出 `FileNotFoundError`；清理后为空时抛出带明确信息的 `ValueError`；编码和系统 I/O 异常保持原样向上传递。

### `src/chunker.py`

提供：

```python
def split_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
```

切分采用滑动窗口。每次窗口起点增加 `chunk_size - chunk_overlap`，因此相邻非末尾窗口共享 `chunk_overlap` 个字符。参数必须满足 `chunk_size > 0`、`chunk_overlap >= 0` 且 `chunk_overlap < chunk_size`，否则抛出 `ValueError`。每个候选块清理首尾空白后，仅将非空块加入结果；窗口游标无论候选块是否为空都会前进，避免死循环。

### `src/embedding.py`

提供 `EmbeddingService`：

```python
class EmbeddingService:
    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ) -> None:
        ...

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        ...

    def encode_query(self, query: str) -> np.ndarray:
        ...
```

构造函数只实例化一次 `SentenceTransformer`。文档使用一次批量 `encode` 调用；问题使用单条列表调用并取第一行。两种接口都要求输入经过清理后非空，返回浮点 NumPy 数组，并通过 `normalize_embeddings=True` 请求模型输出单位向量。代码只依赖模型的 `encode` 行为，不增加抽象基类，未来可通过更换模型名或替换该类实现来扩展。

### `src/retriever.py`

提供 `SemanticRetriever`：

```python
class SemanticRetriever:
    def __init__(
        self,
        chunks: list[str],
        embeddings: np.ndarray,
    ) -> None:
        ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict[str, int | float | str]]:
        ...
```

构造函数检查文本块非空、Embedding 为二维数组、行数与文本块数量一致、每个向量范数非零。为使该类可独立使用，构造函数会用 NumPy 对文档向量再次归一化。`search` 检查 `top_k > 0`、查询向量是一维或单行二维向量、维度匹配且范数非零，然后归一化查询向量。余弦相似度由归一化矩阵和查询向量的 NumPy 点积得到，使用稳定降序排序；实际结果数量为 `min(top_k, len(chunks))`。

每条结果包含从 1 开始的展示排名、浮点分数、原文，以及从 0 开始的 `chunk_index`：

```python
{
    "rank": 1,
    "score": 0.82,
    "text": "相关原文",
    "chunk_index": 3,
}
```

### `src/main.py`

入口通过 `Path(__file__).resolve().parents[1] / "data" / "knowledge.txt"` 找到知识文件，因此从项目根目录执行 `python -m src.main` 时不依赖当前操作系统的路径分隔符。

启动顺序为：加载文本、切分文本、创建 `EmbeddingService`、批量生成文档向量、创建检索器、打印块数量。交互循环清理用户输入；空输入、`exit` 或 `quit` 结束程序。其他输入被编码后检索 Top 3，并打印排名、四位小数相似度、Chunk 编号和原文。文件、参数、模型下载等异常不使用宽泛的 `except Exception` 隐藏，错误会保留可诊断信息。

## 4. 数据流

```text
data/knowledge.txt
        ↓ load_text
完整中文文本
        ↓ split_text
Chunk 列表
        ↓ encode_documents（启动时一次）
归一化文档向量矩阵

用户问题
        ↓ encode_query
归一化查询向量
        ↓ SemanticRetriever.search
NumPy 点积 → 余弦分数 → 降序排序 → Top 3
        ↓
排名、四位小数分数、Chunk 编号、原文
```

## 5. 示例知识库

`data/knowledge.txt` 使用简短中文段落解释 FastAPI、router 层、service 层、repository 层、PostgreSQL、Embedding 和 RAG。段落需要让以下语义改写问题能找到对应内容：

- 业务逻辑应该写在哪一层？
- 哪个模块负责访问数据库？
- 如何将文本表示成向量？

## 6. 错误处理

- 文件缺失：`FileNotFoundError`
- 文件清理后为空：`ValueError`
- Chunk 参数非法：`ValueError`
- 文档或问题输入为空：`ValueError`
- Chunk 数量和向量行数不一致：`ValueError`
- 向量维度或形状不一致：`ValueError`
- 零向量无法计算余弦相似度：`ValueError`
- `top_k <= 0`：`ValueError`
- 模型安装、下载或缓存错误：保留依赖库原始异常和上下文，不静默降级为关键词检索

## 7. 测试策略

开发遵循测试驱动流程：先写会因模块或行为缺失而失败的测试并确认失败原因，再编写最小实现，最后运行全部测试。

- Loader：UTF-8 正常读取、缺失文件、空文件
- Chunker：正常切分、精确重叠、短文本、非法大小、非法重叠、忽略空白块
- Embedding：模型只在构造时加载一次、文档批量编码、查询编码、NumPy 输出、空输入拒绝；通过 monkeypatch 注入轻量假 `SentenceTransformer`，不访问网络
- Retriever：降序、Top K 数量、超量 Top K、非法 Top K、数量不一致、最相关结果第一，以及形状、维度和零向量校验

单元测试必须离线运行。最终验证命令为：

```powershell
pytest -v
```

真实模型仅用于最终命令行冒烟测试，依次输入三个指定问题后输入 `exit`。冒烟测试要保留实际输出；若 Python 3.14.6 与底层机器学习依赖不兼容，则记录具体安装错误，并优先使用本机已安装的兼容 Python 3.x 创建 `.venv`，不改变应用设计。

## 8. 文档与交付

中文 README 解释项目范围、目录、PowerShell 环境搭建和命令、测试、示例问题、完整 RAG 数据流、Chunk、Embedding、余弦相似度、Top K、常见错误和下一阶段如何在检索结果之后增加大模型生成。README 明确说明当前没有实现生成、Web API、数据库或向量数据库。

`.gitignore` 覆盖 `.venv`、Python 缓存、pytest 缓存、模型缓存、环境变量文件、常见 IDE 和操作系统文件。

最终汇报列出文件职责、完整数据流、测试结果、三个问题的实际冒烟输出、未解决问题、建议优先阅读的三个文件，并给出五个理解检查问题。完成后停止，不自动加入下一阶段功能。
