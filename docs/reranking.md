# 两阶段检索与 Reranker

## 什么是两阶段检索

两阶段检索（two-stage retrieval）在原有向量检索（第一阶段）之后增加一个
Reranker（第二阶段排序器），对更大的候选池进行精排：

1. **Candidate Retrieval** — 向量检索返回 Top-15 候选（`candidate_top_k`）。
2. **Reranking** — CrossEncoder 对每个 `(query, chunk_text)` pair 重新打分，
   按新分数排序，取 Top-5（`final_top_k`）返回。

## Candidate Retrieval 和 Reranking 的区别

| 维度 | Candidate Retrieval | Reranking |
| --- | --- | --- |
| 模型 | Embedding 模型 | CrossEncoder 模型 |
| 输入 | query embedding vs chunk embeddings | (query_text, chunk_text) pairs |
| 分数 | 余弦相似度 (`retrieval_score`) | CrossEncoder 输出 (`rerank_score`) |
| 分数范围 | [-1, 1] | 模型相关，不假设 [0, 1] |
| 语义 | 向量空间相似度 | query-chunk 相关性 |
| 用途 | 候选池生成 + 拒答决策 | 精排 |

## 为什么 Candidate Top-K 要大于 Final Top-K

向量检索是粗排，可能把正确证据排在第 6-15 位。增大候选池给 Reranker
更多机会把正确证据提升到 Top-5。如果候选池只有 5 个，Reranker 只能在这 5
个中重排，无法救回落在 Top-5 之外的证据。

## retrieval_score 与 rerank_score 的区别

- **`retrieval_score`** — 向量余弦相似度，来自第一阶段 Embedding 检索。
  这是当前生产系统用于拒答决策的分数。
- **`rerank_score`** — CrossEncoder 输出，来自第二阶段精排。
  语义不同，不可与 `retrieval_score` 直接比较。

API 响应中，`score` 字段继续代表 `retrieval_score`（保持向后兼容）。
新增可选字段 `retrieval_rank`、`rerank_score`、`reranker_applied`。

## 请求级回退可见性

`/search` 与 `/ask` 的顶层响应都新增两个布尔字段：

- **`reranker_applied`** — 本次请求成功使用了 Reranker 排序。
- **`reranker_fallback`** — 本次请求原本尝试 Reranker，但调用失败并回退
  vector-only 排序。

两者都为 `false` 时，表示 Reranker 未启用（或当前没有 RetrievalService）。

两者不会同时为 `true`。

```json
{
  "query": "问题",
  "reranker_applied": false,
  "reranker_fallback": true,
  "results": []
}
```

`/ask` 在 `insufficient_context` 与 `answered` 两种状态下都会返回这两个字段。

## health 中的模型名脱敏

`/health` 返回的 `reranker_model` 会经过安全显示函数处理：

- Hugging Face 风格模型 ID（如 `BAAI/bge-reranker-v2-m3`）原样显示。
- 本地绝对路径（POSIX、Windows、UNC）、`file://` URI、`~` 开头路径统一
  显示为 `<local-model>`，不会暴露用户名、盘符、缓存目录或项目路径。

内部模型加载仍使用原始配置值，只有公开响应使用安全显示名。

## `config_invalid` 的 enabled 语义

| 状态 | `reranker_enabled` | `reranker_ready` | 说明 |
| --- | --- | --- | --- |
| `disabled` | false | false | 正常禁用 |
| `ready` | true | true | 启用且加载成功 |
| `load_failed` | true | false | 启用、配置有效、加载失败 |
| `config_invalid` | 视情况 | false | 配置非法 |

`RAG_RERANKER_ENABLED` 本身无法解析（如 `maybe`）时，系统不能确认用户请求
启用，因此公开字段保持 `reranker_enabled=false`、`reranker_status=config_invalid`。
请求启用但模型名为空或候选数非法时，`reranker_enabled=true`、
`reranker_status=config_invalid`。

## 未变案例（unchanged）

每个可回答问题必须且只能属于 `improved` / `regressed` / `unchanged` 之一。
**两个分支都未命中**（vector 与 reranked 均 miss）的题目归入 `unchanged`。
不可回答题目不进入任何检索排名变化集合。`rank_change` 只在两分支都有
相关排名时计算；任一边 miss 时保持 `None`，不使用虚构排名。

## 为什么本 PR 不用 rerank_score 做拒答

当前拒答阈值 `0.35`（或校准后的 `0.49`）来自 Embedding 分数分布。
Reranker 分数语义不同，分布不同，直接套用会导致拒答行为不可预测。

本 PR 只验证 Reranker 的**排序价值**，不改变拒答语义。Reranker 拒答阈值
需要后续单独校准。

## 默认为什么关闭

- Reranker 增加额外模型加载成本和请求延迟
- 需要独立的分数校准才能用于拒答
- 离线 A/B 评估需要先验证收益
- 生产系统不应因可选增强功能引入单点故障

## 如何配置

在 `.env` 中设置：

```env
# 启用 Reranker（默认关闭）
RAG_RERANKER_ENABLED=false

# CrossEncoder 模型名称（启用时必填）
# 必须与语料语言匹配：本项目语料为中文，应使用中文或多语言 CrossEncoder 模型，
# 不要默认使用英文 MS MARCO 模型（如 cross-encoder/ms-marco-MiniLM-L-6-v2）。
# 模型必须提前下载并存在于本地缓存或本地目录；加载以 local_files_only 方式，
# 不会自动联网下载。
RAG_RERANKER_MODEL=<本地缓存的中文或多语言 CrossEncoder 模型>

# 候选池大小（默认 15，最小 5，最大 100）
RAG_RERANKER_CANDIDATE_TOP_K=15
```

`RAG_RERANKER_ENABLED=false` 时：
- 不加载 CrossEncoder 模型
- `/search` 和 `/ask` 行为与原来完全一致
- `/health` 中 `reranker_enabled=false`

## 离线模型加载（强制）

CrossEncoder 模型以 `local_files_only=True` 方式加载：

- 模型已缓存在本地 → 正常加载。
- 模型不存在于本地 → 清晰失败，**不会自动联网下载**。
- 评估 CLI 与生产加载都遵循此规则。

若未提前下载模型，请勿启用 Reranker；启用后加载失败不会导致应用启动失败，
`/search` 和 `/ask` 仍按 vector-only 工作，`/health` 报告
`reranker_status=load_failed`（不会暴露本地路径、堆栈或内部异常原文）。

## Reranker 模型语言必须匹配语料

- 本项目语料为中文，**不应默认使用英文 MS MARCO 模型**。
- Reranker 模型必须使用中文或多语言 CrossEncoder 模型。
- 真实 baseline 必须记录完整模型名称，推荐同时记录 model revision。
- 没有合适的本地模型时，指标保持 `N/A`，**不得**用 FakeReranker 结果冒充真实效果。

## 只有真实 CrossEncoder 才能产生启用建议

A/B 评估记录显式的运行来源（`reranker_backend` / `real_model_run`）：

- 只有默认 CLI factory 成功构造真实 `CrossEncoderReranker` 时，
  `backend=cross_encoder`、`real_model_run=true`。
- 测试注入的 `FakeReranker`：`backend=fake`、`real_model_run=false`。
- 其他自定义 factory：`backend=custom`、`real_model_run=false`。

启用建议的检查顺序：**真实模型身份 → 回退 → 数据完整性 → 指标门槛 →
决策一致性**。FakeReranker 或任何 `real_model_run=false` 的运行，即使指标
看起来很好，也永远不会产生生产启用建议；理由为「本次运行未使用真实
CrossEncoder，不能建议生产启用」。

## 决策一致性与独立分数

评估对每道题**分别独立计算**两个分支的最高向量检索分数：

- `candidate_max_retrieval_score` — 完整候选池最高向量分数。
- `vector_max_retrieval_score` — vector 分支视角的最高向量分数。
- `reranked_max_retrieval_score` — reranked 完整候选列表（截断前）的
  最高向量分数。

两者都必须基于完整候选池，且在 reranked Top-K 截断前计算。`reranked_max`
仍然是 retrieval score，不使用 rerank score。若重排过程丢失候选、修改
retrieval score 或使用错误集合，两分支分数发散，决策一致性检查会失败并
阻止启用建议——该检查是真实断言，不是「设计上应一致」的假检查。

## 如何运行 A/B 评估

```bash
python -m scripts.evaluate_reranker \
  --manifest eval/corpus_manifest.json \
  --dataset eval/dataset.jsonl \
  --candidate-top-k 15 \
  --final-top-k 5 \
  --reranker-model "<本地缓存的中文或多语言 CrossEncoder 模型>" \
  --output-dir reports/generated/reranking
```

> 说明：Reranker 模型必须与语料语言匹配。本项目语料为中文，**不应默认使用英文
> MS MARCO 模型**（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`）。模型必须提前下载
> 并存在于本地缓存或本地目录；评估以 `local_files_only` 方式加载，不会自动联网
> 下载。真实 baseline 必须记录完整模型名称（推荐同时记录 model revision）。没有
> 合适的本地模型时，保持指标为 `N/A`，**不得**用 FakeReranker 结果冒充真实效果。

输出文件：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 完整指标、差值、分组对比、推荐 |
| `cases.csv` | 每题详细指标 |
| `report.md` | 人类可读报告 |
| `latency.json` | 延迟统计（仅 `--include-latency`） |

## 如何理解 Candidate Hit@15

Candidate Hit@15 表示正确证据是否进入了 Top-15 候选池。如果
Candidate Hit@15 = 1.0，说明所有正确证据都进入了候选池，Reranker 有
机会重排。如果 Candidate Hit@15 < 1.0，说明有些证据连候选池都没进，
Reranker 无法救回——这需要改进 Embedding 模型或 Chunk 策略。

## 如何理解改善与退化案例

- **改善** — vector miss / low rank → reranked hit / higher rank
- **退化** — vector hit → reranked miss / lower rank

Reranker 不保证所有问题都改善。退化案例需要单独审查，确认是否可接受。

## Reranker 失败时如何回退

当 Reranker 不可用或失败时（模型未配置、加载失败、调用异常、返回无效
分数等），系统自动回退到原始向量排序：

```
Vector Retrieval 成功 → Reranker 失败
→ 回退到原始 Vector 排序
→ 返回正常搜索/问答结果
→ reranker_fallback=true
```

回退不会：
- 丢失全部结果
- 返回损坏排序
- 让 `/search` 返回 500
- 让 `/ask` 无法回答本来可回答的问题

## test split 已公开

`test` split 的结果公开后应视为已发布的回归测试集。正式比较多个
Reranker 方案时，需要新增一份从未用于开发决策的独立 final holdout。

## 延迟结果依赖设备

延迟数据使用 `time.perf_counter()` 测量，依赖设备和运行环境。
默认不包含在确定性质量 baseline 中。

## 真实模型文件不会提交 Git

模型权重和缓存不提交到 Git 仓库。`.gitignore` 已排除相关目录。
