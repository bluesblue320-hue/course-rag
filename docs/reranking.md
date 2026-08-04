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
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# 候选池大小（默认 15，最小 5，最大 100）
RAG_RERANKER_CANDIDATE_TOP_K=15
```

`RAG_RERANKER_ENABLED=false` 时：
- 不加载 CrossEncoder 模型
- `/search` 和 `/ask` 行为与原来完全一致
- `/health` 中 `reranker_enabled=false`

## 如何运行 A/B 评估

```bash
python -m scripts.evaluate_reranker \
  --manifest eval/corpus_manifest.json \
  --dataset eval/dataset.jsonl \
  --candidate-top-k 15 \
  --final-top-k 5 \
  --reranker-model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
  --output-dir reports/generated/reranking
```

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
