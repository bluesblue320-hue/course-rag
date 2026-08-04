# RAG 离线评估基线报告（真实 Embedding）

本报告由生产实现直接运行生成：`DocumentLoader → chunk_document → EmbeddingService → KnowledgeIndex`，不调用任何 LLM。

## 运行配置

| 配置项 | 值 |
| --- | --- |
| Embedding 模型 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Chunk size | 300 |
| Chunk overlap | 50 |
| Top-K | 5 |
| 数据集版本 | v1（`eval/dataset.jsonl`，60 题） |
| 阈值范围 | 0.20 – 0.60，步长 0.01 |
| 错误放行权重 | 3（业务选择，非行业标准） |
| 错误拒答权重 | 1 |
| 运行命令 | `python -m scripts.evaluate_rag --manifest eval/corpus_manifest.json --dataset eval/dataset.jsonl --top-k 5 --threshold-start 0.20 --threshold-end 0.60 --threshold-step 0.01 --false-answer-weight 3 --false-refusal-weight 1 --output-dir reports/generated/rag-evaluation` |

模型从本地 Hugging Face 缓存加载（`HF_HUB_OFFLINE=1`），未访问网络。

## 数据集

60 题：calibration 42 / test 18；answerable 36 / unanswerable 24。

category：direct 15、paraphrase 15、multi_evidence 6、out_of_scope_near 12、out_of_scope_far 12。
difficulty：easy 14、medium 28、hard 18。

## 检索指标（answerable=36）

| 指标 | 值 |
| --- | --- |
| Hit@1 | 0.5000 |
| Hit@3 | 0.8056 |
| Hit@5 | 0.8889 |
| Recall@1 | 0.4722 |
| Recall@3 | 0.7500 |
| Recall@5 | 0.8472 |
| MRR | 0.6491 |

按 category（Hit@5 / MRR）：direct 0.8000 / 0.6167；paraphrase 0.9333 / 0.6522；multi_evidence 1.0000 / 0.7222。
按 difficulty（Hit@5 / MRR）：easy 0.7857 / 0.6369；medium 0.9286 / 0.6155；hard 1.0000 / 0.7292。

## 相似度分布

| 分组 | count | min | max | mean | median | p25 | p75 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| answerable | 36 | 0.4151 | 0.8628 | 0.5964 | 0.5716 | 0.5140 | 0.6659 |
| unanswerable | 24 | 0.0035 | 0.7425 | 0.3212 | 0.2973 | 0.1086 | 0.5026 |

两组分布存在明显重叠（unanswerable 的 p75 约 0.50，answerable 的 p25 约 0.51），单阈值能力有限，建议后续考虑检索改进或二阶段判断。

## 阈值校准结果

推荐阈值只从 calibration（42 题）选择。选择规则：加权成本最低 → 错误放行更少 → 错误拒答更少 → 准确率更高 → 阈值更高（保守优先）。

| 拆分 | 阈值 | 准确率 | 错误放行率 | 错误拒答率 |
| --- | --- | --- | --- | --- |
| calibration | 0.35（当前） | 0.8333 | 0.4118 | 0.0000 |
| calibration | 0.53（推荐） | 0.7857 | 0.0588 | 0.3200 |
| test | 0.35（当前） | 0.7778 | 0.5714 | 0.0000 |
| test | 0.53（推荐） | 0.7222 | 0.2857 | 0.2727 |

当前生产阈值 0.35 的错误拒答为 0，但错误放行率过高（calibration 41%，test 57%）；推荐阈值 0.53 把错误放行降到 6%/29%，代价是错误拒答上升到 32%/27%。当前阈值与推荐阈值各有取舍：`0.35` 偏向回答（拒绝不足），`0.53` 偏向保守。是否调整生产阈值属于单独决策，本报告不自动修改配置。

## 失败案例

- 检索失败（Hit@5 未命中）：q009、q012、q013、q022。
- 推荐阈值 0.53 下的错误拒答：q002、q004、q005、q009、q012、q013、q017、q018、q023、q029、q033。
- 当前阈值 0.35 下的错误放行：q037–q048（11 个资料外问题，多为 out_of_scope_near）。

## 已知限制

- 语料为 3 篇受控 Markdown 文档，不等同于真实用户文档。
- 不评估生成答案质量与 LLM 忠实度。
- Hit@5 高不代表生成答案正确。
- 本基线与当前 Embedding 模型绑定，更换模型或 Chunk 参数后必须重新运行。

详细逐题数据见 `reports/generated/rag-evaluation/`。
