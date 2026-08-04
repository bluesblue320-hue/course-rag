# RAG 离线评估

本目录提供一套可重复运行的 RAG 评估基线，用数据衡量检索质量、拒答决策和相关性阈值是否合理。

## 为什么需要评估

没有评估就无法回答"当前 RAG 效果如何"。本套件回答以下问题：

- 正确证据能否进入 Top-K，通常排在第几位（Hit@K、MRR）
- 资料内问题是否被错误拒答（false refusal）
- 资料外问题是否被错误放行（false answer）
- 当前阈值 `0.35` 是否合理（阈值扫描与校准）
- 更换 Chunk、Embedding、Reranker 或检索实现后是否发生退化（重跑同一套评估对比）

## 评估的三部分

1. **检索评估**：只衡量"证据是否被检索到"，不生成答案。
2. **拒答决策评估**：用混淆矩阵衡量"该答是否答、该拒是否拒"。
3. **阈值校准**：在 calibration 拆分上扫描阈值并选择推荐值，再用 test 拆分验证一次。

本阶段**不评估生成答案质量**，不调用 LLM，也不做 LLM-as-a-Judge。

## Calibration 与 Test

- **calibration**（42 题）：用于选择推荐阈值。选择规则：加权成本最低 → 错误放行更少 → 错误拒答更少 → 决策准确率更高 → 阈值更高（保守优先）。
- **test**（18 题）：只用于对固定的推荐阈值做一次最终验证，绝不参与阈值选择。

校准集与测试集题目不重复，且不是简单替换几个词的近似重复。

## 目录结构

```text
eval/
├── README.md               # 本说明
├── corpus_manifest.json    # 语料清单（document_id / path / filename）
├── dataset.jsonl           # 标注数据集（每行一题）
└── corpus/                 # 三篇原创受控语料
    ├── backend-architecture.md
    ├── rag-fundamentals.md
    └── database-basics.md
```

## 如何运行

```powershell
python -m scripts.evaluate_rag `
  --manifest eval/corpus_manifest.json `
  --dataset eval/dataset.jsonl `
  --top-k 5 `
  --threshold-start 0.20 `
  --threshold-end 0.60 `
  --threshold-step 0.01 `
  --false-answer-weight 3 `
  --false-refusal-weight 1 `
  --output-dir reports/generated/rag-evaluation
```

评估复用生产代码：`DocumentLoader → chunk_document → EmbeddingService → KnowledgeIndex`，不重新实现切分、向量或相似度。

输出：

- `reports/generated/rag-evaluation/summary.json`：结构化指标（标准 JSON，无 NaN）
- `reports/generated/rag-evaluation/cases.csv`：逐题明细（UTF-8 BOM，便于 Excel 打开）
- `reports/generated/rag-evaluation/report.md`：可读报告

正式基线快照（真实 Embedding 模型运行结果）保存在 `reports/rag-evaluation-baseline.md` 和 `.json`。`reports/generated/` 已加入 `.gitignore`，不会进入 Git。

## 指标定义

- **Hit@K**：Top-K 中是否至少命中一个标准 evidence group（只统计 answerable 题目）。
- **Recall@K**：Top-K 命中的 evidence group 数 / 该题 evidence group 总数，按题做 macro average。
- **MRR**：第一个证据命中排名的倒数（1/1=1.0，1/2=0.5，1/3≈0.333），未命中记 0。
- **混淆矩阵**：TP=可回答且回答；FN=可回答但拒答；TN=不可回答且拒答；FP=不可回答但回答。
- **false answer rate** = FP / 实际不可回答数；**false refusal rate** = FN / 实际可回答数。分母为 0 时输出 null 而不是错误数字。
- **weighted_cost** = FP × 错误放行权重 + FN × 错误拒答权重。

> 错误放行权重默认 3、错误拒答权重默认 1，是本项目的业务选择，不是通用行业标准。

## Ground Truth 与防循环标注

每条 answerable 题目的 `expected_evidence` 只从语料内容本身标注（document_id + 页码规则 + required_terms），不允许先跑检索再把检索结果写进标签。新增题目时必须：

1. 在 `eval/corpus/` 中确认或补充语料内容；
2. 从语料原文中挑选 `required_terms`（同一 evidence 的词必须能出现在同一个 Chunk 内）；
3. 保持 `split`、`category`、`difficulty` 取值合法（见 `src/evaluation/dataset.py` 中的允许值）；
4. 运行评估前数据集校验会自动检查"每个 answerable 证据能否在语料中定位"，定位失败会直接报错。

## 如何增加评估问题

在 `eval/dataset.jsonl` 末尾追加一行 JSON，然后运行上面的命令。校验失败会给出具体 case id 和原因。新增题目后，正式基线应重新生成并在提交说明中注明数据集版本变化。

## 注意事项

- 受控语料基准不等同于所有真实用户文档，分数只在本语料上有意义。
- 推荐阈值**不会自动写入生产配置**；是否修改生产阈值需要单独决策。
- 更换 Embedding 模型或 Chunk 参数后必须重新校准，旧基线不可直接沿用。
