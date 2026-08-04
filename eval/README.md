# 离线 RAG 检索与拒答评估基准

这个目录是**离线评估基准**：一份受控语料、一份带标注的问题集，以及围绕它们的可复现评估流程。
它回答四个问题：

1. 正确证据有没有进入 Top-K，通常排在第几名。
2. 资料内的问题有多少被错误拒答。
3. 资料外的问题有多少被错误放行。
4. 仓库默认对比阈值 `0.35` 在这份受控基准上是否合理，换成多少更好。

评估**只测检索质量和回答/拒答决策**。它不调用 LLM，不评估生成答案的质量，也不评估答案忠实度。

---

## 目录结构

```text
eval/
├── README.md               本文件
├── corpus_manifest.json    语料清单：document_id、相对路径、文件名、content_type
├── corpus/                 受控语料（3 份原创 Markdown）
│   ├── backend-architecture.md
│   ├── rag-fundamentals.md
│   └── database-basics.md
└── dataset.jsonl           60 道带标注问题，每行一个 JSON 对象
```

配套代码在仓库其他位置：

```text
src/evaluation/     评估库（models / dataset / corpus / matching / metrics / threshold / runner / report）
scripts/evaluate_rag.py  命令行入口
reports/baseline/   已提交的基线快照，用于回归对比
reports/generated/  本地临时输出（已 gitignore）
```

---

## 快速开始

```bash
# 在仓库根目录、已激活虚拟环境的前提下
python -m scripts.evaluate_rag
```

默认读取 `eval/corpus_manifest.json` 与 `eval/dataset.jsonl`，把三份报告写入 `reports/generated/`：

| 文件 | 用途 |
| --- | --- |
| `summary.json` | 机器可读汇总，含 `schema_version`，用于回归对比 |
| `cases.csv` | 每题一行的明细（UTF-8 BOM，Excel 可直接打开中文） |
| `report.md` | 人读报告：指标、分布、阈值扫描、失败案例、已知限制 |

常用参数：

```bash
python -m scripts.evaluate_rag \
  --top-k 5 \
  --threshold-start 0.20 --threshold-end 0.60 --threshold-step 0.01 \
  --comparison-threshold 0.35 \
  --false-answer-weight 3.0 --false-refusal-weight 1.0 \
  --output-dir reports/generated
```

`python -m scripts.evaluate_rag --help` 会列出全部参数与默认值。`--help` 不会加载 Embedding 模型。

退出码：

| 码 | 含义 |
| --- | --- |
| 0 | 正常完成 |
| 2 | 输入非法（清单、数据集、参数校验失败） |
| 3 | Embedding 模型不可用（例如首次运行且没有本地缓存） |

**首次运行需要联网下载模型**（`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`）。
模型缓存好之后，加上 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 可以完全离线跑。

---

## 语料说明

三份 Markdown 是**专为评估编写的原创内容**，不是运行时的用户上传资料，也不是内置课程资料的副本。
这样做有两个原因：一是标注可以精确到句子，二是语料不会随产品资料变动而漂移。

因此：**结论只适用于这份受控基准**。它不能直接外推到任意真实文档集。

语料通过生产管线加载和切分——`DocumentLoader` → `chunk_document`，参数取
`src/chunker.py` 的 `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP`，评估侧不重新定义切分逻辑。
当前 3 份文档切成 25 个 Chunk。

### `corpus_manifest.json` 格式

```json
{
  "documents": [
    {
      "document_id": "eval-backend",
      "path": "eval/corpus/backend-architecture.md",
      "filename": "backend-architecture.md",
      "content_type": "text/markdown"
    }
  ]
}
```

`path` 必须是相对路径、不含 `..`，且解析后必须落在 `eval/` 目录内。
绝对路径、路径穿越、目录外文件都会被拒绝，保证任何人 clone 之后跑出来的语料一致。

---

## 数据集格式

`dataset.jsonl` 每行一个 JSON 对象：

```json
{
  "id": "d-001",
  "split": "calibration",
  "question": "结算服务在这份资料里负责什么？",
  "answerable": true,
  "category": "direct",
  "difficulty": "easy",
  "expected_evidence": [
    {
      "document_id": "eval-backend",
      "page_number": null,
      "required_terms": ["结算服务"]
    }
  ],
  "notes": ""
}
```

字段约定：

| 字段 | 说明 |
| --- | --- |
| `id` | 全局唯一 |
| `split` | `calibration`（用于扫描和选择阈值）或 `test`（留出，本次没有参与阈值扫描；结果公开后适合作为回归基准，不再是未来完全未见的最终 holdout） |
| `answerable` | 资料内能否回答 |
| `category` | `direct` / `paraphrase` / `multi_evidence` / `out_of_scope_far` / `out_of_scope_near` |
| `difficulty` | `easy` / `medium` / `hard` |
| `expected_evidence` | 证据组数组；`answerable=false` 时必须为空数组 |
| `required_terms` | 一个证据组内的词必须**全部**出现在同一个 Chunk 里才算命中 |
| `page_number` | Markdown 语料恒为 `null`；PDF 语料可指定页码 |

校验规则（`load_dataset` + `validate_dataset_against_corpus`）：

- `answerable=true` 必须至少有一个证据组；`answerable=false` 必须没有证据组。
- `multi_evidence` 至少两个证据组，且分布在语料的不同小节。
- 每个 `document_id` 必须在语料清单里存在。
- `id` 不能重复；非法行会报出**行号**，方便定位。

### 反循环标注

标注时刻意避免“照抄检索结果”：

- 问题先写，再回语料里找证据，不是先跑检索再补标注。
- `paraphrase` 类刻意换掉原文措辞，避免字面重合带来的虚高。
- `out_of_scope_near` 是**话题相邻但资料里确实没有答案**的问题（例如问某个模块的灰度发布方案），
  专门用来暴露“看起来相关所以被放行”的失败模式。
- 证据用的是原文里真实存在的短语，但一道题的 `required_terms` 不会长到把整段抄下来。

### 当前分布

| 维度 | 分布 |
| --- | --- |
| 总量 | 60 |
| split | calibration 42 / test 18 |
| answerable | 可回答 36 / 不可回答 24 |
| category | direct 15、paraphrase 15、multi_evidence 6、out_of_scope_far 12、out_of_scope_near 12 |
| difficulty | easy 21、medium 28、hard 11 |

---

## 指标定义

**检索指标只在 `answerable=true` 的题目上计算**，资料外问题不进入任何分子或分母。

| 指标 | 定义 |
| --- | --- |
| Hit@K | Top-K 里至少命中一个证据组的题目占比 |
| Recall@K | 单题「命中证据组数 / 总证据组数」，再对题目做宏平均 |
| MRR | 第一个命中证据的排名倒数；Top-K 内完全没命中记 0 |

**决策指标**复用生产判定函数 `src.rag_service.has_sufficient_context`，规则是 `max_score >= threshold`，
评估和线上永远不会漂移。混淆矩阵：

| | 预测回答 | 预测拒答 |
| --- | --- | --- |
| 实际可回答 | TP | FN（**错误拒答**） |
| 实际不可回答 | FP（**错误放行**） | TN |

- 错误放行率 = FP / (FP + TN)
- 错误拒答率 = FN / (FN + TP)
- 分母为 0 时报 `null`，不会悄悄写一个错的数

---

## 阈值扫描与推荐规则

```text
weighted_cost = FP × false_answer_weight + FN × false_refusal_weight
```

默认 `false_answer_weight=3.0`、`false_refusal_weight=1.0`。
错误放行权重更高是**本项目的业务选择**，不是通用行业标准：资料外问题被放行更容易导致幻觉，
而错误拒答只是让用户换个问法。换一个业务场景就要重新设定这两个权重。

阈值网格用 `Decimal` 累加生成，避免 `0.35000000000000004` 这类浮点漂移，两个端点一定包含在内。

推荐阈值的选择是**确定性**的，排序规则依次是：

1. `weighted_cost` 最低
2. 错误放行更少
3. 错误拒答更少
4. 决策准确率更高
5. 阈值更高（更保守）

两条纪律：

- **只用 calibration split 选阈值。** 用 test split 选会把留出集泄漏进调参。
- **推荐阈值不会自动写进生产配置。** 脚本只报告，是否采纳是独立决策，需要单独提交。

---

## 当前基线（快照见 `reports/baseline/`）

模型 `paraphrase-multilingual-MiniLM-L12-v2`，chunk_size=300 / overlap=50，Top-5：

| 指标 | 值 |
| --- | --- |
| Hit@1 / Hit@3 / Hit@5 | 0.5556 / 0.7778 / 0.9167 |
| Recall@1 / Recall@3 / Recall@5 | 0.5000 / 0.7222 / 0.9028 |
| MRR | 0.6894 |

| 场景 | 决策准确率 | 错误放行率 | 错误拒答率 |
| --- | ---: | ---: | ---: |
| calibration @ 0.35（仓库默认对比值） | 0.8810 | 0.2941 | 0.0000 |
| calibration @ 0.49（推荐） | 0.9524 | 0.0588 | 0.0400 |
| test @ 0.35（仓库默认对比值） | 0.7778 | 0.5714 | 0.0000 |
| test @ 0.49（推荐） | 0.8333 | 0.1429 | 0.1818 |

**关于 0.35 是否合理**：`0.35` 是仓库代码里的拒答默认值，仅作为报告里的对比基准；它**不一定等于**部署环境实际生效的阈值（部署值可能由 `RAG_MIN_RELEVANCE_SCORE` 覆盖）。在这份基准上，把仓库默认对比值 `0.35` 当阈值不合理，它太松。test split 上 0.35 会放行 57% 的资料外问题。
推荐阈值 0.49 把错误放行率压到 14%，代价是 18% 的错误拒答。

**但阈值不是全部问题。** 两组分数在 `[0.4377, 0.5879]` 区间重叠，
其中资料内 19 题、资料外 4 题落在重叠带内。**这个数量只是一个描述性统计**：它说明有多少题的得分落在两组都覆盖的区间里，并不等于"最少必然出错的题数"。事实上不存在能实现零错误的单一全局阈值（否则两组分数应完全分离），但具体某个阈值到底会错几题，要看每题真正的可回答性，而不是区间里的题数。继续压低两类错误需要改进检索本身（更好的 Embedding、Reranker、更细的切分），
而不是继续微调阈值。

---

## 回归对比怎么做

改动 Chunk 参数、Embedding 模型、Reranker 或 pgvector 之后：

```bash
python -m scripts.evaluate_rag --output-dir reports/generated
```

然后把 `reports/generated/summary.json` 和 `reports/baseline/summary.json` 对比。
重点看 `retrieval_metrics.overall`、`test_metrics_recommended`、`score_range_overlap` 三块。
`summary.json` 带 `schema_version`，字段变动时能识别出来。

在 Fake Embedding 路径下，同一份输入跑两次的输出是**逐字节一致**的（测试里有断言），因此该路径上的任何 diff 都来自真实改动。
但使用真实 Embedding（`sentence-transformers`）时，跨机器、跨依赖版本、跨运行可能引入末位数值差异，这类差异不应直接解释为产品效果变化；做跨机器对比前需先对齐依赖版本。

确认新版本更好之后，再把新报告拷进 `reports/baseline/` 作为新基线，作为单独一次提交。

---

## 已知限制

- 语料是受控基准，规模有限，不代表所有真实用户文档。
- 第一版 60 题，指标存在抽样波动，小数点后第二位不要过度解读。
- 不评估生成答案质量，也不评估答案忠实度。Hit@5 高不代表最终答案一定正确。
- 索引只有 25 个 Chunk，Top-5 覆盖了 20% 的语料，Hit@5 会偏乐观。
- 错误放行与错误拒答的权重是业务选择，不是通用标准。
- `page_number` 匹配逻辑写了但当前语料全是 Markdown，实际没有被覆盖到真实页码场景。

---

## 测试

评估相关测试**完全离线**：`tests/conftest.py` 把 `sentence_transformers` 替换成会直接失败的桩，
测试用 `tests/evaluation_helpers.py` 里的 `FakeEmbeddingService`（字符 bigram 哈希）代替真实模型。

```bash
pytest tests/test_evaluation_*.py -q
```
