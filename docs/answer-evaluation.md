# 回答质量评估框架

本框架为最终 RAG 回答提供**确定性、可审计、可复现**的质量评估，与现有的检索质量评估
（`scripts/evaluate_rag.py`）互补。它回答这样一个问题：*检索到的证据就算都对，模型生成的
最终答案真的覆盖了必要事实、引用合法、并且没有说出与语料矛盾的话吗？*

> 重要界限：这是基于**人工标注事实和引用**的确定性检查，不是完整语义正确性证明。
> 它不能发现所有幻觉，不等同于人工评审，也不等同于 LLM Judge。

---

## 为什么检索正确不等于答案正确

现有的 `scripts/evaluate_rag.py` 只评估检索环节：正确证据有没有进入 Top-K、排在第几位、
资料内问题有没有被误拒、资料外问题有没有被误放。它**不检查最终答案文本**。

一个系统可能 Hit@5 很高，却仍然生成错误答案：

- 检索到了证据，但模型没有把必要事实写进答案；
- 答案写了事实，却用了不存在的来源编号；
- 答案引用了一个来源，但该来源并不支持它声称的事实；
- 答案包含与语料直接冲突的说法（已知矛盾）。

本框架把这些行为变成可计算、可回归的指标，而不是依赖人工抽查或主观印象。

---

## 评估边界（必须明确）

- 它**不是**完整语义正确性证明。
- 它**不能**发现所有幻觉。
- 它**不等同**于人工评审。
- 它**不等同**于 LLM Judge。
- 它**只检查**已标注的必要事实、引用编号、来源支持和已知矛盾短语。
- `reference_answer` 只用于**人工审计和报告展示**，不参与自动语义相似度评分。
- **不使用** BLEU、ROUGE 或任何字符串相似度冒充事实正确性。
- **不把**词面覆盖率称为完整 faithfulness。
- **不声称**结果代表生产环境的真实用户分布。

建议使用的指标措辞：

- `annotated fact coverage`（标注事实覆盖率）
- `supported fact citation coverage`（来源支持引用覆盖率）
- `citation validity`（引用合法率）
- `known contradiction detection`（已知矛盾检测）
- `strict annotated pass rate`（严格标注通过率）

禁止将指标命名为：绝对正确率、完整忠实度、无幻觉率、事实正确率。

---

## 目录结构

```text
src/evaluation/
├── answer_models.py       数据模型（frozen dataclass）
├── answer_annotations.py  标注加载与严格校验
├── answer_responses.py    回答结果 JSONL 加载与严格校验
├── answer_citations.py    [来源N] 引用解析器
├── answer_metrics.py      指标计算（纯函数）
├── answer_runner.py       Runner：离线 / --live 两种模式
└── answer_report.py       报告渲染（summary.json / cases.jsonl / report.md）

scripts/evaluate_answers.py   命令行入口
eval/answer_annotations.jsonl 60 题回答质量标注
docs/answer-evaluation.md     本文档
tests/fixtures/answer_evaluation/  独立的确定性测试 Fixture
```

---

## 标注 Schema

`eval/answer_annotations.jsonl` 每行一个 JSON 对象，覆盖 `eval/dataset.jsonl` 全部 60 个 case：

```json
{
  "case_id": "d-001",
  "reference_answer": "不可以。Router 层不应该承载业务逻辑。",
  "required_facts": [
    {
      "fact_id": "router-no-business-logic",
      "accepted_phrases": ["Router 层不应该承载业务逻辑"],
      "supporting_evidence_indexes": [1]
    }
  ],
  "forbidden_phrases": ["Router 层可以直接写业务逻辑"],
  "notes": "reference_answer 供人工审计，不参与语义相似度评分。"
}
```

字段约定：

| 字段 | 说明 |
| --- | --- |
| `case_id` | 必须存在于 dataset，全文件唯一 |
| `reference_answer` | 人工审计用，**不参与语义相似度评分** |
| `required_facts` | `answerable=true` 时至少一个；`answerable=false` 时必须为空 |
| `required_facts[].fact_id` | 非空，case 内唯一，稳定 kebab-case / snake_case |
| `required_facts[].accepted_phrases` | 非空数组；每项非空、不重复；至少一个出现在规范化后的 reference_answer 中 |
| `required_facts[].supporting_evidence_indexes` | 1-based，指向该 case 的 `expected_evidence`；不得为 0、越界、重复 |
| `forbidden_phrases` | 只标注明确错误或与语料直接冲突的说法；不确定时保持为空 |
| `notes` | 字符串，可为空 |

加载 `annotations` 时**联合验证原始 dataset**：

- annotation case ids 必须与 dataset 完全一致（不允许缺失、多余、重复）。
- 输出顺序以 dataset 顺序为准。
- `answerable` case 必须有 `required_facts`；`unanswerable` case 不允许有。
- `unanswerable` case 的 `reference_answer` 必须等于生产固定拒答 `INSUFFICIENT_CONTEXT_ANSWER`。
- `expected_evidence` 的每一组都必须至少被一个 fact 引用。
- `forbidden_phrases` 不得出现在 `reference_answer` 中。
- 非法 JSON 报错包含行号，错误消息不含绝对路径。

标注文件不修改现有 `eval/dataset.jsonl` 的 Schema，也不把回答质量字段塞进检索 dataset。

---

## 回答结果 JSONL Schema（离线评分输入）

```json
{
  "case_id": "d-001",
  "answer_status": "answered",
  "answer": "不可以……[来源1]",
  "sources": [
    {
      "rank": 1,
      "score": 0.8123,
      "text": "Router 层不应该承载业务逻辑……",
      "chunk_index": 0,
      "document_id": "eval-backend",
      "filename": "backend.md",
      "page_number": null
    }
  ],
  "max_relevance_score": 0.8123,
  "relevance_threshold": 0.35,
  "retrieval_elapsed_ms": 2.1,
  "generation_elapsed_ms": 420.5,
  "total_elapsed_ms": 422.6,
  "reranker_applied": false,
  "reranker_fallback": false
}
```

严格校验（带行号、不含绝对路径）：

- `case_id` 必须存在于 dataset，不得重复；response case ids 必须与 annotations 完全一致。
- `answer_status` 仅允许 `answered` / `insufficient_context`。
- `sources[].rank` 必须从 1 开始**连续递增**（1..N，与数组位置完全一致）；
  `score` 为 `[-1, 1]` 有限数字；`text` / `document_id` 非空；
  `page_number` 为正整数或 null；`chunk_index` 非负整数。
- `max_relevance_score`：无 reranker 时必须与最终返回来源的最高检索分数一致；
  启用 Reranker 时允许高于最终来源的最高分（见下文"Reranker 与 max_relevance_score"）；
  无 sources 时允许 null；不允许 NaN/Infinity。
- `relevance_threshold` 为 `[0, 1]` 有限数字。
- 耗时字段非负有限；布尔字段必须是真正的 bool（不接受 0/1）。

### sources rank 连续性

`[来源N]` 的编号直接映射到 `sources[N-1]`，因此 rank 必须与数组位置完全一致：
`1, 2, 3, ..., N`。`[1, 3]`、`[1, 2, 4]`、`[2, 3]`、`[1, 1]` 都会被拒绝，否则引用编号
会与实际来源错位。

### Reranker 与 max_relevance_score

生产 RetrievalService 中 `max_relevance_score` 是**向量候选池**中的最高 retrieval score，
它是拒答决策的依据，Reranker 不会改变它。Reranker 重排后返回的 final Top-K 可能不包含
该最高分来源（高分候选被排到 K 名之后）。

因此：

- `reranker_applied=false`（含 fallback 保持原向量顺序）：final Top-K 包含最高向量候选，
  `max_relevance_score` 必须与最终来源最高分一致（允许 `1e-9` 容差）。
- `reranker_applied=true`：`max_relevance_score` 允许高于最终来源最高分（候选池最高分
  可能被重排移出 Top-K），但**不得低于**任何最终返回来源的检索分数。

---

## 引用解析器

生产 Prompt 要求模型使用 `[来源N]` 格式。解析器只接受**严格格式**：

- 合法：`[来源1]`、`[来源2]`、`[来源12]`，支持连续 `[来源1][来源2]`。
- 正则：`r"\[来源([1-9]\d*)\]"`。
- 格式错误（记为 malformed）：`[来源0]`、`[来源01]`、`[来源]`、`[来源A]`、`[source1]`、
  `[SOURCE1]`、`【来源1】`、`[来源 1]`、`[来源1, 来源2]`。

**普通 Markdown 方括号不是引用**。`[FastAPI]`、`[Python]`、`[Service]`、`[1]`、`[abc]`
是普通文本，**不会被**当作 malformed citation。只有以 `来源` / `source`（不区分大小写）
开头、形似引用的片段才会进入 malformed 检查。

重复引用允许，分别统计 occurrence 与 unique source。解析器不修改原始答案，
也不把完整答案写入异常消息。

**来源编号合法性**：引用编号 1-based，对应 response `sources` 顺序。合法条件
`1 <= source_number <= len(sources)`。越界（如 `[来源4]` 但只有 3 条 sources）记为
`invalid_citation_numbers = [4]`，记录但**不抛出**异常，不终止评估。

---

## 指标定义

### 事实覆盖（annotated fact coverage）

规范化是透明、保守的文本规范化（strip、NFKC、casefold、合并连续空白、
移除 CJK 相邻空白、统一常见全角标点）。禁止模糊语义匹配、编辑距离、Embedding 相似度、
LLM 判断、同义词扩展、中文分词。

**引用标记不参与事实匹配**。事实覆盖与矛盾短语检测使用去除全部 source-like citation
markup（`[来源N]` 及所有 malformed 形似片段）后的答案副本，以中性分隔符替换而不是直接
删除，避免 `"Service[来源1]层"` 被拼接成 `"Service层"` 造成人为匹配。普通 Markdown 方括号
（如 `[Python]`）保留原样。引用数量、编号、malformed 统计与原始答案输出不受影响。

一个 fact 被覆盖，**当且仅当**任一 `accepted_phrase` 经相同规范化后是（去除引用标记的）
答案规范化文本的子串。这是 lexical annotated fact coverage，不是完整语义正确性。

### 来源支持（supported fact citation coverage）


对于每个被覆盖的 fact：

1. 取 `fact.supporting_evidence_indexes`，映射到 `EvaluationCase.expected_evidence`；
2. 取答案中所有**合法**引用的 source；
3. 一个 source 满足任一 supporting evidence 当且仅当：`document_id` 匹配、
   `page_number` 匹配或标注为 null、`source.text` 满足现有 `text_matches_expectation`；
4. 至少一个被引用 source 满足 → `grounded_by_citation=true`。

逐事实输出：`fact_id`、`covered`、`matched_phrase`、`supporting_evidence_indexes`、
`supporting_citation_numbers`、`grounded_by_citation`。

**V1 限制**：使用答案级引用集合判断标注事实是否有支持来源，**不做句子级
claim-to-citation 对齐**。不实现脆弱的句子级引用归属。

### 已知矛盾检测

任一 `forbidden_phrase` 出现在规范化答案中即命中，记录命中的原始短语，**不在日志中
输出整段答案**。

### 决策正确性

| 实际 | answer_status | 判定 |
| --- | --- | --- |
| answerable=true | answered | 正确回答 |
| answerable=true | insufficient_context | **错误拒答** |
| answerable=false | insufficient_context | 正确拒答 |
| answerable=false | answered | **错误放行** |

以结构化字段 `answer_status` 为准，**不根据答案文本猜测状态**。

### strict_pass（严格标注通过）

`answerable` case 同时满足：

- `answer_status == answered`
- `fact_coverage == 1.0`（全部标注事实覆盖）
- `grounded_fact_coverage == 1.0`（全部覆盖事实都有支持来源）
- `invalid_citation_occurrence_count == 0`
- `malformed_citation_count == 0`
- `forbidden_phrase_hits` 为空
- 至少存在一个合法引用

`unanswerable` case：

- `answer_status == insufficient_context`
- 答案与 `INSUFFICIENT_CONTEXT_ANSWER` **完全一致**
- 不包含严格有效引用、不包含 malformed citation

标准不悄悄放宽。这是严格的 annotated pass，不是绝对正确率。

### 汇总指标

| 指标 | 定义 |
| --- | --- |
| decision_accuracy | 决策正确数 / 总题数 |
| false_answer_rate | 错误放行数 / 实际不可答题数 |
| false_refusal_rate | 错误拒答数 / 实际可答题数 |
| average_fact_coverage | 仅以 answerable 为分母；false refusal 记为 0，不允许排除失败案例美化 |
| average_grounded_fact_coverage | 同上，针对 grounded facts |
| complete_fact_coverage_rate | 全部事实覆盖的 answerable 题占比 |
| complete_grounded_fact_coverage_rate | 全部事实 grounded 的 answerable 题占比 |
| contradiction_free_rate | 无矛盾题占比 |
| citation_validity_rate | 合法引用出现次数 / 全部严格引用出现次数（**occurrence 加权**，非每题简单平均）；无引用为 null |
| invalid_citation_case_rate | 含越界引用的题占比 |
| malformed_citation_case_rate | 含格式错误引用的题占比 |
| strict_pass_rate | 严格标注通过率 |

分组输出：`overall`、`by_split`、`by_category`、`by_difficulty`。

---

## 使用方式

### 模式 A：离线评分已有回答（完全确定性）

```bash
python -m scripts.evaluate_answers \
  --dataset eval/dataset.jsonl \
  --annotations eval/answer_annotations.jsonl \
  --responses <path/to/responses.jsonl> \
  --output-dir reports/generated/answer-quality \
  --run-name my-run \
  --model-label my-model
```

该模式**不调用** Embedding、Reranker、LLM、数据库、网络。相同输入与相同代码，
`summary.json`、`cases.jsonl`、`report.md` **逐字节一致**，不写时间戳。

### 模式 B：显式 live 生成（必须 `--live`）

```bash
python -m scripts.evaluate_answers --live \
  --manifest eval/corpus_manifest.json \
  --dataset eval/dataset.jsonl \
  --annotations eval/answer_annotations.jsonl \
  --output-dir reports/generated/answer-quality/manual-live \
  --run-name manual-live \
  --model-label local-configured-model \
  --top-k 3 \
  --threshold 0.35
```

- 只有显式传入 `--live` 才允许调用真实 Embedding 和 LLM。
- 文档 Embedding 只计算一次；每题只 `encode_query` 一次。
- 使用生产 `KnowledgeIndex`、`EmbeddingService`、`PromptBuilder`、`GenerationService`、
  `RagService` 与 `has_sufficient_context` 语义，不复制另一套拒答规则。
- `generation service` 在 `finally` 中 close。
- 不使用数据库；不自动读取/写入生产数据；不自动修改阈值（live 用显式 CLI threshold）。
- V1 live 模式使用 Memory KnowledgeIndex，不加入 pgvector 模式评估。
- 不打印 Prompt、API Key、Authorization Header；Provider 错误使用现有稳定异常。

### 参数

`--manifest`（默认 `eval/corpus_manifest.json`）、`--dataset`（默认 `eval/dataset.jsonl`）、
`--annotations`（默认 `eval/answer_annotations.jsonl`）、`--responses`、`--output-dir`
（默认 `reports/generated/answer-quality`）、`--live`、`--top-k`（默认 3）、`--threshold`
（默认 0.35）、`--embedding-model`、`--run-name`、`--model-label`、`--prompt-version`
（默认 `prompt-builder-v1`）。

模式规则：

- `--live` 与 `--responses` **互斥**，两者必须选其一；不允许无意调用真实 LLM。
- `--responses` 模式不实例化模型；`--help` 不加载 torch / sentence-transformers / 模型。
- 路径相对仓库根解析；报告中只显示仓库相对路径或文件名。

退出码：`0` 成功；`2` 输入或标注无效；`3` 回答结果无效；`4` Embedding 不可用；
`5` LLM 配置无效；`6` LLM 调用失败；`7` 报告写入失败。

---

## 输出文件

每次评估输出（输出目录由 `--output-dir` 指定）：

| 文件 | 说明 |
| --- | --- |
| `summary.json` | 机器可读汇总：schema_version、evaluation_type、run_configuration、dataset_counts、aggregate_metrics、metrics_by_split/category/difficulty、latency_summary、各类失败 case id 列表、limitations |
| `cases.jsonl` | 每题一行，dataset 顺序，含逐题指标、fact 明细、引用明细、答案与 reference_answer |
| `report.md` | 人读报告：运行配置、数据集统计、指标定义、整体/分组指标、引用质量、事实覆盖、严格通过率、延迟统计、失败案例摘要（最多 5 条）、方法限制、复现命令 |

live 模式额外输出 `responses.jsonl`（dataset 顺序、UTF-8、每行一个严格 JSON、
`allow_nan=False`；保留答案、来源、耗时、reranker 字段；**不保存** Prompt、API Key、
HTTP Header、完整 Provider 响应、traceback）。

所有报告：`allow_nan=False`、数值稳定舍入、key 顺序稳定、无绝对路径、无时间戳、
无 API Key、无 Authorization、无 Provider 原始错误、无完整 Prompt。
`reports/generated/` 继续不进入 Git；不自动提交 live 输出。

**路径安全**：离线模式的 `run_configuration` 记录 `dataset_path`、`annotations_path`、
`responses_path` 三个字段（`responses_path` 记录实际评分的输入文件，不为 null）。
路径只以仓库相对路径（文件位于仓库内时，如 `tests/fixtures/answer_evaluation/dataset.jsonl`）
或裸文件名（文件位于仓库外时，如 `responses.jsonl`）记录，**绝不包含本地绝对路径**。

---

## 如何比较两个模型

不同运行写入不同目录，例如：

```text
reports/generated/answer-quality/model-a-top3/
reports/generated/answer-quality/model-b-top5/
```

对比 `summary.json` 中的：

- `strict_pass_rate`
- `average_fact_coverage`
- `average_grounded_fact_coverage`
- `false_answer_rate` / `false_refusal_rate`
- `citation_validity_rate`
- `latency_summary`

### 如何比较两个 top_k

固定 `--model-label` 与 `--prompt-version`，分别用 `--top-k 3` / `--top-k 5` 跑两次，
对比汇总指标与逐题 `cases.jsonl`。

### 如何比较两个 Prompt 版本

`--prompt-version` 只作为报告标签记录，**不修改任何生产 Prompt**。生产 Prompt 变更后，
用新版本重新跑 live 评估并记录 `prompt_version` 标签，再与旧版本报告对比。
本框架本身不自动优化 Prompt。

---

## 如何人工检查失败案例

`summary.json` 提供失败 case id 列表：`strict_failure_case_ids`、`false_answer_case_ids`、
`false_refusal_case_ids`、`incomplete_fact_case_ids`、`ungrounded_fact_case_ids`、
`invalid_citation_case_ids`、`malformed_citation_case_ids`、`contradiction_case_ids`。

打开 `cases.jsonl` 对应行，对照 `answer` 与 `reference_answer`（人工审计用）、
`facts` 明细与 `sources`，即可判断是模型问题、标注问题还是检索问题。

---

## 评估限制（limitations）

- 本框架是基于人工标注事实和引用的确定性检查，不是完整语义正确性证明。
- 它不能发现所有幻觉，也不等同于人工评审或 LLM Judge。
- 它只检查已标注的必要事实、引用编号与已知矛盾短语。
- `reference_answer` 只用于人工审计，不参与任何语义相似度评分；不使用 BLEU、ROUGE
  或字符串相似度冒充事实正确性。
- 来源支持只针对已标注的 required_facts 判断；V1 使用答案级引用集合，不做句子级
  claim-to-citation 对齐。
- 结果代表这份受控标注集的 annotated 表现，不声称代表生产环境的真实用户分布。

---

## CI 为什么不调用真实 LLM

CI 环境设置 `LLM_API_KEY=""`、`LLM_BASE_URL=""`、`LLM_MODEL=""`、`HF_HUB_OFFLINE=1`、
`TRANSFORMERS_OFFLINE=1`。Backend job 增加一个确定性的离线 Smoke Step，只用
`tests/fixtures/answer_evaluation/` 的 Fixture 运行 `--responses` 离线模式，并断言三个
报告存在、JSON 合法、`evaluation_type == "answer_quality"`、报告包含方法限制与 strict 说明。

Smoke test 证明：不加载 Embedding、不调用 LLM、不访问网络、不依赖 PostgreSQL、
不使用真实 Reranker。CI 不使用 `continue-on-error`、`|| true`、不跳过失败测试、
不添加真实密钥、不下载真实模型。

---

## `reference_answer` 为什么不做字符串相似度评分

词面覆盖率不能证明事实正确。`reference_answer` 是人工编写的理想答案，与模型输出做
BLEU/ROUGE 只会奖励措辞相似而非事实正确，还会把改写能力强的模型误判为差。因此
`reference_answer` 只用于人工审计与报告展示；自动评分只看标注的 `required_facts` 与引用。

---

## `reports/generated` 为什么不提交

`reports/generated/` 是本地临时输出（gitignore）。它包含每次运行的具体答案文本与耗时，
会随环境与输入变化，不应进入版本历史。已提交的基线快照在 `reports/baseline/`。
live 输出（`responses.jsonl` 与报告）同样不自动提交。

---

## 如何安全处理 API Key

- 密钥只通过环境变量或本地 `.env` 提供，`.env` 与真实 `.env` 绝不提交到 Git。
- live 模式通过 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 环境变量读取，
  不在日志、报告或 `responses.jsonl` 中输出密钥或 Authorization Header。
- 报告与 responses 文件只包含答案、来源、耗时与 reranker 字段。
