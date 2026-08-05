# RAG Reranking A/B 评估报告

## 实验目标

验证可选的两阶段检索（CrossEncoder Reranker）能否：
1. 正确证据稳定进入更大的候选池
2. Reranker 把正确证据从候选池后部提升到 Top-1 / Top-3
3. paraphrase 问题明显改善
4. 原有 direct 和 multi-evidence 问题不退化
5. Reranker 失败时安全回退到 vector-only
6. 额外模型和延迟成本是否值得

## 运行配置

- Embedding 模型: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Reranker 模型: `models/mmarco-mMiniLMv2-L12-H384-v1`
- Reranker backend: `cross_encoder`
- Real model run: 是
- Model revision: N/A
- Candidate Top-K: 15
- Final Top-K: 5
- 数据集题目数: 60

## 数据集角色

- **calibration**: 用于阈值校准，在本 A/B 中也参与指标计算。
- **test**: 已公开，作为回归基准使用，不是完全未见的最终结果。
- 正式比较多个 Reranker 方案时，需要新增独立 final holdout。

## 候选池质量

| 指标 | 值 |
| --- | ---: |
| Candidate Hit@5 | 0.9167 |
| Candidate Hit@10 | 0.9722 |
| Candidate Hit@15 | 1.0000 |
| Candidate Recall@5 | 0.9028 |
| Candidate Recall@10 | 0.9722 |
| Candidate Recall@15 | 1.0000 |
| Candidate MRR | 0.7005 |

## Vector-only 基线

| 指标 | 值 |
| --- | ---: |
| Hit@1 | 0.5556 |
| Hit@3 | 0.7778 |
| Hit@5 | 0.9167 |
| Recall@1 | 0.5000 |
| Recall@3 | 0.7222 |
| Recall@5 | 0.9028 |
| MRR | 0.6894 |

## Reranked 结果

| 指标 | 值 |
| --- | ---: |
| Hit@1 | 0.8611 |
| Hit@3 | 1.0000 |
| Hit@5 | 1.0000 |
| Recall@1 | 0.7778 |
| Recall@3 | 0.9722 |
| Recall@5 | 0.9861 |
| MRR | 0.9259 |

## 整体指标差值

| 指标 | Vector | Reranked | Delta |
| --- | ---: | ---: | ---: |
| Hit@1 | 0.5556 | 0.8611 | +0.3056 (+55.00%) |
| Hit@3 | 0.7778 | 1.0000 | +0.2222 (+28.57%) |
| Hit@5 | 0.9167 | 1.0000 | +0.0833 (+9.09%) |
| Recall@5 | 0.9028 | 0.9861 | +0.0833 (+9.23%) |
| MRR | 0.6894 | 0.9259 | +0.2366 (+34.32%) |

## 按 category 对比

| Category | Vector Hit@1 | Reranked Hit@1 | Vector MRR | Reranked MRR |
| --- | ---: | ---: | ---: | ---: |
| direct | 0.6667 | 0.8000 | 0.8056 | 0.8889 |
| multi_evidence | 0.6667 | 1.0000 | 0.8056 | 1.0000 |
| paraphrase | 0.4000 | 0.8667 | 0.5267 | 0.9333 |

## 按 difficulty 对比

| Difficulty | Vector Hit@1 | Reranked Hit@1 | Vector MRR | Reranked MRR |
| --- | ---: | ---: | ---: | ---: |
| easy | 0.6667 | 0.7778 | 0.7870 | 0.8704 |
| hard | 0.6667 | 1.0000 | 0.8056 | 1.0000 |
| medium | 0.4762 | 0.8571 | 0.6143 | 0.9286 |

## 重点 paraphrase 对比

| 指标 | Vector | Reranked |
| --- | ---: | ---: |
| Hit@1 | 0.4000 | 0.8667 |
| Hit@3 | 0.5333 | 1.0000 |
| Hit@5 | 0.8000 | 1.0000 |
| MRR | 0.5267 | 0.9333 |

## p-001 / p-010 / p-012

### p-001
- 问题: 接口函数里堆满了分支判断和流程编排，这种写法合适吗？
- 是否进入 Candidate Top-15: True
- Candidate rank: 6
- Vector rank: miss
- Reranked rank: 1
- Rank change: N/A
- Top-5 vector: miss → Top-5 reranked: hit

### p-010
- 问题: 一笔转账在中途失败，怎么保证账目不会对不上？
- 是否进入 Candidate Top-15: True
- Candidate rank: 7
- Vector rank: miss
- Reranked rank: 2
- Rank change: N/A
- Top-5 vector: miss → Top-5 reranked: hit

### p-012
- 问题: 一个只有两种取值的状态列，值不值得单独加索引？
- 是否进入 Candidate Top-15: True
- Candidate rank: 11
- Vector rank: miss
- Reranked rank: 2
- Rank change: N/A
- Top-5 vector: miss → Top-5 reranked: hit

## 改善案例

| Case ID | Vector Rank | Reranked Rank | Rank Change |
| --- | ---: | ---: | ---: |
| d-001 | 2 | 1 | -1 |
| d-002 | 4 | 2 | -2 |
| d-003 | 3 | 1 | -2 |
| d-010 | 2 | 1 | -1 |
| d-011 | 2 | 1 | -1 |
| p-001 | miss | 1 | 0 |
| p-002 | 5 | 1 | -4 |
| p-004 | 4 | 1 | -3 |
| p-005 | 4 | 1 | -3 |
| p-006 | 2 | 1 | -1 |
| p-009 | 2 | 1 | -1 |
| p-010 | miss | 2 | 0 |
| p-012 | miss | 2 | 0 |
| p-015 | 5 | 1 | -4 |
| m-003 | 3 | 1 | -2 |
| m-005 | 2 | 1 | -1 |

## 退化案例

| Case ID | Vector Rank | Reranked Rank | Rank Change |
| --- | ---: | ---: | ---: |
| d-004 | 1 | 3 | 2 |
| d-007 | 1 | 2 | 1 |

## 未变案例

- unchanged 包括两个分支排名相同，以及两个分支都未命中（vector 与 reranked 均 miss）的案例。
- 每个可回答问题必须且只能属于 improved / regressed / unchanged 之一。
- 未变案例（18）: d-005, d-006, d-008, d-009, d-012, d-013, d-014, d-015, p-003, p-007, p-008, p-011, p-013, p-014, m-001, m-002, m-004, m-006

## 拒答决策一致性

- 决策一致性检查: 通过
- 对每个案例分别独立计算 vector 分支与 reranked 分支的候选池最高向量分数（vector_max_retrieval_score / reranked_max_retrieval_score），再调用生产函数 has_sufficient_context 在对比阈值与推荐阈值下执行真实断言。两分支分数均由完整候选池独立导出；若重排阶段丢失候选或修改检索分数导致两分支决策不一致，本检查会失败并阻止启用建议。

## Reranker 应用与回退

- 应用 Reranker 的案例数: 60
- 发生回退的案例数: 0
- 回退率: 0.0000

## 延迟说明

本次运行未启用延迟测量（`--include-latency` 默认关闭）。
- 延迟数据依赖设备和运行环境，不写入确定性质量 baseline。

## 可重复性边界

- Fake Reranker 模式下，输出逐字节稳定。
- 真实 Reranker 基线在相同代码、模型、依赖版本和环境下才稳定。
- 延迟数据不写入提交的确定性 baseline。

## 已知限制

- 语料规模有限（25 Chunk），指标存在抽样波动。
- 不评估生成答案质量，只评估检索排序。
- test split 已公开，作为回归基准使用。
- Reranker 分数语义不同于向量相似度，不可直接比较或复用阈值。
- 延迟结果依赖设备和运行环境。

## 是否建议启用生产 Reranker

**建议启用**，满足以下条件：

- Reranked Hit@1 > Vector Hit@1
- Reranked MRR > Vector MRR
- Paraphrase MRR 提升 0.4067
- Hit@5 无明显退化
- Direct Hit@5 无明显退化
- 退化案例数 2 在上限内
- 拒答决策一致性检查通过

- 只有 backend=cross_encoder 且 real_model_run=true 的真实 CrossEncoder 完整运行才可能产生启用建议；FakeReranker 或测试替身永远不会产生生产启用建议。

项目验收标准（非行业标准）：
- Hit@1 绝对提升 >= 0.05
- MRR 绝对提升 >= 0.03
- Paraphrase MRR 提升 >= 0.05
- Hit@5 退化 <= 0.01
- 退化案例数不超过 5
- 即使达到标准，也不自动修改生产配置。
