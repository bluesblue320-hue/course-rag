"""Render evaluation runs as JSON, CSV, and Markdown reports."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

from src.evaluation.dataset import count_by
from src.evaluation.metrics import (
    DecisionMetrics,
    RetrievalMetrics,
    ScoreDistribution,
    predict_answerable,
)
from src.evaluation.models import EvaluationResult
from src.evaluation.runner import EvaluationRun
from src.evaluation.threshold import ThresholdCandidate

SCHEMA_VERSION = "1.0.0"
SUMMARY_FILENAME = "summary.json"
CASES_FILENAME = "cases.csv"
REPORT_FILENAME = "report.md"
MAX_FAILURE_EXAMPLES = 5

# Stating each split's role in the artifacts themselves keeps the methodology
# limits attached to the numbers, instead of only living in a README that a
# future reader may never open.
SPLIT_ROLES: tuple[tuple[str, str], ...] = (
    (
        "calibration",
        "用于扫描和选择相似度阈值。",
    ),
    (
        "test",
        "在本次运行中，用于评估由 calibration 选出的固定阈值；"
        "它没有参与本次阈值扫描。",
    ),
    (
        "regression",
        "test 结果公开后应视为已发布的回归测试集，"
        "适合用来发现明显效果退化。",
    ),
    (
        "future_final_holdout",
        "正式比较多个 Embedding、Chunk 或 Reranker 方案时，"
        "需要新增一份从未用于开发决策的独立 final holdout，"
        "或采用嵌套交叉验证等更严格的方法。",
    ),
)

REPRODUCIBILITY_NOTES: tuple[tuple[str, str], ...] = (
    (
        "fake_embedding",
        "自动化测试证明：在 Fake Embedding、相同输入和相同代码下，"
        "JSON、CSV 与 Markdown 输出逐字节稳定。",
    ),
    (
        "real_embedding",
        "真实 Embedding 基线只有在相同代码、相同模型、相同依赖版本"
        "和相近运行环境下才应当稳定。",
    ),
    (
        "environment_drift",
        "不同设备或依赖版本造成的末位数值差异，"
        "不应直接解释为产品效果变化。",
    ),
)

CASES_CSV_COLUMNS: tuple[str, ...] = (
    "case_id",
    "split",
    "category",
    "difficulty",
    "question",
    "answerable",
    "max_relevance_score",
    "first_relevant_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "reciprocal_rank",
    "predicted_at_comparison_threshold",
    "predicted_at_recommended_threshold",
    "decision_correct_comparison",
    "decision_correct_recommended",
    "top_source_document",
    "top_source_page",
    "top_source_score",
)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _retrieval_metrics_payload(metrics: RetrievalMetrics) -> dict[str, object]:
    return {
        "case_count": metrics.case_count,
        "hit_at_1": _round(metrics.hit_at_1),
        "hit_at_3": _round(metrics.hit_at_3),
        "hit_at_5": _round(metrics.hit_at_5),
        "recall_at_1": _round(metrics.recall_at_1),
        "recall_at_3": _round(metrics.recall_at_3),
        "recall_at_5": _round(metrics.recall_at_5),
        "mean_reciprocal_rank": _round(metrics.mean_reciprocal_rank),
    }


def _decision_metrics_payload(metrics: DecisionMetrics) -> dict[str, object]:
    return {
        "threshold": _round(metrics.threshold),
        "case_count": metrics.case_count,
        "true_positive": metrics.true_positive,
        "false_negative": metrics.false_negative,
        "true_negative": metrics.true_negative,
        "false_positive": metrics.false_positive,
        "decision_accuracy": _round(metrics.decision_accuracy),
        "answerable_precision": _round(metrics.answerable_precision),
        "answerable_recall": _round(metrics.answerable_recall),
        "false_answer_rate": _round(metrics.false_answer_rate),
        "false_refusal_rate": _round(metrics.false_refusal_rate),
        "answer_rate": _round(metrics.answer_rate),
        "refusal_rate": _round(metrics.refusal_rate),
    }


def _score_distribution_payload(
    distribution: ScoreDistribution,
) -> dict[str, object]:
    return {
        "count": distribution.count,
        "min": _round(distribution.minimum),
        "max": _round(distribution.maximum),
        "mean": _round(distribution.mean),
        "median": _round(distribution.median),
        "p25": _round(distribution.p25),
        "p75": _round(distribution.p75),
    }


def _threshold_table_payload(
    candidates: tuple[ThresholdCandidate, ...],
) -> list[dict[str, object]]:
    return [
        {
            "threshold": _round(candidate.threshold),
            "true_positive": candidate.metrics.true_positive,
            "false_negative": candidate.metrics.false_negative,
            "true_negative": candidate.metrics.true_negative,
            "false_positive": candidate.metrics.false_positive,
            "decision_accuracy": _round(candidate.metrics.decision_accuracy),
            "false_answer_rate": _round(candidate.metrics.false_answer_rate),
            "false_refusal_rate": _round(candidate.metrics.false_refusal_rate),
            "weighted_cost": _round(candidate.weighted_cost),
        }
        for candidate in candidates
    ]


def retrieval_failures(run: EvaluationRun) -> tuple[EvaluationResult, ...]:
    """Return answerable cases whose evidence never reached the Top-5."""
    return tuple(
        result
        for result in run.results
        if result.answerable and result.hit_at_5 is False
    )


def false_refusals(
    run: EvaluationRun,
    threshold: float,
) -> tuple[EvaluationResult, ...]:
    """Return answerable cases the system would refuse at one threshold."""
    return tuple(
        result
        for result in run.results
        if result.answerable and not predict_answerable(result, threshold)
    )


def false_answers(
    run: EvaluationRun,
    threshold: float,
) -> tuple[EvaluationResult, ...]:
    """Return unanswerable cases the system would answer at one threshold."""
    return tuple(
        result
        for result in run.results
        if not result.answerable and predict_answerable(result, threshold)
    )


def _score_range_overlap_payload(run: EvaluationRun) -> dict[str, object] | None:
    overlap = score_range_overlap(run)
    if overlap is None:
        return None
    lower, upper, answerable_count, unanswerable_count = overlap
    return {
        "lower": _round(lower),
        "upper": _round(upper),
        "answerable_in_range": answerable_count,
        "unanswerable_in_range": unanswerable_count,
        "interpretation": (
            "两组最高相似度的取值范围发生重叠，因此本数据集上不存在能够实现"
            "零错误的单一全局阈值。in_range 只是描述性统计，"
            "不是最小错误数，也不代表这些题目都会被某一个阈值判断错误。"
        ),
    }


def build_summary(run: EvaluationRun) -> dict[str, object]:
    """Build the stable, JSON-serializable summary payload."""
    configuration = run.configuration
    comparison = configuration.comparison_threshold
    recommended = run.recommended_threshold
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": run.embedding_model,
        "chunk_configuration": asdict(run.chunk_configuration),
        "dataset_counts": {
            "total": len(run.cases),
            "answerable": sum(1 for case in run.cases if case.answerable),
            "unanswerable": sum(1 for case in run.cases if not case.answerable),
        },
        "split_counts": count_by(run.cases, "split"),
        "category_counts": count_by(run.cases, "category"),
        "difficulty_counts": count_by(run.cases, "difficulty"),
        "retrieval_metrics": {
            "overall": _retrieval_metrics_payload(run.retrieval_metrics),
            "by_category": {
                key: _retrieval_metrics_payload(value)
                for key, value in run.retrieval_metrics_by_category.items()
            },
            "by_difficulty": {
                key: _retrieval_metrics_payload(value)
                for key, value in run.retrieval_metrics_by_difficulty.items()
            },
            "by_split": {
                key: _retrieval_metrics_payload(value)
                for key, value in run.retrieval_metrics_by_split.items()
            },
        },
        "calibration_threshold_table": _threshold_table_payload(
            run.calibration_candidates
        ),
        "comparison_threshold": _round(comparison),
        "recommended_threshold": _round(recommended),
        "calibration_metrics_comparison": _decision_metrics_payload(
            run.calibration_metrics_comparison
        ),
        "calibration_metrics_recommended": _decision_metrics_payload(
            run.calibration_metrics_recommended
        ),
        "test_metrics_comparison": _decision_metrics_payload(
            run.test_metrics_comparison
        ),
        "test_metrics_recommended": _decision_metrics_payload(
            run.test_metrics_recommended
        ),
        "score_distribution_summary": {
            key: _score_distribution_payload(value)
            for key, value in sorted(run.score_distribution.items())
        },
        "score_range_overlap": _score_range_overlap_payload(run),
        "split_roles": dict(SPLIT_ROLES),
        "reproducibility": dict(REPRODUCIBILITY_NOTES),
        "failure_case_ids": {
            "retrieval_miss_at_5": [
                result.case_id for result in retrieval_failures(run)
            ],
            "false_refusal_comparison": [
                result.case_id for result in false_refusals(run, comparison)
            ],
            "false_answer_comparison": [
                result.case_id for result in false_answers(run, comparison)
            ],
            "false_refusal_recommended": [
                result.case_id for result in false_refusals(run, recommended)
            ],
            "false_answer_recommended": [
                result.case_id for result in false_answers(run, recommended)
            ],
        },
        "run_configuration": asdict(configuration),
    }


def write_summary_json(run: EvaluationRun, output_path: Path) -> Path:
    """Write summary.json as strict UTF-8 JSON with no NaN or Infinity."""
    payload = build_summary(run)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    output_path.write_text(text + "\n", encoding="utf-8")
    return output_path


def _csv_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _csv_number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{round(float(value), digits)}"


def build_case_rows(run: EvaluationRun) -> list[dict[str, str]]:
    """Build one stable CSV row per case in dataset order."""
    comparison = run.configuration.comparison_threshold
    recommended = run.recommended_threshold
    rows: list[dict[str, str]] = []
    for result in run.results:
        predicted_comparison = predict_answerable(result, comparison)
        predicted_recommended = predict_answerable(result, recommended)
        top_source = result.sources[0] if result.sources else None
        rows.append(
            {
                "case_id": result.case_id,
                "split": result.split,
                "category": result.category,
                "difficulty": result.difficulty,
                "question": result.question,
                "answerable": _csv_bool(result.answerable),
                "max_relevance_score": _csv_number(result.max_relevance_score),
                "first_relevant_rank": (
                    "" if result.first_relevant_rank is None
                    else str(result.first_relevant_rank)
                ),
                "hit_at_1": _csv_bool(result.hit_at_1),
                "hit_at_3": _csv_bool(result.hit_at_3),
                "hit_at_5": _csv_bool(result.hit_at_5),
                "recall_at_1": _csv_number(result.recall_at(1)),
                "recall_at_3": _csv_number(result.recall_at(3)),
                "recall_at_5": _csv_number(result.recall_at(5)),
                "reciprocal_rank": _csv_number(result.reciprocal_rank),
                "predicted_at_comparison_threshold": _csv_bool(
                    predicted_comparison
                ),
                "predicted_at_recommended_threshold": _csv_bool(
                    predicted_recommended
                ),
                "decision_correct_comparison": _csv_bool(
                    predicted_comparison == result.answerable
                ),
                "decision_correct_recommended": _csv_bool(
                    predicted_recommended == result.answerable
                ),
                "top_source_document": (
                    "" if top_source is None else top_source.document_id
                ),
                "top_source_page": (
                    ""
                    if top_source is None or top_source.page_number is None
                    else str(top_source.page_number)
                ),
                "top_source_score": (
                    "" if top_source is None else _csv_number(top_source.score)
                ),
            }
        )
    return rows


def write_cases_csv(run: EvaluationRun, output_path: Path) -> Path:
    """Write cases.csv as UTF-8 with a BOM so Excel opens Chinese correctly."""
    rows = build_case_rows(run)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CASES_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _format_optional(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _retrieval_table(rows: dict[str, RetrievalMetrics]) -> list[str]:
    lines = [
        "| 分组 | 题数 | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 |"
        " Recall@5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, metrics in rows.items():
        lines.append(
            f"| {key} | {metrics.case_count} |"
            f" {_format_optional(metrics.hit_at_1)} |"
            f" {_format_optional(metrics.hit_at_3)} |"
            f" {_format_optional(metrics.hit_at_5)} |"
            f" {_format_optional(metrics.recall_at_1)} |"
            f" {_format_optional(metrics.recall_at_3)} |"
            f" {_format_optional(metrics.recall_at_5)} |"
            f" {_format_optional(metrics.mean_reciprocal_rank)} |"
        )
    return lines


def _decision_table(rows: list[tuple[str, DecisionMetrics]]) -> list[str]:
    lines = [
        "| 场景 | 阈值 | TP | FN | TN | FP | 决策准确率 | 错误放行率 |"
        " 错误拒答率 | 回答率 | 拒答率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        " ---: | ---: |",
    ]
    for label, metrics in rows:
        lines.append(
            f"| {label} | {metrics.threshold:.2f} | {metrics.true_positive} |"
            f" {metrics.false_negative} | {metrics.true_negative} |"
            f" {metrics.false_positive} |"
            f" {_format_optional(metrics.decision_accuracy)} |"
            f" {_format_optional(metrics.false_answer_rate)} |"
            f" {_format_optional(metrics.false_refusal_rate)} |"
            f" {_format_optional(metrics.answer_rate)} |"
            f" {_format_optional(metrics.refusal_rate)} |"
        )
    return lines


def _failure_section(
    title: str,
    failure_type: str,
    results: tuple[EvaluationResult, ...],
    run: EvaluationRun,
) -> list[str]:
    lines = [f"### {title}", ""]
    if not results:
        lines.extend(["本类别没有失败案例。", ""])
        return lines

    lines.append(f"共 {len(results)} 例，下面展示最多 {MAX_FAILURE_EXAMPLES} 例。")
    lines.append("")
    for result in results[:MAX_FAILURE_EXAMPLES]:
        case = next(item for item in run.cases if item.id == result.case_id)
        evidence = (
            "、".join(
                f"{group.document_id}:{'/'.join(group.required_terms)}"
                for group in case.expected_evidence
            )
            or "无（资料外问题）"
        )
        top_sources = " / ".join(
            f"#{source.rank} {source.document_id}"
            f"(p={source.page_number}, {source.score:.4f})"
            for source in result.sources[:3]
        ) or "无检索结果"
        lines.extend(
            [
                f"- **{result.case_id}**（{result.category} / {result.difficulty}）",
                f"  - 问题：{result.question}",
                f"  - 失败类型：{failure_type}",
                f"  - 期望证据：{evidence}",
                f"  - 最高分：{_format_optional(result.max_relevance_score)}",
                f"  - Top-3 来源：{top_sources}",
            ]
        )
    lines.append("")
    return lines


def score_range_overlap(
    run: EvaluationRun,
) -> tuple[float, float, int, int] | None:
    """Return the overlapping score range and how many cases fall inside it.

    The range is ``[min answerable score, max unanswerable score]``. It exists
    only when ``max unanswerable >= min answerable``, which proves that no
    single global threshold can score zero errors on this dataset.

    The two counts are **descriptive statistics about the score distribution**.
    They are not a minimum error count, and they do not say that every case in
    the     range is misclassified by some threshold. With answerable scores
    ``0.50, 0.55`` and an unanswerable score ``0.60`` all three cases sit in the
    range, yet a threshold of ``0.50`` makes only one mistake. Deriving the real
    minimum would need an optimal-split search, which this report deliberately
    does not attempt.
    """
    answerable_scores = [
        result.max_relevance_score
        for result in run.results
        if result.answerable and result.max_relevance_score is not None
    ]
    unanswerable_scores = [
        result.max_relevance_score
        for result in run.results
        if not result.answerable and result.max_relevance_score is not None
    ]
    if not answerable_scores or not unanswerable_scores:
        return None

    lower = min(answerable_scores)
    upper = max(unanswerable_scores)
    if upper < lower:
        return None
    return (
        lower,
        upper,
        sum(1 for score in answerable_scores if lower <= score <= upper),
        sum(1 for score in unanswerable_scores if lower <= score <= upper),
    )


def _overlap_note(run: EvaluationRun) -> list[str]:
    """Describe the separability of the two score distributions honestly."""
    overlap = score_range_overlap(run)
    if overlap is None:
        return [
            "资料内问题的最低分高于资料外问题的最高分，两组取值范围完全分离，"
            "在本基准上存在能同时避免误答与误拒的阈值。",
        ]
    lower, upper, answerable_count, unanswerable_count = overlap
    return [
        f"两组最高相似度的取值范围在 [{lower:.4f}, {upper:.4f}] 内发生重叠。"
        f"其中资料内 {answerable_count} 题、资料外 {unanswerable_count} 题的分数"
        "落在该范围内。",
        "",
        "这证明本数据集上**不存在能够实现零错误的单一全局阈值**，"
        "但区间内题数只是分布描述，**并不等于最少必然出错的题数**，"
        "也不代表这些题目都会被某一个阈值判断错误。",
        "",
        "举例：资料内分数 0.50、0.55，资料外分数 0.60，三题都落在区间内，"
        "但阈值取 0.50 时只错 1 题。本报告不计算理论最小错误数。",
        "",
        "继续压低两类错误需要改进检索本身或引入二阶段判断，"
        "而不是只靠继续微调阈值。",
    ]


def build_markdown_report(run: EvaluationRun) -> str:
    """Build the human-readable Markdown report."""
    configuration = run.configuration
    comparison = configuration.comparison_threshold
    recommended = run.recommended_threshold
    answerable_count = sum(1 for case in run.cases if case.answerable)
    distribution = run.score_distribution

    lines: list[str] = [
        "# 离线 RAG 检索与拒答评估报告",
        "",
        f"Schema 版本：`{SCHEMA_VERSION}`",
        "",
        "本报告只评估检索质量和回答/拒答决策，不评估生成答案的质量，"
        "也不评估答案忠实度。",
        "",
        "## 运行配置",
        "",
        "```text",
        f"embedding_model     = {run.embedding_model}",
        f"chunk_size          = {run.chunk_configuration.chunk_size}",
        f"chunk_overlap       = {run.chunk_configuration.chunk_overlap}",
        f"requested_top_k     = {configuration.requested_top_k}",
        f"effective_top_k     = {configuration.effective_top_k}",
        f"manifest            = {configuration.manifest_path}",
        f"dataset             = {configuration.dataset_path}",
        f"threshold_range     = [{configuration.threshold_start}, "
        f"{configuration.threshold_end}] step {configuration.threshold_step}",
        f"comparison_threshold= {comparison}",
        f"false_answer_weight = {configuration.false_answer_weight}",
        f"false_refusal_weight= {configuration.false_refusal_weight}",
        "```",
        "",
        f"`comparison_threshold = {comparison}` 是**仓库默认对比阈值**，"
        "只用于把推荐阈值放在一个参照系里看。评估脚本不读取 "
        "`RAG_MIN_RELEVANCE_SCORE`，因此这个数值**不一定等于任何部署环境"
        "当前实际使用的阈值**；要对比真实部署值，请显式传入 "
        "`--comparison-threshold`。",
        "",
        "## 语料说明",
        "",
        f"评估语料共 {run.chunk_configuration.document_count} 份受控文档，"
        f"使用生产 DocumentLoader 与 chunk_document 切分为 "
        f"{run.chunk_configuration.chunk_count} 个 Chunk。",
        "语料是专为评估编写的原创内容，不是运行时用户上传资料，"
        "因此结论只适用于这份受控基准。",
        "",
        "## 数据集分布",
        "",
        f"- 总题数：{len(run.cases)}",
        f"- 可回答：{answerable_count}",
        f"- 不可回答：{len(run.cases) - answerable_count}",
        f"- split：{count_by(run.cases, 'split')}",
        f"- category：{count_by(run.cases, 'category')}",
        f"- difficulty：{count_by(run.cases, 'difficulty')}",
        "",
        "## 检索指标",
        "",
        "检索指标只在 `answerable=true` 的题目上计算，资料外问题不进入分母。",
        "",
    ]
    lines.extend(_retrieval_table({"全部": run.retrieval_metrics}))
    lines.extend(
        [
            "",
            "### 按 category 拆分",
            "",
        ]
    )
    lines.extend(_retrieval_table(run.retrieval_metrics_by_category))
    lines.extend(["", "### 按 difficulty 拆分", ""])
    lines.extend(_retrieval_table(run.retrieval_metrics_by_difficulty))
    lines.extend(["", "### 按 split 拆分", ""])
    lines.extend(_retrieval_table(run.retrieval_metrics_by_split))

    answerable_dist = distribution["answerable"]
    unanswerable_dist = distribution["unanswerable"]
    overlap_note = _overlap_note(run)
    lines.extend(
        [
            "",
            "## 相似度分布摘要",
            "",
            "| 组别 | 数量 | min | p25 | median | mean | p75 | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| 资料内 | {answerable_dist.count} |"
            f" {_format_optional(answerable_dist.minimum)} |"
            f" {_format_optional(answerable_dist.p25)} |"
            f" {_format_optional(answerable_dist.median)} |"
            f" {_format_optional(answerable_dist.mean)} |"
            f" {_format_optional(answerable_dist.p75)} |"
            f" {_format_optional(answerable_dist.maximum)} |",
            f"| 资料外 | {unanswerable_dist.count} |"
            f" {_format_optional(unanswerable_dist.minimum)} |"
            f" {_format_optional(unanswerable_dist.p25)} |"
            f" {_format_optional(unanswerable_dist.median)} |"
            f" {_format_optional(unanswerable_dist.mean)} |"
            f" {_format_optional(unanswerable_dist.p75)} |"
            f" {_format_optional(unanswerable_dist.maximum)} |",
            "",
            "### 分数范围重叠",
            "",
        ]
    )
    lines.extend(overlap_note)
    lines.extend(
        [
            "",
            "## 阈值扫描（仅 calibration split）",
            "",
            "`weighted_cost = FP × false_answer_weight + FN × "
            "false_refusal_weight`。",
            "错误放行的权重更高是**本项目的业务选择**，不是通用行业标准："
            "资料外问题被放行更容易导致幻觉。",
            "",
        ]
    )

    lines.extend(
        [
            "| 阈值 | TP | FN | TN | FP | 决策准确率 | 错误放行率 | 错误拒答率"
            " | weighted_cost |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in run.calibration_candidates:
        metrics = candidate.metrics
        lines.append(
            f"| {candidate.threshold:.2f} | {metrics.true_positive} |"
            f" {metrics.false_negative} | {metrics.true_negative} |"
            f" {metrics.false_positive} |"
            f" {_format_optional(metrics.decision_accuracy)} |"
            f" {_format_optional(metrics.false_answer_rate)} |"
            f" {_format_optional(metrics.false_refusal_rate)} |"
            f" {candidate.weighted_cost:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 推荐阈值及选择规则",
            "",
            f"- 推荐阈值：**{recommended:.2f}**",
            f"- 仓库默认对比阈值：**{comparison:.2f}**"
            "（不一定等于实际部署值）",
            "- 选择只使用 calibration split，test split 没有参与本次阈值扫描。",
            "- 确定性排序规则：weighted_cost 最低 → false_answer 更少 →"
            " false_refusal 更少 → 决策准确率更高 → 阈值更高（保守）。",
            "- 推荐阈值**不会**自动写入生产配置，是否采纳需要单独决策。",
            "",
            "## 对比阈值与推荐阈值对比",
            "",
        ]
    )
    lines.extend(
        _decision_table(
            [
                ("calibration @ 对比阈值", run.calibration_metrics_comparison),
                ("calibration @ 推荐阈值", run.calibration_metrics_recommended),
                ("test @ 对比阈值", run.test_metrics_comparison),
                ("test @ 推荐阈值", run.test_metrics_recommended),
            ]
        )
    )

    lines.extend(
        [
            "",
            "## 初始留出测试结果",
            "",
            "推荐阈值在 calibration 上固定之后，只在 test split 上运行一次，"
            "本次结果不再用于回头调整阈值。",
            "",
            f"- test 决策准确率（推荐阈值）："
            f"{_format_optional(run.test_metrics_recommended.decision_accuracy)}",
            f"- test 错误放行率（推荐阈值）："
            f"{_format_optional(run.test_metrics_recommended.false_answer_rate)}",
            f"- test 错误拒答率（推荐阈值）："
            f"{_format_optional(run.test_metrics_recommended.false_refusal_rate)}",
            "",
            "## 数据集 split 的角色与生命周期",
            "",
            "| split | 角色 |",
            "| --- | --- |",
            *(f"| `{name}` | {description} |" for name, description in SPLIT_ROLES),
            "",
            "本次推荐阈值只由 calibration split 选择，随后在 test split 上评估一次。",
            "由于 test 结果现已公开（写入仓库、出现在 PR 描述、并被用于判断推荐阈值"
            "是否合理），它后续适合作为 **regression benchmark** 使用，"
            "**不应再被视为未来完全未见的最终 holdout**。",
            "",
            "## 可重复性的适用范围",
            "",
            "| 场景 | 结论 |",
            "| --- | --- |",
            *(
                f"| `{name}` | {description} |"
                for name, description in REPRODUCIBILITY_NOTES
            ),
            "",
            "换句话说：逐字节稳定这条结论来自 Fake Embedding 路径的自动化测试，"
            "**不能外推**到不同机器、不同依赖版本上的真实 Embedding 基线。",
            "",
            "## 主要失败案例",
            "",
        ]
    )
    lines.extend(
        _failure_section(
            "检索失败（answerable 且 Hit@5 = false）",
            "检索失败",
            retrieval_failures(run),
            run,
        )
    )
    lines.extend(
        _failure_section(
            f"错误拒答（对比阈值 {comparison:.2f}）",
            "错误拒答",
            false_refusals(run, comparison),
            run,
        )
    )
    lines.extend(
        _failure_section(
            f"错误放行（对比阈值 {comparison:.2f}）",
            "错误放行",
            false_answers(run, comparison),
            run,
        )
    )

    lines.extend(
        [
            "## 已知限制",
            "",
            "- 评估语料是受控基准，规模有限，不代表所有真实用户文档。",
            "- 数据集第一版题量有限，指标存在抽样波动。",
            "- 本阶段不评估生成答案质量，也不评估答案忠实度。",
            "- Hit@5 高不代表最终答案一定正确。",
            "- 错误放行与错误拒答的权重是业务选择，换一个业务场景需要重新设定。",
            "- 索引规模较小时 Top-5 覆盖了较大比例的 Chunk，Hit@5 会偏乐观。",
            "- 分数范围重叠只说明不存在零错误阈值，区间内题数不是最小错误数。",
            "- 逐字节可重复性只在 Fake Embedding 路径上被验证过，"
            "真实模型跨环境可能有末位差异。",
            "- test 结果已公开，后续只应作为回归集，不能当作未见的最终 holdout。",
            "",
            "## 后续建议",
            "",
            "- 扩充语料和题量，特别是 `out_of_scope_near` 类型。",
            "- 更换 Chunk 参数、Embedding 模型或引入 Reranker 后重新跑一次基线，"
            "对比是否退化。",
            "- 若两组分数取值范围重叠严重，优先改进检索，而不是继续微调阈值。",
            "- 正式比较多个检索方案时，先准备一份从未参与开发决策的独立 "
            "final holdout。",
            "- 是否把推荐阈值写入生产配置，应作为独立决策单独提交。",
            "",
        ]
    )
    # Section builders append a trailing separator line each; drop the extras so
    # the file ends with exactly one newline instead of a blank line at EOF.
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def write_markdown_report(run: EvaluationRun, output_path: Path) -> Path:
    """Write report.md as UTF-8 Markdown."""
    output_path.write_text(build_markdown_report(run), encoding="utf-8")
    return output_path


def write_reports(run: EvaluationRun, output_dir: Path) -> tuple[Path, ...]:
    """Write summary.json, cases.csv, and report.md into one directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        write_summary_json(run, output_dir / SUMMARY_FILENAME),
        write_cases_csv(run, output_dir / CASES_FILENAME),
        write_markdown_report(run, output_dir / REPORT_FILENAME),
    )
