"""Write evaluation reports as JSON, CSV, and Markdown."""

import csv
import json
from pathlib import Path
from typing import Any

from src.rag_service import has_sufficient_context

from .models import EvaluationResult

CSV_COLUMNS = [
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
    "predicted_at_current_threshold",
    "predicted_at_recommended_threshold",
    "decision_correct_current",
    "decision_correct_recommended",
    "top_source_document",
    "top_source_page",
    "top_source_score",
]


def _json_safe(value: Any) -> Any:
    """Replace non-finite numbers with None for valid standard JSON."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_summary(path: Path, data: dict[str, Any]) -> None:
    """Write the summary JSON with stable ordering and no NaN values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = _json_safe(data)
    path.write_text(
        json.dumps(safe_data, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _csv_value(result: EvaluationResult, column: str) -> object:
    if column == "case_id":
        return result.case_id
    if column == "split":
        return result.split
    if column == "category":
        return result.category
    if column == "difficulty":
        return result.difficulty
    if column == "question":
        return result.question
    if column == "answerable":
        return result.answerable
    if column == "max_relevance_score":
        return result.max_relevance_score
    if column == "first_relevant_rank":
        return result.first_relevant_rank
    if column == "hit_at_1":
        return result.hit_at_1
    if column == "hit_at_3":
        return result.hit_at_3
    if column == "hit_at_5":
        return result.hit_at_5
    if column == "recall_at_1":
        return _recall(result, 1)
    if column == "recall_at_3":
        return _recall(result, 3)
    if column == "recall_at_5":
        return _recall(result, 5)
    if column == "reciprocal_rank":
        return result.reciprocal_rank
    if column == "predicted_at_current_threshold":
        return has_sufficient_context(
            result.max_relevance_score,
            _thresholds[0],
        )
    if column == "predicted_at_recommended_threshold":
        return has_sufficient_context(
            result.max_relevance_score,
            _thresholds[1],
        )
    if column == "decision_correct_current":
        return (
            has_sufficient_context(result.max_relevance_score, _thresholds[0])
            == result.answerable
        )
    if column == "decision_correct_recommended":
        return (
            has_sufficient_context(result.max_relevance_score, _thresholds[1])
            == result.answerable
        )
    if column == "top_source_document":
        return result.sources[0].get("document_id") if result.sources else None
    if column == "top_source_page":
        return result.sources[0].get("page_number") if result.sources else None
    if column == "top_source_score":
        return result.sources[0].get("score") if result.sources else None
    raise ValueError(f"未知的 CSV 列：{column}")


def _recall(result: EvaluationResult, k: int) -> float | None:
    if result.total_evidence_count <= 0:
        return None
    matched = {
        1: result.matched_evidence_count_at_1,
        3: result.matched_evidence_count_at_3,
        5: result.matched_evidence_count_at_5,
    }[k]
    return round(matched / result.total_evidence_count, 6)


_thresholds: list[float] = []


def write_cases_csv(
    path: Path,
    results: list[EvaluationResult],
    current_threshold: float,
    recommended_threshold: float,
) -> None:
    """Write one row per case with UTF-8 BOM for Excel compatibility."""
    global _thresholds
    path.parent.mkdir(parents=True, exist_ok=True)
    _thresholds = [current_threshold, recommended_threshold]
    try:
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for result in results:
                writer.writerow(
                    [_csv_value(result, column) for column in CSV_COLUMNS]
                )
    finally:
        _thresholds = []


def _format_metric(name: str, value: Any) -> str:
    if value is None:
        return "N/A（样本不足）"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table_row(row: dict[str, Any]) -> list[str]:
    return [
        f"{row['threshold']:.2f}",
        str(row["tp"]),
        str(row["fp"]),
        str(row["tn"]),
        str(row["fn"]),
        _format_metric("accuracy", row["decision_accuracy"]),
        _format_metric("false_answer_rate", row["false_answer_rate"]),
        _format_metric("false_refusal_rate", row["false_refusal_rate"]),
        f"{row['weighted_cost']:.1f}",
    ]


def write_report_md(
    path: Path,
    data: dict[str, Any],
    results: list[EvaluationResult],
) -> None:
    """Write the human-readable Markdown evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config = data["run_configuration"]
    lines: list[str] = []
    lines.append("# RAG 离线评估报告")
    lines.append("")

    lines.append("## 运行配置")
    lines.append("")
    lines.append("| 配置项 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| Embedding 模型 | {data['embedding_model']} |")
    lines.append(
        f"| Chunk 配置 | size={data['chunk_configuration']['chunk_size']}, "
        f"overlap={data['chunk_configuration']['chunk_overlap']} |"
    )
    lines.append(f"| Top-K | {config['top_k']} |")
    lines.append(f"| 数据集 | {config['dataset']} |")
    lines.append(f"| 语料清单 | {config['manifest']} |")
    lines.append(
        f"| 阈值扫描 | {config['threshold_start']:.2f} - "
        f"{config['threshold_end']:.2f}，步长 {config['threshold_step']:.2f} |"
    )
    lines.append(
        f"| 错误放行权重 | {config['false_answer_weight']:.1f} |"
    )
    lines.append(
        f"| 错误拒答权重 | {config['false_refusal_weight']:.1f} |"
    )
    lines.append("")
    lines.append("> 权重是本项目的业务选择，不是通用行业标准。")
    lines.append("")

    lines.append("## 语料与数据集")
    lines.append("")
    lines.append(
        "评估语料为 3 篇受控 Markdown 文档（backend-architecture / "
        "rag-fundamentals / database-basics），仅用于离线评估，"
        "不来自运行时上传资料。"
    )
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 总题数 | {data['dataset_counts']['total']} |")
    lines.append(
        f"| 可回答 / 不可回答 | {data['dataset_counts']['answerable']} / "
        f"{data['dataset_counts']['unanswerable']} |"
    )
    lines.append(
        f"| Calibration / Test | {data['split_counts']['calibration']} / "
        f"{data['split_counts']['test']} |"
    )
    for category, count in data["category_counts"].items():
        lines.append(f"| category={category} | {count} |")
    for difficulty, count in data["difficulty_counts"].items():
        lines.append(f"| difficulty={difficulty} | {count} |")
    lines.append("")

    retrieval = data["retrieval_metrics"]["overall"]
    lines.append("## 检索指标（仅可回答问题）")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    for key, label in [
        ("hit_at_1", "Hit@1"),
        ("hit_at_3", "Hit@3"),
        ("hit_at_5", "Hit@5"),
        ("recall_at_1", "Recall@1"),
        ("recall_at_3", "Recall@3"),
        ("recall_at_5", "Recall@5"),
        ("mrr", "MRR"),
    ]:
        lines.append(f"| {label} | {_format_metric(key, retrieval[key])} |")
    lines.append("")

    lines.append("### 按 category")
    lines.append("")
    lines.append("| category | Hit@5 | Recall@5 | MRR |")
    lines.append("| --- | --- | --- | --- |")
    for category, metrics in data["retrieval_metrics"]["by_category"].items():
        lines.append(
            f"| {category} | {_format_metric('hit', metrics['hit_at_5'])} | "
            f"{_format_metric('recall', metrics['recall_at_5'])} | "
            f"{_format_metric('mrr', metrics['mrr'])} |"
        )
    lines.append("")

    lines.append("### 按 difficulty")
    lines.append("")
    lines.append("| difficulty | Hit@5 | Recall@5 | MRR |")
    lines.append("| --- | --- | --- | --- |")
    for difficulty, metrics in data["retrieval_metrics"][
        "by_difficulty"
    ].items():
        lines.append(
            f"| {difficulty} | {_format_metric('hit', metrics['hit_at_5'])} | "
            f"{_format_metric('recall', metrics['recall_at_5'])} | "
            f"{_format_metric('mrr', metrics['mrr'])} |"
        )
    lines.append("")

    distribution = data["score_distribution_summary"]
    lines.append("## 相似度分布摘要")
    lines.append("")
    lines.append("| 分组 | count | min | max | mean | median | p25 | p75 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for label, summary in [
        ("可回答", distribution["answerable"]),
        ("不可回答", distribution["unanswerable"]),
    ]:
        if summary.get("count", 0) == 0:
            lines.append(f"| {label} | 0 | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {label} | {summary['count']} | {summary['min']:.4f} | "
            f"{summary['max']:.4f} | {summary['mean']:.4f} | "
            f"{summary['median']:.4f} | {summary['p25']:.4f} | "
            f"{summary['p75']:.4f} |"
        )
    lines.append("")
    lines.append(
        "两组分布明显分离时固定阈值较有效；大量重叠时单阈值能力有限，"
        "应考虑检索改进或二阶段判断。"
    )
    lines.append("")

    lines.append("## 阈值扫描与推荐")
    lines.append("")
    lines.append(
        "推荐阈值只从 calibration 拆分中选出，选择规则按顺序为："
        "加权成本最低 → 错误放行更少 → 错误拒答更少 → 决策准确率更高 → "
        "阈值更高（保守优先）。test 拆分不参与阈值选择。"
    )
    lines.append("")
    lines.append("| 阈值 | TP | FP | TN | FN | 准确率 | 错误放行率 | 错误拒答率 | 加权成本 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in data["calibration_threshold_table"]:
        lines.append("| " + " | ".join(_table_row(row)) + " |")
    lines.append("")

    lines.append("## 当前阈值与推荐阈值对比")
    lines.append("")
    lines.append("| 拆分 | 阈值 | 决策准确率 | 错误放行率 | 错误拒答率 | 回答率 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    comparisons = [
        (
            "Calibration",
            data["current_threshold"],
            data["calibration_metrics_current"],
        ),
        (
            "Calibration",
            data["recommended_threshold"],
            data["calibration_metrics_recommended"],
        ),
        (
            "Test",
            data["current_threshold"],
            data["test_metrics_current"],
        ),
        (
            "Test",
            data["recommended_threshold"],
            data["test_metrics_recommended"],
        ),
    ]
    for label, threshold, block in comparisons:
        decision = block["decision"]
        lines.append(
            f"| {label} | {threshold:.2f} | "
            f"{_format_metric('acc', decision['decision_accuracy'])} | "
            f"{_format_metric('fa', decision['false_answer_rate'])} | "
            f"{_format_metric('fr', decision['false_refusal_rate'])} | "
            f"{_format_metric('ar', decision['answer_rate'])} |"
        )
    lines.append("")
    lines.append(
        f"当前生产阈值：{data['current_threshold']:.2f}；"
        f"校准集推荐阈值：{data['recommended_threshold']:.2f}。"
        "推荐阈值仅作为建议，不会自动写入生产配置。"
    )
    lines.append("")

    lines.append("## 测试集最终结果")
    lines.append("")
    test_recommended = data["test_metrics_recommended"]
    lines.append(
        "test 拆分只使用固定的推荐阈值评估一次，不再用于调整阈值。"
    )
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append(
        "| 决策准确率 | "
        f"{_format_metric('acc', test_recommended['decision']['decision_accuracy'])} |"
    )
    lines.append(
        "| 错误放行率 | "
        f"{_format_metric('fa', test_recommended['decision']['false_answer_rate'])} |"
    )
    lines.append(
        "| 错误拒答率 | "
        f"{_format_metric('fr', test_recommended['decision']['false_refusal_rate'])} |"
    )
    lines.append(
        "| Hit@5 | "
        f"{_format_metric('hit', test_recommended['retrieval']['hit_at_5'])} |"
    )
    lines.append(
        "| MRR | "
        f"{_format_metric('mrr', test_recommended['retrieval']['mrr'])} |"
    )
    lines.append("")

    failures = data["failure_case_ids"]
    lines.append("## 主要失败案例")
    lines.append("")
    if failures["retrieval_failures"]:
        lines.append(
            "检索失败（answerable 且 Hit@5 未命中）："
            + "、".join(failures["retrieval_failures"])
        )
        lines.append("")
    if failures["false_refusals"]:
        lines.append(
            "错误拒答（answerable 但推荐阈值下判定不可回答）："
            + "、".join(failures["false_refusals"])
        )
        lines.append("")
    if failures["false_answers"]:
        lines.append(
            "错误放行（unanswerable 但当前阈值下判定可回答）："
            + "、".join(failures["false_answers"])
        )
        lines.append("")

    lines.append("### 失败案例明细")
    lines.append("")
    for result in results:
        predicted_recommended = has_sufficient_context(
            result.max_relevance_score,
            data["recommended_threshold"],
        )
        failure_kinds: list[str] = []
        if result.answerable and result.hit_at_5 is not True:
            failure_kinds.append("检索失败")
        if result.answerable and not predicted_recommended:
            failure_kinds.append("错误拒答")
        if not result.answerable and predicted_recommended:
            failure_kinds.append("错误放行")
        if not failure_kinds:
            continue
        lines.append(f"### {result.case_id}（{'、'.join(failure_kinds)}）")
        lines.append("")
        lines.append(f"- 问题：{result.question}")
        lines.append(f"- category：{result.category}；difficulty：{result.difficulty}")
        lines.append(f"- 最高相似度：{result.max_relevance_score}")
        lines.append(
            f"- 首个证据命中排名：{result.first_relevant_rank}"
        )
        if result.sources:
            top_sources = result.sources[:3]
            lines.append("- Top-3 来源摘要：")
            for source in top_sources:
                document_id = source.get("document_id")
                page = source.get("page_number")
                score = source.get("score")
                lines.append(
                    f"  - [{source['rank']}] {document_id} "
                    f"(page={page}, score={score})："
                    f"{str(source['text'])[:60]}"
                )
        lines.append("")

    lines.append("## 已知限制")
    lines.append("")
    lines.append(
        "- 评估语料是受控基准，不等同于所有真实用户文档。\n"
        "- 本报告不评估生成答案质量或 LLM 忠实度。\n"
        "- Hit@5 高不代表生成答案一定正确。\n"
        "- 阈值建议基于本语料与当前 Embedding 模型，更换模型后需要重新校准。\n"
        "- 权重（错误放行 3、错误拒答 1）是本项目的业务选择。\n"
        "- 推荐阈值不会自动写入生产配置。"
    )
    lines.append("")

    lines.append("## 后续建议")
    lines.append("")
    lines.append(
        "- 扩充数据集规模并补充 PDF 页码证据。\n"
        "- 更换 Chunk、Embedding 或引入 Reranker 后重跑本评估。\n"
        "- 观察可回答与不可回答分数分布的重叠程度，评估是否需要二阶段判断。\n"
        "- 单独决策是否修改生产阈值，并记录决策依据。"
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    output_dir: Path,
    data: dict[str, Any],
    results: list[EvaluationResult],
    current_threshold: float,
    recommended_threshold: float,
) -> None:
    """Write summary.json, cases.csv, and report.md into output_dir."""
    output_dir = Path(output_dir)
    write_summary(output_dir / "summary.json", data)
    write_cases_csv(
        output_dir / "cases.csv",
        results,
        current_threshold,
        recommended_threshold,
    )
    write_report_md(output_dir / "report.md", data, results)
