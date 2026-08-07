"""Render answer-quality evaluation runs as deterministic reports.

Every artifact is reproducible: identical inputs produce byte-identical
files.  Reports never contain timestamps, absolute paths, API keys,
authorization headers, provider errors, or full prompts.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.evaluation.answer_metrics import (
    AggregateAnswerMetrics,
)
from src.evaluation.answer_models import (
    AnswerEvaluationRun,
    CaseAnswerMetrics,
    FactScoring,
    LatencySummary,
)

SCHEMA_VERSION = "1.0.0"
SUMMARY_FILENAME = "summary.json"
CASES_FILENAME = "cases.jsonl"
REPORT_FILENAME = "report.md"
MAX_FAILURE_EXAMPLES = 5

LIMITATIONS: tuple[str, ...] = (
    "本框架是基于人工标注事实和引用的确定性检查，不是完整语义正确性证明。",
    "它不能发现所有幻觉，也不等同于人工评审或 LLM Judge。",
    "它只检查已标注的必要事实、引用编号与已知矛盾短语。",
    "reference_answer 只用于人工审计，不参与任何语义相似度评分；不使用 BLEU、ROUGE 或字符串相似度冒充事实正确性。",
    "来源支持（grounding）只针对已标注的 required_facts 判断；V1 使用答案级引用集合，不做句子级 claim-to-citation 对齐。",
    "结果代表这份受控标注集的 annotated 表现，不声称代表生产环境的真实用户分布。",
)

METRIC_DEFINITIONS: tuple[tuple[str, str], ...] = (
    (
        "annotated fact coverage",
        "每题被覆盖的 required_facts 占比（accepted_phrase 经同一规范化后出现在答案文本中），仅对 answerable 题目计算；错误拒答记为 0。",
    ),
    (
        "supported fact citation coverage",
        "每题被覆盖且至少有一个合法引用来源满足其 supporting evidence 的 required_facts 占比。",
    ),
    (
        "citation validity",
        "严格合法引用出现次数 / 全部严格引用出现次数（occurrence 加权，非每题比例平均）；无引用时为 null。",
    ),
    (
        "known contradiction detection",
        "任一 forbidden_phrase 出现在规范化答案中即命中；记录命中短语，不输出整段答案。",
    ),
    (
        "strict annotated pass rate",
        "满足 strict_pass 全部条件的题目占比；是标注级指标，不是绝对正确率。",
    ),
)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _round_latency(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _metrics_payload(metrics: AggregateAnswerMetrics) -> dict[str, Any]:
    return {
        "case_count": metrics.case_count,
        "answerable_count": metrics.answerable_count,
        "unanswerable_count": metrics.unanswerable_count,
        "answered_count": metrics.answered_count,
        "refused_count": metrics.refused_count,
        "true_answer_count": metrics.true_answer_count,
        "false_refusal_count": metrics.false_refusal_count,
        "true_refusal_count": metrics.true_refusal_count,
        "false_answer_count": metrics.false_answer_count,
        "decision_accuracy": _round(metrics.decision_accuracy),
        "false_answer_rate": _round(metrics.false_answer_rate),
        "false_refusal_rate": _round(metrics.false_refusal_rate),
        "average_fact_coverage": _round(metrics.average_fact_coverage),
        "average_grounded_fact_coverage": _round(
            metrics.average_grounded_fact_coverage
        ),
        "complete_fact_coverage_rate": _round(
            metrics.complete_fact_coverage_rate
        ),
        "complete_grounded_fact_coverage_rate": _round(
            metrics.complete_grounded_fact_coverage_rate
        ),
        "contradiction_free_rate": _round(metrics.contradiction_free_rate),
        "citation_occurrence_count": metrics.citation_occurrence_count,
        "valid_citation_occurrence_count": metrics.valid_citation_occurrence_count,
        "invalid_citation_occurrence_count": metrics.invalid_citation_occurrence_count,
        "malformed_citation_count": metrics.malformed_citation_count,
        "citation_validity_rate": _round(metrics.citation_validity_rate),
        "invalid_citation_case_rate": _round(metrics.invalid_citation_case_rate),
        "malformed_citation_case_rate": _round(
            metrics.malformed_citation_case_rate
        ),
        "strict_pass_count": metrics.strict_pass_count,
        "strict_pass_rate": _round(metrics.strict_pass_rate),
    }


def _latency_payload(summary: LatencySummary) -> dict[str, Any]:
    return {
        "count": summary.count,
        "mean": _round_latency(summary.mean),
        "median": _round_latency(summary.median),
        "p95": _round_latency(summary.p95),
        "minimum": _round_latency(summary.minimum),
        "maximum": _round_latency(summary.maximum),
    }


def _fact_payload(fact: FactScoring) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id,
        "covered": fact.covered,
        "matched_phrase": fact.matched_phrase,
        "supporting_evidence_indexes": list(fact.supporting_evidence_indexes),
        "supporting_citation_numbers": list(fact.supporting_citation_numbers),
        "grounded_by_citation": fact.grounded_by_citation,
    }


def _case_payload(case: CaseAnswerMetrics) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "split": case.split,
        "category": case.category,
        "difficulty": case.difficulty,
        "answerable": case.answerable,
        "answer_status": case.answer_status,
        "decision_correct": case.decision_correct,
        "question": case.question,
        "reference_answer": case.reference_answer,
        "answer": case.answer,
        "required_fact_count": case.required_fact_count,
        "covered_fact_count": case.covered_fact_count,
        "grounded_fact_count": case.grounded_fact_count,
        "fact_coverage": _round(case.fact_coverage),
        "grounded_fact_coverage": _round(case.grounded_fact_coverage),
        "complete_fact_coverage": case.complete_fact_coverage,
        "complete_grounded_fact_coverage": case.complete_grounded_fact_coverage,
        "forbidden_phrase_hits": list(case.forbidden_phrase_hits),
        "contradiction_free": case.contradiction_free,
        "citation_occurrence_count": case.citation_occurrence_count,
        "unique_cited_source_count": case.unique_cited_source_count,
        "valid_citation_occurrence_count": case.valid_citation_occurrence_count,
        "invalid_citation_occurrence_count": case.invalid_citation_occurrence_count,
        "invalid_citation_numbers": list(case.invalid_citation_numbers),
        "malformed_citation_count": case.malformed_citation_count,
        "malformed_citation_fragments": list(case.malformed_citation_fragments),
        "citation_validity_rate": _round(case.citation_validity_rate),
        "facts": [_fact_payload(fact) for fact in case.facts],
        "max_relevance_score": _round(case.max_relevance_score),
        "relevance_threshold": _round(case.relevance_threshold),
        "retrieval_elapsed_ms": _round_latency(case.retrieval_elapsed_ms),
        "generation_elapsed_ms": _round_latency(case.generation_elapsed_ms),
        "total_elapsed_ms": _round_latency(case.total_elapsed_ms),
        "reranker_applied": case.reranker_applied,
        "reranker_fallback": case.reranker_fallback,
        "strict_pass": case.strict_pass,
    }


def _strict_failure_case_ids(
    case_metrics: tuple[CaseAnswerMetrics, ...],
) -> list[str]:
    return [case.case_id for case in case_metrics if not case.strict_pass]


def _case_ids_where(
    case_metrics: tuple[CaseAnswerMetrics, ...],
    predicate: Any,
) -> list[str]:
    return [case.case_id for case in case_metrics if predicate(case)]


def _summary_payload(run: AnswerEvaluationRun) -> dict[str, Any]:
    aggregate = run.aggregate_metrics
    cases = run.case_metrics
    failure_id_lists = {
        "strict_failure_case_ids": _strict_failure_case_ids(cases),
        "false_answer_case_ids": _case_ids_where(
            cases,
            lambda case: not case.answerable and case.answer_status == "answered",
        ),
        "false_refusal_case_ids": _case_ids_where(
            cases,
            lambda case: case.answerable and case.answer_status == "insufficient_context",
        ),
        "incomplete_fact_case_ids": _case_ids_where(
            cases,
            lambda case: case.answerable and not case.complete_fact_coverage,
        ),
        "ungrounded_fact_case_ids": _case_ids_where(
            cases,
            lambda case: case.answerable and not case.complete_grounded_fact_coverage,
        ),
        "invalid_citation_case_ids": _case_ids_where(
            cases,
            lambda case: case.invalid_citation_occurrence_count > 0,
        ),
        "malformed_citation_case_ids": _case_ids_where(
            cases,
            lambda case: case.malformed_citation_count > 0,
        ),
        "contradiction_case_ids": _case_ids_where(
            cases,
            lambda case: not case.contradiction_free,
        ),
    }
    return {
        "schema_version": run.schema_version,
        "evaluation_type": "answer_quality",
        "run_configuration": asdict(run.configuration),
        "dataset_counts": {
            "total": aggregate.case_count,
            "answerable": aggregate.answerable_count,
            "unanswerable": aggregate.unanswerable_count,
        },
        "aggregate_metrics": _metrics_payload(aggregate),
        "metrics_by_split": {
            key: _metrics_payload(value)
            for key, value in sorted(run.metrics_by_split.items())
        },
        "metrics_by_category": {
            key: _metrics_payload(value)
            for key, value in sorted(run.metrics_by_category.items())
        },
        "metrics_by_difficulty": {
            key: _metrics_payload(value)
            for key, value in sorted(run.metrics_by_difficulty.items())
        },
        "latency_summary": {
            "retrieval_elapsed_ms": _latency_payload(run.latency_retrieval),
            "generation_elapsed_ms": _latency_payload(run.latency_generation),
            "total_elapsed_ms": _latency_payload(run.latency_total),
        },
        **failure_id_lists,
        "limitations": list(LIMITATIONS),
    }


def _write_summary_json(run: AnswerEvaluationRun, output_dir: Path) -> None:
    payload = _summary_payload(run)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    (output_dir / SUMMARY_FILENAME).write_text(text + "\n", encoding="utf-8")


def _write_cases_jsonl(run: AnswerEvaluationRun, output_dir: Path) -> None:
    lines = [
        json.dumps(
            _case_payload(case),
            ensure_ascii=False,
            allow_nan=False,
        )
        for case in run.case_metrics
    ]
    (output_dir / CASES_FILENAME).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _format_metric_table(metrics: AggregateAnswerMetrics) -> str:
    def fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    rows = [
        ("题目数", str(metrics.case_count)),
        ("可回答 / 不可回答", f"{metrics.answerable_count} / {metrics.unanswerable_count}"),
        ("回答 / 拒答", f"{metrics.answered_count} / {metrics.refused_count}"),
        ("正确回答 / 错误拒答 / 正确拒答 / 错误放行", (
            f"{metrics.true_answer_count} / {metrics.false_refusal_count} / "
            f"{metrics.true_refusal_count} / {metrics.false_answer_count}"
        )),
        ("决策准确率", fmt(metrics.decision_accuracy)),
        ("错误放行率", fmt(metrics.false_answer_rate)),
        ("错误拒答率", fmt(metrics.false_refusal_rate)),
        ("平均标注事实覆盖率", fmt(metrics.average_fact_coverage)),
        ("平均来源支持覆盖率", fmt(metrics.average_grounded_fact_coverage)),
        ("完全事实覆盖率", fmt(metrics.complete_fact_coverage_rate)),
        ("完全来源支持覆盖率", fmt(metrics.complete_grounded_fact_coverage_rate)),
        ("无已知矛盾率", fmt(metrics.contradiction_free_rate)),
        ("引用合法率", fmt(metrics.citation_validity_rate)),
        ("非法引用题目占比", fmt(metrics.invalid_citation_case_rate)),
        ("格式错误引用题目占比", fmt(metrics.malformed_citation_case_rate)),
        ("严格标注通过率", fmt(metrics.strict_pass_rate)),
    ]
    lines = ["| 指标 | 值 |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def _format_latency_table(name: str, summary: LatencySummary) -> str:
    def fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    return "\n".join(
        [
            f"### {name}",
            "",
            "| 统计量 | 值 |",
            "| --- | ---: |",
            f"| count | {summary.count} |",
            f"| mean | {fmt(summary.mean)} |",
            f"| median | {fmt(summary.median)} |",
            f"| p95 | {fmt(summary.p95)} |",
            f"| minimum | {fmt(summary.minimum)} |",
            f"| maximum | {fmt(summary.maximum)} |",
            "",
        ]
    )


def _write_report_md(run: AnswerEvaluationRun, output_dir: Path) -> None:
    config = run.configuration
    aggregate = run.aggregate_metrics
    lines: list[str] = [
        "# 回答质量评估报告",
        "",
        "## 运行配置",
        "",
        f"- run_name: `{config.run_name}`",
        f"- mode: `{config.mode}`",
        f"- model_label: `{config.model_label or 'N/A'}`",
        f"- prompt_version: `{config.prompt_version or 'N/A'}`",
        f"- embedding_model: `{config.embedding_model or 'N/A'}`",
        f"- top_k: `{config.top_k if config.top_k is not None else 'N/A'}`",
        f"- relevance_threshold: "
        f"`{config.relevance_threshold if config.relevance_threshold is not None else 'N/A'}`",
        f"- dataset: `{config.dataset_path}`",
        f"- annotations: `{config.annotations_path}`",
        f"- responses: `{config.responses_path or 'N/A'}`",
        f"- reranker_applied_any: `{config.reranker_applied_any}`",
        f"- reranker_fallback_any: `{config.reranker_fallback_any}`",
        "",
        "## 数据集统计",
        "",
        f"- 题目总数: {aggregate.case_count}",
        f"- 可回答: {aggregate.answerable_count}",
        f"- 不可回答: {aggregate.unanswerable_count}",
        "",
        "## 指标定义",
        "",
    ]
    for name, definition in METRIC_DEFINITIONS:
        lines.append(f"- **{name}**: {definition}")
    lines.append("")
    lines.append("## 整体指标")
    lines.append("")
    lines.append(_format_metric_table(aggregate))
    lines.append("")

    def group_section(title: str, grouped: dict[str, AggregateAnswerMetrics]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for key in sorted(grouped):
            lines.append(f"### {key}")
            lines.append("")
            lines.append(_format_metric_table(grouped[key]))
            lines.append("")

    group_section("按 split 分组", run.metrics_by_split)
    group_section("按 category 分组", run.metrics_by_category)
    group_section("按 difficulty 分组", run.metrics_by_difficulty)

    lines.append("## 引用质量")
    lines.append("")
    lines.append(_format_metric_table(aggregate))
    lines.append("")

    lines.append("## 必要事实覆盖")
    lines.append("")
    lines.append(_format_metric_table(aggregate))
    lines.append("")

    lines.append("## 严格通过率")
    lines.append("")
    lines.append(
        f"严格标注通过 {aggregate.strict_pass_count} / {aggregate.case_count} "
        f"（{fmt_rate(aggregate.strict_pass_rate)}）。"
    )
    lines.append(
        "这是标注级指标，不是绝对正确率；答案只有在覆盖全部标注事实、"
        "全部事实都有支持来源、无非法/格式错误引用、无已知矛盾短语时才通过。"
    )
    lines.append("")

    lines.append("## 延迟统计")
    lines.append("")
    lines.append(_format_latency_table("retrieval_elapsed_ms", run.latency_retrieval))
    lines.append(_format_latency_table("generation_elapsed_ms", run.latency_generation))
    lines.append(_format_latency_table("total_elapsed_ms", run.latency_total))

    lines.append("## 失败案例摘要")
    lines.append("")
    failures = [case for case in run.case_metrics if not case.strict_pass]
    if not failures:
        lines.append("无失败案例。")
    else:
        lines.append(
            f"共 {len(failures)} 个未通过 strict_pass 的题目，"
            f"最多展示 {MAX_FAILURE_EXAMPLES} 条。"
        )
        lines.append("")
        for case in failures[:MAX_FAILURE_EXAMPLES]:
            lines.append(f"### {case.case_id}")
            lines.append("")
            lines.append(f"- answer_status: `{case.answer_status}`")
            lines.append(f"- fact_coverage: {fmt_value(case.fact_coverage)}")
            lines.append(
                f"- grounded_fact_coverage: {fmt_value(case.grounded_fact_coverage)}"
            )
            lines.append(
                f"- invalid_citation: {case.invalid_citation_occurrence_count}"
            )
            lines.append(f"- malformed_citation: {case.malformed_citation_count}")
            lines.append(
                f"- forbidden_phrase_hits: {list(case.forbidden_phrase_hits)}"
            )
            lines.append(f"- strict_pass: {case.strict_pass}")
            lines.append("")
    lines.append("## 方法限制")
    lines.append("")
    for limitation in LIMITATIONS:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("## 复现命令")
    lines.append("")
    if config.mode == "offline":
        lines.append("```bash")
        lines.append(
            "python -m scripts.evaluate_answers "
            f"--dataset {config.dataset_path} "
            f"--annotations {config.annotations_path} "
            f"--responses {config.responses_path or '<responses.jsonl>'} "
            f"--output-dir <output-dir> "
            f"--run-name {config.run_name}"
        )
        lines.append("```")
    else:
        lines.append("```bash")
        lines.append(
            "python -m scripts.evaluate_answers --live "
            f"--dataset {config.dataset_path} "
            f"--annotations {config.annotations_path} "
            f"--output-dir <output-dir> "
            f"--run-name {config.run_name}"
        )
        lines.append("```")
    lines.append("")

    (output_dir / REPORT_FILENAME).write_text("\n".join(lines), encoding="utf-8")


def fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def fmt_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_answer_reports(run: AnswerEvaluationRun, output_dir: Path) -> None:
    """Write summary.json, cases.jsonl, and report.md for one run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_json(run, output_dir)
    _write_cases_jsonl(run, output_dir)
    _write_report_md(run, output_dir)
