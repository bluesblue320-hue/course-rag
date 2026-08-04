"""Render A/B reranking evaluation runs as JSON, CSV, and Markdown reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation.reranking_metrics import (
    BranchMetrics,
    CandidateMetrics,
    MetricDelta,
    MetricDeltas,
    summarize_latency,
)
from src.evaluation.reranking_models import RerankingCaseResult
from src.evaluation.reranking_runner import (
    FOCUS_CASE_IDS,
    RerankingRun,
)

SCHEMA_VERSION = "2.0.0"
SUMMARY_FILENAME = "summary.json"
CASES_FILENAME = "cases.csv"
REPORT_FILENAME = "report.md"
LATENCY_FILENAME = "latency.json"


CASES_CSV_COLUMNS: tuple[str, ...] = (
    "case_id",
    "split",
    "category",
    "difficulty",
    "answerable",
    "candidate_first_relevant_rank",
    "candidate_hit_at_5",
    "candidate_hit_at_10",
    "candidate_hit_at_15",
    "candidate_recall_at_5",
    "candidate_recall_at_10",
    "candidate_recall_at_15",
    "vector_first_relevant_rank",
    "vector_hit_at_1",
    "vector_hit_at_3",
    "vector_hit_at_5",
    "vector_recall_at_1",
    "vector_recall_at_3",
    "vector_recall_at_5",
    "vector_reciprocal_rank",
    "reranked_first_relevant_rank",
    "reranked_hit_at_1",
    "reranked_hit_at_3",
    "reranked_hit_at_5",
    "reranked_recall_at_1",
    "reranked_recall_at_3",
    "reranked_recall_at_5",
    "reranked_reciprocal_rank",
    "rank_change",
    "improved",
    "regressed",
    "unchanged",
)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _branch_metrics_payload(metrics: BranchMetrics) -> dict[str, object]:
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


def _candidate_metrics_payload(metrics: CandidateMetrics) -> dict[str, object]:
    return {
        "case_count": metrics.case_count,
        "hit_at_5": _round(metrics.hit_at_5),
        "hit_at_10": _round(metrics.hit_at_10),
        "hit_at_15": _round(metrics.hit_at_15),
        "recall_at_5": _round(metrics.recall_at_5),
        "recall_at_10": _round(metrics.recall_at_10),
        "recall_at_15": _round(metrics.recall_at_15),
        "mean_reciprocal_rank": _round(metrics.mean_reciprocal_rank),
    }


def _delta_payload(delta: MetricDelta) -> dict[str, object]:
    return {
        "vector_value": _round(delta.vector_value),
        "reranked_value": _round(delta.reranked_value),
        "absolute_delta": _round(delta.absolute_delta),
        "relative_delta": _round(delta.relative_delta),
    }


def _deltas_payload(deltas: MetricDeltas) -> dict[str, object]:
    return {
        "hit_at_1": _delta_payload(deltas.hit_at_1),
        "hit_at_3": _delta_payload(deltas.hit_at_3),
        "hit_at_5": _delta_payload(deltas.hit_at_5),
        "recall_at_1": _delta_payload(deltas.recall_at_1),
        "recall_at_3": _delta_payload(deltas.recall_at_3),
        "recall_at_5": _delta_payload(deltas.recall_at_5),
        "mean_reciprocal_rank": _delta_payload(deltas.mean_reciprocal_rank),
    }


def _build_summary(run: RerankingRun) -> dict[str, object]:
    """Build the complete summary.json payload."""
    answerable_count = sum(1 for c in run.cases if c.answerable)
    unanswerable_count = len(run.cases) - answerable_count

    categories: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    splits: dict[str, int] = {}
    for case in run.cases:
        categories[case.category] = categories.get(case.category, 0) + 1
        difficulties[case.difficulty] = difficulties.get(case.difficulty, 0) + 1
        splits[case.split] = splits.get(case.split, 0) + 1

    # Focus cases
    focus_results: list[dict[str, object]] = []
    for case_id in FOCUS_CASE_IDS:
        for r in run.results:
            if r.case_id == case_id:
                focus_results.append({
                    "case_id": r.case_id,
                    "question": r.question,
                    "candidate_first_relevant_rank": (
                        r.candidate.first_relevant_rank
                        if r.candidate
                        else None
                    ),
                    "candidate_hit_at_15": (
                        r.candidate.hit_at_15 if r.candidate else None
                    ),
                    "vector_first_relevant_rank": (
                        r.vector.first_relevant_rank
                        if r.vector
                        else None
                    ),
                    "reranked_first_relevant_rank": (
                        r.reranked.first_relevant_rank
                        if r.reranked
                        else None
                    ),
                    "rank_change": r.rank_change,
                    "improved": r.improved,
                    "regressed": r.regressed,
                    "reranker_applied": r.reranker_applied,
                    "reranker_fallback": r.reranker_fallback,
                })
                break

    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": run.configuration.embedding_model,
        "reranker_model": run.configuration.reranker_model,
        "candidate_top_k": run.configuration.candidate_top_k,
        "final_top_k": run.configuration.final_top_k,
        "dataset_counts": {
            "total": len(run.cases),
            "answerable": answerable_count,
            "unanswerable": unanswerable_count,
        },
        "candidate_metrics": _candidate_metrics_payload(
            run.candidate_metrics
        ),
        "vector_metrics": _branch_metrics_payload(run.vector_metrics),
        "reranked_metrics": _branch_metrics_payload(run.reranked_metrics),
        "metric_deltas": _deltas_payload(run.metric_deltas),
        "metrics_by_category": {
            "vector": {
                k: _branch_metrics_payload(v)
                for k, v in run.vector_metrics_by_category.items()
            },
            "reranked": {
                k: _branch_metrics_payload(v)
                for k, v in run.reranked_metrics_by_category.items()
            },
        },
        "metrics_by_difficulty": {
            "vector": {
                k: _branch_metrics_payload(v)
                for k, v in run.vector_metrics_by_difficulty.items()
            },
            "reranked": {
                k: _branch_metrics_payload(v)
                for k, v in run.reranked_metrics_by_difficulty.items()
            },
        },
        "metrics_by_split": {
            "vector": {
                k: _branch_metrics_payload(v)
                for k, v in run.vector_metrics_by_split.items()
            },
            "reranked": {
                k: _branch_metrics_payload(v)
                for k, v in run.reranked_metrics_by_split.items()
            },
        },
        "focus_case_results": focus_results,
        "improved_case_ids": list(run.improved_case_ids),
        "regressed_case_ids": list(run.regressed_case_ids),
        "unchanged_case_ids": list(run.unchanged_case_ids),
        "decision_invariance_check": {
            "passed": run.decision_invariance_passed,
            "note": (
                "拒答决策基于 max_retrieval_score，vector-only 和 reranked "
                "分支共享同一候选池，因此决策结果必然一致。"
            ),
        },
        "split_roles": [
            {
                "split": "calibration",
                "role": "用于阈值校准的 split，在本 A/B 中也参与指标计算。",
            },
            {
                "split": "test",
                "role": (
                    "test 结果公开后应视为已发布的回归测试集。"
                    "正式比较多个 Reranker 方案时，需要新增一份"
                    "从未用于开发决策的独立 final holdout。"
                ),
            },
        ],
        "reproducibility": {
            "fake_reranker": (
                "自动化测试使用 FakeReranker，在相同输入和代码下"
                "输出逐字节稳定。"
            ),
            "real_reranker": (
                "真实 Reranker 基线在相同代码、模型、依赖版本和环境下"
                "才应当稳定。"
            ),
            "latency": (
                "延迟数据依赖设备和运行环境，不写入确定性质量 baseline。"
            ),
        },
        "known_limits": [
            "语料规模有限（25 Chunk），指标存在抽样波动。",
            "不评估生成答案质量，只评估检索排序。",
            "test split 已公开，作为回归基准使用，不是完全未见的最终结果。",
            "Reranker 分数语义不同于向量相似度，不可直接比较或复用阈值。",
            "延迟结果依赖设备和运行环境。",
        ],
        "recommendation": {
            "recommend_enable": run.recommend_enable,
            "reasons": list(run.recommendation_reasons),
            "note": (
                "即使达到推荐标准，也不自动修改生产配置。"
                "RAG_RERANKER_ENABLED 仍需人工评估后显式设置。"
            ),
        },
    }


def _write_summary_json(run: RerankingRun, path: Path) -> None:
    payload = _build_summary(run)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_cases_csv(run: RerankingRun, path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CASES_CSV_COLUMNS))
        writer.writeheader()
        for r in run.results:
            row: dict[str, object] = {
                "case_id": r.case_id,
                "split": r.split,
                "category": r.category,
                "difficulty": r.difficulty,
                "answerable": r.answerable,
            }
            if r.candidate is not None:
                row["candidate_first_relevant_rank"] = (
                    r.candidate.first_relevant_rank or ""
                )
                row["candidate_hit_at_5"] = r.candidate.hit_at_5
                row["candidate_hit_at_10"] = r.candidate.hit_at_10
                row["candidate_hit_at_15"] = r.candidate.hit_at_15
                row["candidate_recall_at_5"] = _round(r.candidate.recall_at_5) or ""
                row["candidate_recall_at_10"] = _round(r.candidate.recall_at_10) or ""
                row["candidate_recall_at_15"] = _round(r.candidate.recall_at_15) or ""
            else:
                for col in CASES_CSV_COLUMNS:
                    if col.startswith("candidate_"):
                        row[col] = ""
            if r.vector is not None:
                row["vector_first_relevant_rank"] = (
                    r.vector.first_relevant_rank or ""
                )
                row["vector_hit_at_1"] = r.vector.hit_at_1
                row["vector_hit_at_3"] = r.vector.hit_at_3
                row["vector_hit_at_5"] = r.vector.hit_at_5
                row["vector_recall_at_1"] = _round(r.vector.recall_at_1) or ""
                row["vector_recall_at_3"] = _round(r.vector.recall_at_3) or ""
                row["vector_recall_at_5"] = _round(r.vector.recall_at_5) or ""
                row["vector_reciprocal_rank"] = _round(r.vector.reciprocal_rank) or ""
            if r.reranked is not None:
                row["reranked_first_relevant_rank"] = (
                    r.reranked.first_relevant_rank or ""
                )
                row["reranked_hit_at_1"] = r.reranked.hit_at_1
                row["reranked_hit_at_3"] = r.reranked.hit_at_3
                row["reranked_hit_at_5"] = r.reranked.hit_at_5
                row["reranked_recall_at_1"] = _round(r.reranked.recall_at_1) or ""
                row["reranked_recall_at_3"] = _round(r.reranked.recall_at_3) or ""
                row["reranked_recall_at_5"] = _round(r.reranked.recall_at_5) or ""
                row["reranked_reciprocal_rank"] = _round(r.reranked.reciprocal_rank) or ""
            row["rank_change"] = r.rank_change if r.rank_change is not None else ""
            row["improved"] = r.improved
            row["regressed"] = r.regressed
            row["unchanged"] = r.unchanged
            writer.writerow(row)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _fmt_delta(delta: MetricDelta, digits: int = 4) -> str:
    if delta.absolute_delta is None:
        return "N/A"
    sign = "+" if delta.absolute_delta >= 0 else ""
    rel = ""
    if delta.relative_delta is not None:
        rel = f" ({sign}{delta.relative_delta * 100:.2f}%)"
    return f"{sign}{delta.absolute_delta:.{digits}f}{rel}"


def _write_report_md(run: RerankingRun, path: Path) -> None:
    lines: list[str] = []
    vm = run.vector_metrics
    rm = run.reranked_metrics
    cm = run.candidate_metrics
    d = run.metric_deltas

    lines.append("# RAG Reranking A/B 评估报告")
    lines.append("")

    # Experiment goal
    lines.append("## 实验目标")
    lines.append("")
    lines.append("验证可选的两阶段检索（CrossEncoder Reranker）能否：")
    lines.append("1. 正确证据稳定进入更大的候选池")
    lines.append("2. Reranker 把正确证据从候选池后部提升到 Top-1 / Top-3")
    lines.append("3. paraphrase 问题明显改善")
    lines.append("4. 原有 direct 和 multi-evidence 问题不退化")
    lines.append("5. Reranker 失败时安全回退到 vector-only")
    lines.append("6. 额外模型和延迟成本是否值得")
    lines.append("")

    # Configuration
    lines.append("## 运行配置")
    lines.append("")
    lines.append(f"- Embedding 模型: `{run.configuration.embedding_model}`")
    lines.append(f"- Reranker 模型: `{run.configuration.reranker_model}`")
    lines.append(f"- Candidate Top-K: {run.configuration.candidate_top_k}")
    lines.append(f"- Final Top-K: {run.configuration.final_top_k}")
    lines.append(f"- 数据集题目数: {len(run.cases)}")
    lines.append("")

    # Split roles
    lines.append("## 数据集角色")
    lines.append("")
    lines.append("- **calibration**: 用于阈值校准，在本 A/B 中也参与指标计算。")
    lines.append("- **test**: 已公开，作为回归基准使用，不是完全未见的最终结果。")
    lines.append("- 正式比较多个 Reranker 方案时，需要新增独立 final holdout。")
    lines.append("")

    # Candidate pool quality
    lines.append("## 候选池质量")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | ---: |")
    lines.append(f"| Candidate Hit@5 | {_fmt(cm.hit_at_5)} |")
    lines.append(f"| Candidate Hit@10 | {_fmt(cm.hit_at_10)} |")
    lines.append(f"| Candidate Hit@15 | {_fmt(cm.hit_at_15)} |")
    lines.append(f"| Candidate Recall@5 | {_fmt(cm.recall_at_5)} |")
    lines.append(f"| Candidate Recall@10 | {_fmt(cm.recall_at_10)} |")
    lines.append(f"| Candidate Recall@15 | {_fmt(cm.recall_at_15)} |")
    lines.append(f"| Candidate MRR | {_fmt(cm.mean_reciprocal_rank)} |")
    lines.append("")

    # Vector-only baseline
    lines.append("## Vector-only 基线")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | ---: |")
    lines.append(f"| Hit@1 | {_fmt(vm.hit_at_1)} |")
    lines.append(f"| Hit@3 | {_fmt(vm.hit_at_3)} |")
    lines.append(f"| Hit@5 | {_fmt(vm.hit_at_5)} |")
    lines.append(f"| Recall@1 | {_fmt(vm.recall_at_1)} |")
    lines.append(f"| Recall@3 | {_fmt(vm.recall_at_3)} |")
    lines.append(f"| Recall@5 | {_fmt(vm.recall_at_5)} |")
    lines.append(f"| MRR | {_fmt(vm.mean_reciprocal_rank)} |")
    lines.append("")

    # Reranked results
    lines.append("## Reranked 结果")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | ---: |")
    lines.append(f"| Hit@1 | {_fmt(rm.hit_at_1)} |")
    lines.append(f"| Hit@3 | {_fmt(rm.hit_at_3)} |")
    lines.append(f"| Hit@5 | {_fmt(rm.hit_at_5)} |")
    lines.append(f"| Recall@1 | {_fmt(rm.recall_at_1)} |")
    lines.append(f"| Recall@3 | {_fmt(rm.recall_at_3)} |")
    lines.append(f"| Recall@5 | {_fmt(rm.recall_at_5)} |")
    lines.append(f"| MRR | {_fmt(rm.mean_reciprocal_rank)} |")
    lines.append("")

    # Overall deltas
    lines.append("## 整体指标差值")
    lines.append("")
    lines.append("| 指标 | Vector | Reranked | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| Hit@1 | {_fmt(vm.hit_at_1)} | {_fmt(rm.hit_at_1)} | {_fmt_delta(d.hit_at_1)} |")
    lines.append(f"| Hit@3 | {_fmt(vm.hit_at_3)} | {_fmt(rm.hit_at_3)} | {_fmt_delta(d.hit_at_3)} |")
    lines.append(f"| Hit@5 | {_fmt(vm.hit_at_5)} | {_fmt(rm.hit_at_5)} | {_fmt_delta(d.hit_at_5)} |")
    lines.append(f"| Recall@5 | {_fmt(vm.recall_at_5)} | {_fmt(rm.recall_at_5)} | {_fmt_delta(d.recall_at_5)} |")
    lines.append(f"| MRR | {_fmt(vm.mean_reciprocal_rank)} | {_fmt(rm.mean_reciprocal_rank)} | {_fmt_delta(d.mean_reciprocal_rank)} |")
    lines.append("")

    # By category
    lines.append("## 按 category 对比")
    lines.append("")
    lines.append("| Category | Vector Hit@1 | Reranked Hit@1 | Vector MRR | Reranked MRR |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    all_categories = sorted(
        set(run.vector_metrics_by_category.keys())
        | set(run.reranked_metrics_by_category.keys())
    )
    for cat in all_categories:
        v = run.vector_metrics_by_category.get(cat)
        r = run.reranked_metrics_by_category.get(cat)
        lines.append(
            f"| {cat} | {_fmt(v.hit_at_1 if v else None)} |"
            f" {_fmt(r.hit_at_1 if r else None)} |"
            f" {_fmt(v.mean_reciprocal_rank if v else None)} |"
            f" {_fmt(r.mean_reciprocal_rank if r else None)} |"
        )
    lines.append("")

    # By difficulty
    lines.append("## 按 difficulty 对比")
    lines.append("")
    lines.append("| Difficulty | Vector Hit@1 | Reranked Hit@1 | Vector MRR | Reranked MRR |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    all_difficulties = sorted(
        set(run.vector_metrics_by_difficulty.keys())
        | set(run.reranked_metrics_by_difficulty.keys())
    )
    for diff in all_difficulties:
        v = run.vector_metrics_by_difficulty.get(diff)
        r = run.reranked_metrics_by_difficulty.get(diff)
        lines.append(
            f"| {diff} | {_fmt(v.hit_at_1 if v else None)} |"
            f" {_fmt(r.hit_at_1 if r else None)} |"
            f" {_fmt(v.mean_reciprocal_rank if v else None)} |"
            f" {_fmt(r.mean_reciprocal_rank if r else None)} |"
        )
    lines.append("")

    # Paraphrase focus
    lines.append("## 重点 paraphrase 对比")
    lines.append("")
    para_v = run.vector_metrics_by_category.get("paraphrase")
    para_r = run.reranked_metrics_by_category.get("paraphrase")
    lines.append("| 指标 | Vector | Reranked |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| Hit@1 | {_fmt(para_v.hit_at_1 if para_v else None)} | {_fmt(para_r.hit_at_1 if para_r else None)} |")
    lines.append(f"| Hit@3 | {_fmt(para_v.hit_at_3 if para_v else None)} | {_fmt(para_r.hit_at_3 if para_r else None)} |")
    lines.append(f"| Hit@5 | {_fmt(para_v.hit_at_5 if para_v else None)} | {_fmt(para_r.hit_at_5 if para_r else None)} |")
    lines.append(f"| MRR | {_fmt(para_v.mean_reciprocal_rank if para_v else None)} | {_fmt(para_r.mean_reciprocal_rank if para_r else None)} |")
    lines.append("")

    # Focus cases
    lines.append("## p-001 / p-010 / p-012")
    lines.append("")
    for case_id in FOCUS_CASE_IDS:
        r = next((r for r in run.results if r.case_id == case_id), None)
        if r is None:
            lines.append(f"### {case_id}: 未找到")
            lines.append("")
            continue
        lines.append(f"### {case_id}")
        lines.append(f"- 问题: {r.question}")
        lines.append(f"- 是否进入 Candidate Top-15: {r.candidate is not None and r.candidate.hit_at_15 if r.candidate else 'N/A'}")
        if r.candidate:
            lines.append(f"- Candidate rank: {r.candidate.first_relevant_rank or 'miss'}")
        if r.vector:
            lines.append(f"- Vector rank: {r.vector.first_relevant_rank or 'miss'}")
        if r.reranked:
            lines.append(f"- Reranked rank: {r.reranked.first_relevant_rank or 'miss'}")
        lines.append(f"- Rank change: {r.rank_change if r.rank_change is not None else 'N/A'}")
        top5_vector_hit = r.vector.hit_at_5 if r.vector else False
        top5_reranked_hit = r.reranked.hit_at_5 if r.reranked else False
        lines.append(f"- Top-5 vector: {'hit' if top5_vector_hit else 'miss'} → Top-5 reranked: {'hit' if top5_reranked_hit else 'miss'}")
        lines.append("")

    # Improved cases
    lines.append("## 改善案例")
    lines.append("")
    if run.improved_case_ids:
        lines.append("| Case ID | Vector Rank | Reranked Rank | Rank Change |")
        lines.append("| --- | ---: | ---: | ---: |")
        for cid in run.improved_case_ids:
            r = next((x for x in run.results if x.case_id == cid), None)
            if r and r.vector and r.reranked:
                lines.append(
                    f"| {cid} | {r.vector.first_relevant_rank or 'miss'} |"
                    f" {r.reranked.first_relevant_rank or 'miss'} |"
                    f" {r.rank_change or 0} |"
                )
    else:
        lines.append("无改善案例。")
    lines.append("")

    # Regressed cases
    lines.append("## 退化案例")
    lines.append("")
    if run.regressed_case_ids:
        lines.append("| Case ID | Vector Rank | Reranked Rank | Rank Change |")
        lines.append("| --- | ---: | ---: | ---: |")
        for cid in run.regressed_case_ids:
            r = next((x for x in run.results if x.case_id == cid), None)
            if r and r.vector and r.reranked:
                lines.append(
                    f"| {cid} | {r.vector.first_relevant_rank or 'miss'} |"
                    f" {r.reranked.first_relevant_rank or 'miss'} |"
                    f" {r.rank_change or 0} |"
                )
    else:
        lines.append("无退化案例。")
    lines.append("")

    # Decision invariance
    lines.append("## 拒答决策一致性")
    lines.append("")
    lines.append(
        f"- 决策一致性检查: {'通过' if run.decision_invariance_passed else '未通过'}"
    )
    lines.append("- 拒答基于 max_retrieval_score，两个分支共享同一候选池，决策必然一致。")
    lines.append("")

    # Latency
    lines.append("## 延迟说明")
    lines.append("")
    if run.latency_records:
        from src.evaluation.reranking_metrics import summarize_latency as sl
        vr = sl(run.latency_records, "vector_retrieval_ms")
        rr = sl(run.latency_records, "rerank_ms")
        ee = sl(run.latency_records, "end_to_end_ms")
        lines.append("| 阶段 | Count | Mean (ms) | Median (ms) | P95 (ms) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        lines.append(f"| Vector Retrieval | {vr.count} | {_fmt(vr.mean, 2)} | {_fmt(vr.median, 2)} | {_fmt(vr.p95, 2)} |")
        lines.append(f"| Rerank | {rr.count} | {_fmt(rr.mean, 2)} | {_fmt(rr.median, 2)} | {_fmt(rr.p95, 2)} |")
        lines.append(f"| End-to-End | {ee.count} | {_fmt(ee.mean, 2)} | {_fmt(ee.median, 2)} | {_fmt(ee.p95, 2)} |")
    else:
        lines.append("本次运行未启用延迟测量（`--include-latency` 默认关闭）。")
    lines.append("- 延迟数据依赖设备和运行环境，不写入确定性质量 baseline。")
    lines.append("")

    # Reproducibility
    lines.append("## 可重复性边界")
    lines.append("")
    lines.append("- Fake Reranker 模式下，输出逐字节稳定。")
    lines.append("- 真实 Reranker 基线在相同代码、模型、依赖版本和环境下才稳定。")
    lines.append("- 延迟数据不写入提交的确定性 baseline。")
    lines.append("")

    # Known limits
    lines.append("## 已知限制")
    lines.append("")
    lines.append("- 语料规模有限（25 Chunk），指标存在抽样波动。")
    lines.append("- 不评估生成答案质量，只评估检索排序。")
    lines.append("- test split 已公开，作为回归基准使用。")
    lines.append("- Reranker 分数语义不同于向量相似度，不可直接比较或复用阈值。")
    lines.append("- 延迟结果依赖设备和运行环境。")
    lines.append("")

    # Recommendation
    lines.append("## 是否建议启用生产 Reranker")
    lines.append("")
    if run.recommend_enable:
        lines.append("**建议启用**，满足以下条件：")
    else:
        lines.append("**暂不建议启用**，原因如下：")
    lines.append("")
    for reason in run.recommendation_reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("项目验收标准（非行业标准）：")
    lines.append(f"- Hit@1 绝对提升 >= {0.05}")
    lines.append(f"- MRR 绝对提升 >= {0.03}")
    lines.append(f"- Paraphrase MRR 提升 >= {0.05}")
    lines.append(f"- Hit@5 退化 <= {0.01}")
    lines.append("- 退化案例数不超过 5")
    lines.append("- 即使达到标准，也不自动修改生产配置。")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_latency_json(run: RerankingRun, path: Path) -> None:
    from src.evaluation.reranking_metrics import summarize_latency as sl
    vr = sl(run.latency_records, "vector_retrieval_ms")
    rr = sl(run.latency_records, "rerank_ms")
    ee = sl(run.latency_records, "end_to_end_ms")

    def _stage(s) -> dict[str, object]:
        return {
            "count": s.count,
            "mean": _round(s.mean, 4),
            "median": _round(s.median, 4),
            "p95": _round(s.p95, 4),
        }

    payload = {
        "vector_retrieval_ms": _stage(vr),
        "rerank_ms": _stage(rr),
        "end_to_end_ms": _stage(ee),
        "note": "延迟数据依赖设备和运行环境，不写入确定性质量 baseline。",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_reranking_reports(
    run: RerankingRun,
    output_dir: Path,
) -> None:
    """Write all report files to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_json(run, output_dir / SUMMARY_FILENAME)
    _write_cases_csv(run, output_dir / CASES_FILENAME)
    _write_report_md(run, output_dir / REPORT_FILENAME)
    if run.latency_records:
        _write_latency_json(run, output_dir / LATENCY_FILENAME)
