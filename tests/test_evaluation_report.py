"""Offline tests for evaluation report generation."""

import csv
import json
from pathlib import Path

from src.evaluation.report import (
    CSV_COLUMNS,
    write_cases_csv,
    write_report_md,
    write_reports,
    write_summary,
)

from evaluation_helpers import make_result


def _results() -> list[object]:
    return [
        make_result(
            "q001",
            split="calibration",
            question="alpha 服务是什么？",
            category="direct",
            difficulty="easy",
            max_score=0.8,
            first_relevant_rank=1,
            sources=(
                {
                    "rank": 1,
                    "score": 0.8,
                    "text": "alpha 服务内容",
                    "chunk_index": 0,
                    "document_id": "eval-doc-a",
                    "filename": "a.md",
                    "page_number": None,
                },
            ),
        ),
        make_result(
            "q002",
            split="test",
            question="beta 数据库是什么？",
            category="paraphrase",
            difficulty="hard",
            answerable=False,
            max_score=0.9,
            first_relevant_rank=None,
            matched_at_1=0,
            matched_at_3=0,
            matched_at_5=0,
            total_evidence=0,
            reciprocal_rank=None,
            sources=(),
        ),
    ]


def _summary_data() -> dict[str, object]:
    results = _results()
    return {
        "schema_version": "1.0",
        "embedding_model": "fake/eval-embedding",
        "chunk_configuration": {"chunk_size": 300, "chunk_overlap": 50},
        "dataset_counts": {"total": 2, "answerable": 1, "unanswerable": 1},
        "split_counts": {"calibration": 1, "test": 1},
        "category_counts": {"direct": 1, "paraphrase": 1},
        "difficulty_counts": {"easy": 1, "hard": 1},
        "retrieval_metrics": {
            "overall": {
                "hit_at_1": 1.0,
                "hit_at_3": 1.0,
                "hit_at_5": 1.0,
                "recall_at_1": 1.0,
                "recall_at_3": 1.0,
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "answerable_count": 1,
            },
            "by_category": {},
            "by_difficulty": {},
            "by_split": {},
        },
        "calibration_threshold_table": [
            {
                "threshold": 0.35,
                "tp": 1,
                "fp": 1,
                "tn": 0,
                "fn": 0,
                "decision_accuracy": 0.5,
                "answerable_precision": 0.5,
                "answerable_recall": 1.0,
                "false_answer_rate": 1.0,
                "false_refusal_rate": 0.0,
                "answer_rate": 1.0,
                "refusal_rate": 0.0,
                "weighted_cost": 3.0,
            }
        ],
        "current_threshold": 0.35,
        "recommended_threshold": 0.35,
        "calibration_metrics_current": {
            "decision": {
                "tp": 1,
                "fp": 0,
                "tn": 1,
                "fn": 0,
                "decision_accuracy": 1.0,
                "answerable_precision": 1.0,
                "answerable_recall": 1.0,
                "false_answer_rate": 0.0,
                "false_refusal_rate": 0.0,
                "answer_rate": 0.5,
                "refusal_rate": 0.5,
            },
            "retrieval": {},
        },
        "calibration_metrics_recommended": {
            "decision": {
                "tp": 1,
                "fp": 0,
                "tn": 1,
                "fn": 0,
                "decision_accuracy": 1.0,
                "answerable_precision": 1.0,
                "answerable_recall": 1.0,
                "false_answer_rate": 0.0,
                "false_refusal_rate": 0.0,
                "answer_rate": 0.5,
                "refusal_rate": 0.5,
            },
            "retrieval": {},
        },
        "test_metrics_current": {
            "decision": {
                "tp": 0,
                "fp": 1,
                "tn": 0,
                "fn": 0,
                "decision_accuracy": 0.0,
                "answerable_precision": 0.0,
                "answerable_recall": None,
                "false_answer_rate": 1.0,
                "false_refusal_rate": None,
                "answer_rate": 1.0,
                "refusal_rate": 0.0,
            },
            "retrieval": {},
        },
        "test_metrics_recommended": {
            "decision": {
                "tp": 0,
                "fp": 1,
                "tn": 0,
                "fn": 0,
                "decision_accuracy": 0.0,
                "answerable_precision": 0.0,
                "answerable_recall": None,
                "false_answer_rate": 1.0,
                "false_refusal_rate": None,
                "answer_rate": 1.0,
                "refusal_rate": 0.0,
            },
            "retrieval": {
                "hit_at_5": 1.0,
                "mrr": 1.0,
            },
        },
        "score_distribution_summary": {
            "answerable": {
                "count": 1,
                "min": 0.8,
                "max": 0.8,
                "mean": 0.8,
                "median": 0.8,
                "p25": 0.8,
                "p75": 0.8,
            },
            "unanswerable": {
                "count": 1,
                "min": 0.9,
                "max": 0.9,
                "mean": 0.9,
                "median": 0.9,
                "p25": 0.9,
                "p75": 0.9,
            },
        },
        "failure_case_ids": {
            "retrieval_failures": [],
            "false_refusals": [],
            "false_answers": ["q002"],
        },
        "run_configuration": {
            "manifest": "corpus_manifest.json",
            "dataset": "dataset.jsonl",
            "top_k": 5,
            "threshold_start": 0.2,
            "threshold_end": 0.6,
            "threshold_step": 0.01,
            "current_threshold": 0.35,
            "false_answer_weight": 3.0,
            "false_refusal_weight": 1.0,
            "chunk_size": 300,
            "chunk_overlap": 50,
        },
    }


def test_summary_json_has_no_nan_or_infinity(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"

    write_summary(path, _summary_data())

    raw_text = path.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    json.loads(raw_text)
    parsed = json.loads(raw_text)
    assert parsed["schema_version"] == "1.0"


def test_summary_serializes_non_finite_values_as_null(tmp_path: Path) -> None:
    data = _summary_data()
    data["retrieval_metrics"]["overall"]["hit_at_1"] = float("nan")

    write_summary(tmp_path / "summary.json", data)

    parsed = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert parsed["retrieval_metrics"]["overall"]["hit_at_1"] is None


def test_summary_uses_utf8_and_stable_schema_version(tmp_path: Path) -> None:
    data = _summary_data()
    data["dataset_counts"]["note"] = "中文内容"

    write_summary(tmp_path / "summary.json", data)

    raw = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert "中文内容" in raw


def test_cases_csv_has_stable_columns_and_all_rows(tmp_path: Path) -> None:
    results = _results()

    write_cases_csv(
        tmp_path / "cases.csv",
        results,
        current_threshold=0.35,
        recommended_threshold=0.35,
    )

    raw = (tmp_path / "cases.csv").read_text(encoding="utf-8-sig")
    rows = list(csv.reader(raw.splitlines()))
    assert rows[0] == CSV_COLUMNS
    assert len(rows) == 3
    assert rows[1][0] == "q001"
    assert rows[2][0] == "q002"


def test_cases_csv_contains_chinese_question(tmp_path: Path) -> None:
    write_cases_csv(
        tmp_path / "cases.csv",
        _results(),
        current_threshold=0.35,
        recommended_threshold=0.35,
    )

    raw = (tmp_path / "cases.csv").read_text(encoding="utf-8-sig")
    assert "alpha 服务是什么" in raw


def test_report_md_contains_main_sections(tmp_path: Path) -> None:
    write_report_md(
        tmp_path / "report.md",
        _summary_data(),
        _results(),
    )

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    for section in [
        "# RAG 离线评估报告",
        "## 运行配置",
        "## 语料与数据集",
        "## 检索指标",
        "## 相似度分布摘要",
        "## 阈值扫描与推荐",
        "## 当前阈值与推荐阈值对比",
        "## 测试集最终结果",
        "## 主要失败案例",
        "## 已知限制",
        "## 后续建议",
    ]:
        assert section in text


def test_report_md_shows_empty_failure_section(tmp_path: Path) -> None:
    write_report_md(
        tmp_path / "report.md",
        _summary_data(),
        _results(),
    )

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "检索失败（answerable 且 Hit@5 未命中）：" not in text
    assert "错误拒答（" not in text
    assert "### 失败案例明细" in text


def test_report_md_compares_current_and_recommended_thresholds(
    tmp_path: Path,
) -> None:
    write_report_md(
        tmp_path / "report.md",
        _summary_data(),
        _results(),
    )

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Calibration | 0.35" in text
    assert "当前生产阈值：0.35" in text
    assert "推荐阈值仅作为建议" in text


def test_write_reports_is_reproducible(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    write_reports(first_dir, _summary_data(), _results(), 0.35, 0.35)
    write_reports(second_dir, _summary_data(), _results(), 0.35, 0.35)

    for name in ("summary.json", "cases.csv", "report.md"):
        first = (first_dir / name).read_bytes()
        second = (second_dir / name).read_bytes()
        assert first == second
