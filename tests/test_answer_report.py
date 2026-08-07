"""Tests for answer-quality report rendering (determinism and safety)."""

import json
import math
from pathlib import Path

from src.evaluation.answer_annotations import load_answer_annotations
from src.evaluation.answer_report import (
    LIMITATIONS,
    CASES_FILENAME,
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    write_answer_reports,
)
from src.evaluation.answer_responses import load_answer_responses
from src.evaluation.answer_runner import run_offline_answer_evaluation
from src.evaluation.dataset import load_dataset

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "answer_evaluation"


def _build_run(tmp_path: Path):
    cases = load_dataset(FIXTURE_ROOT / "dataset.jsonl")
    annotations = load_answer_annotations(FIXTURE_ROOT / "annotations.jsonl", cases)
    responses = load_answer_responses(FIXTURE_ROOT / "responses.jsonl", cases)
    return run_offline_answer_evaluation(
        cases=cases,
        annotations=annotations,
        responses=responses,
        dataset_path="tests/fixtures/answer_evaluation/dataset.jsonl",
        annotations_path="tests/fixtures/answer_evaluation/annotations.jsonl",
        responses_path="tests/fixtures/answer_evaluation/responses.jsonl",
        run_name="local-fixture",
        model_label="deterministic-fixture",
    )


class TestReports:
    def test_all_three_files_are_written(self, tmp_path: Path) -> None:
        run = _build_run(tmp_path)
        write_answer_reports(run, tmp_path / "out")
        for filename in (SUMMARY_FILENAME, CASES_FILENAME, REPORT_FILENAME):
            assert (tmp_path / "out" / filename).is_file()

    def test_summary_is_strict_json(self, tmp_path: Path) -> None:
        run = _build_run(tmp_path)
        write_answer_reports(run, tmp_path / "out")
        payload = json.loads(
            (tmp_path / "out" / SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
        assert payload["evaluation_type"] == "answer_quality"
        assert payload["schema_version"] == run.schema_version
        assert payload["dataset_counts"]["total"] == 8
        assert "limitations" in payload
        assert "strict_failure_case_ids" in payload

    def test_cases_jsonl_is_strict_json_per_line(self, tmp_path: Path) -> None:
        run = _build_run(tmp_path)
        write_answer_reports(run, tmp_path / "out")
        lines = [
            line
            for line in (tmp_path / "out" / CASES_FILENAME)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert len(lines) == 8
        for line in lines:
            json.loads(line)

    def test_markdown_contains_required_sections(self, tmp_path: Path) -> None:
        run = _build_run(tmp_path)
        write_answer_reports(run, tmp_path / "out")
        report = (tmp_path / "out" / REPORT_FILENAME).read_text(encoding="utf-8")
        for section in (
            "运行配置",
            "数据集统计",
            "指标定义",
            "整体指标",
            "按 split 分组",
            "按 category 分组",
            "按 difficulty 分组",
            "引用质量",
            "必要事实覆盖",
            "严格通过率",
            "延迟统计",
            "失败案例摘要",
            "方法限制",
            "复现命令",
        ):
            assert section in report, f"缺少章节: {section}"

    def test_byte_identical_across_runs(self, tmp_path: Path) -> None:
        run_a = _build_run(tmp_path)
        run_b = _build_run(tmp_path)
        write_answer_reports(run_a, tmp_path / "out-a")
        write_answer_reports(run_b, tmp_path / "out-b")
        for filename in (SUMMARY_FILENAME, CASES_FILENAME, REPORT_FILENAME):
            content_a = (tmp_path / "out-a" / filename).read_bytes()
            content_b = (tmp_path / "out-b" / filename).read_bytes()
            assert content_a == content_b

    def test_no_timestamp_in_outputs(self, tmp_path: Path) -> None:
        run = _build_run(tmp_path)
        write_answer_reports(run, tmp_path / "out")
        for filename in (SUMMARY_FILENAME, CASES_FILENAME, REPORT_FILENAME):
            text = (tmp_path / "out" / filename).read_text(encoding="utf-8")
            assert "2026-" not in text

    def test_no_absolute_path_in_outputs(self, tmp_path: Path) -> None:
        run = _build_run(tmp_path)
        write_answer_reports(run, tmp_path / "out")
        for filename in (SUMMARY_FILENAME, CASES_FILENAME, REPORT_FILENAME):
            text = (tmp_path / "out" / filename).read_text(encoding="utf-8")
            assert "C:" not in text
            assert str(tmp_path) not in text

    def test_no_nan_or_infinity_in_outputs(self, tmp_path: Path) -> None:
        run = _build_run(tmp_path)
        write_answer_reports(run, tmp_path / "out")
        summary = json.loads(
            (tmp_path / "out" / SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
        assert not _contains_non_finite(summary)
        for line in (tmp_path / "out" / CASES_FILENAME).read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                assert not _contains_non_finite(json.loads(line))

    def test_limitations_are_documented(self, tmp_path: Path) -> None:
        run = _build_run(tmp_path)
        write_answer_reports(run, tmp_path / "out")
        summary = json.loads(
            (tmp_path / "out" / SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
        assert len(summary["limitations"]) == len(LIMITATIONS)
        assert any("人工标注" in item for item in summary["limitations"])


def _contains_non_finite(value: object) -> bool:
    """Return True when any nested number is NaN or infinite."""
    import math

    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_non_finite(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite(v) for v in value)
    return False
