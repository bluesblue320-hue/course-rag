"""Tests for the JSON, CSV, and Markdown report writers."""

import csv
import json
from pathlib import Path

import pytest

from src.evaluation.corpus import load_corpus, load_manifest
from src.evaluation.dataset import load_dataset
from src.evaluation.report import (
    CASES_CSV_COLUMNS,
    CASES_FILENAME,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    build_case_rows,
    build_markdown_report,
    build_summary,
    false_answers,
    false_refusals,
    retrieval_failures,
    score_overlap,
    write_cases_csv,
    write_reports,
    write_summary_json,
)
from src.evaluation.runner import EvaluationRun, run_evaluation
from tests.evaluation_helpers import (
    FakeEmbeddingService,
    make_result,
    write_mini_corpus,
    write_mini_dataset,
)


def build_run(tmp_path: Path) -> EvaluationRun:
    """Build one real run over the mini corpus with the fake embedding."""
    manifest_path = write_mini_corpus(tmp_path)
    dataset_path = write_mini_dataset(tmp_path)
    corpus = load_corpus(load_manifest(manifest_path, base_dir=tmp_path))
    return run_evaluation(
        corpus=corpus,
        cases=load_dataset(dataset_path),
        embedding_service=FakeEmbeddingService(),
        embedding_model="fake-bigram-hash",
        manifest_path="eval/corpus_manifest.json",
        dataset_path="eval/dataset.jsonl",
        top_k=5,
        threshold_start=0.2,
        threshold_end=0.6,
        threshold_step=0.05,
    )


@pytest.fixture()
def run(tmp_path: Path) -> EvaluationRun:
    return build_run(tmp_path / "run")


class TestBuildSummary:
    def test_it_reports_the_schema_version_and_model(
        self,
        run: EvaluationRun,
    ) -> None:
        summary = build_summary(run)

        assert summary["schema_version"] == SCHEMA_VERSION
        assert summary["embedding_model"] == "fake-bigram-hash"

    def test_it_reports_the_dataset_distribution(
        self,
        run: EvaluationRun,
    ) -> None:
        summary = build_summary(run)

        assert summary["dataset_counts"] == {
            "total": 5,
            "answerable": 3,
            "unanswerable": 2,
        }
        assert summary["split_counts"] == {"calibration": 3, "test": 2}

    def test_it_records_both_thresholds_without_changing_production(
        self,
        run: EvaluationRun,
    ) -> None:
        summary = build_summary(run)

        assert summary["current_threshold"] == pytest.approx(0.35)
        assert summary["recommended_threshold"] == pytest.approx(
            run.recommended_threshold
        )

    def test_the_threshold_table_covers_the_whole_grid(
        self,
        run: EvaluationRun,
    ) -> None:
        summary = build_summary(run)
        table = summary["calibration_threshold_table"]

        assert isinstance(table, list)
        assert len(table) == len(run.calibration_candidates)
        assert table[0]["threshold"] == pytest.approx(0.2)
        assert table[-1]["threshold"] == pytest.approx(0.6)

    def test_it_lists_failure_case_ids_for_both_thresholds(
        self,
        run: EvaluationRun,
    ) -> None:
        summary = build_summary(run)
        failures = summary["failure_case_ids"]

        assert set(failures) == {
            "retrieval_miss_at_5",
            "false_refusal_current",
            "false_answer_current",
            "false_refusal_recommended",
            "false_answer_recommended",
        }
        known_ids = {case.id for case in run.cases}
        for ids in failures.values():
            assert set(ids) <= known_ids

    def test_it_is_json_serializable_without_nan(
        self,
        run: EvaluationRun,
    ) -> None:
        text = json.dumps(build_summary(run), ensure_ascii=False, allow_nan=False)

        assert "NaN" not in text
        assert "Infinity" not in text

    def test_it_never_leaks_an_absolute_local_path(
        self,
        run: EvaluationRun,
    ) -> None:
        summary = build_summary(run)

        assert summary["run_configuration"]["manifest_path"] == (
            "eval/corpus_manifest.json"
        )
        assert ":" not in str(summary["run_configuration"]["dataset_path"])


class TestScoreOverlap:
    def test_it_returns_none_when_the_two_groups_are_separated(self) -> None:
        run = _fake_run(
            answerable_scores=[0.80, 0.75],
            unanswerable_scores=[0.20, 0.30],
        )

        assert score_overlap(run) is None

    def test_it_reports_the_band_and_the_case_counts(self) -> None:
        run = _fake_run(
            answerable_scores=[0.40, 0.70],
            unanswerable_scores=[0.20, 0.55],
        )

        overlap = score_overlap(run)

        assert overlap is not None
        lower, upper, answerable_in_band, unanswerable_in_band = overlap
        assert lower == pytest.approx(0.40)
        assert upper == pytest.approx(0.55)
        assert answerable_in_band == 1
        assert unanswerable_in_band == 1

    def test_it_returns_none_when_one_group_is_missing(self) -> None:
        run = _fake_run(answerable_scores=[0.5], unanswerable_scores=[])

        assert score_overlap(run) is None

    def test_the_markdown_states_overlap_honestly(self) -> None:
        overlapping = build_markdown_report(
            _fake_run(
                answerable_scores=[0.40, 0.70],
                unanswerable_scores=[0.20, 0.55],
            )
        )
        separated = build_markdown_report(
            _fake_run(
                answerable_scores=[0.80, 0.75],
                unanswerable_scores=[0.20, 0.30],
            )
        )

        assert "重叠" in overlapping
        assert "完全分离" in separated


class TestFailureSelectors:
    def test_retrieval_failures_only_returns_answerable_misses(self) -> None:
        run = _fake_run(
            answerable_scores=[0.8],
            unanswerable_scores=[0.1],
            answerable_matched=(0, 0, 0),
        )

        failures = retrieval_failures(run)

        assert [result.case_id for result in failures] == ["a-0"]

    def test_false_refusals_uses_the_production_boundary(self) -> None:
        run = _fake_run(
            answerable_scores=[0.34, 0.36],
            unanswerable_scores=[0.10],
        )

        assert [result.case_id for result in false_refusals(run, 0.35)] == ["a-0"]

    def test_false_answers_uses_the_production_boundary(self) -> None:
        run = _fake_run(
            answerable_scores=[0.9],
            unanswerable_scores=[0.34, 0.36],
        )

        assert [result.case_id for result in false_answers(run, 0.35)] == ["u-1"]


class TestCaseRows:
    def test_it_writes_one_row_per_case_in_dataset_order(
        self,
        run: EvaluationRun,
    ) -> None:
        rows = build_case_rows(run)

        assert [row["case_id"] for row in rows] == [
            result.case_id for result in run.results
        ]

    def test_every_row_matches_the_declared_columns(
        self,
        run: EvaluationRun,
    ) -> None:
        for row in build_case_rows(run):
            assert tuple(row) == CASES_CSV_COLUMNS

    def test_unanswerable_rows_leave_retrieval_columns_empty(
        self,
        run: EvaluationRun,
    ) -> None:
        rows = {row["case_id"]: row for row in build_case_rows(run)}
        row = rows["mini-003"]

        assert row["answerable"] == "false"
        assert row["hit_at_1"] == ""
        assert row["hit_at_5"] == ""
        assert row["recall_at_5"] == ""
        assert row["reciprocal_rank"] == ""
        assert row["first_relevant_rank"] == ""

    def test_the_decision_columns_agree_with_the_thresholds(
        self,
        run: EvaluationRun,
    ) -> None:
        for row, result in zip(build_case_rows(run), run.results, strict=True):
            expected_current = (
                result.max_relevance_score is not None
                and result.max_relevance_score >= run.configuration.current_threshold
            )
            assert row["predicted_at_current_threshold"] == (
                "true" if expected_current else "false"
            )
            assert row["decision_correct_current"] == (
                "true" if expected_current == result.answerable else "false"
            )

    def test_write_cases_csv_uses_a_bom_so_excel_reads_chinese(
        self,
        run: EvaluationRun,
        tmp_path: Path,
    ) -> None:
        output = write_cases_csv(run, tmp_path / CASES_FILENAME)

        raw = output.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")

        with output.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["case_id"] for row in rows] == [
            result.case_id for result in run.results
        ]
        assert rows[0]["question"] == run.results[0].question


class TestWriteReports:
    def test_it_writes_all_three_artifacts(
        self,
        run: EvaluationRun,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "generated"

        paths = write_reports(run, output_dir)

        assert [path.name for path in paths] == [
            SUMMARY_FILENAME,
            CASES_FILENAME,
            REPORT_FILENAME,
        ]
        assert all(path.is_file() for path in paths)

    def test_it_creates_a_missing_output_directory(
        self,
        run: EvaluationRun,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "deep" / "nested" / "out"

        write_reports(run, output_dir)

        assert (output_dir / SUMMARY_FILENAME).is_file()

    def test_the_summary_json_round_trips(
        self,
        run: EvaluationRun,
        tmp_path: Path,
    ) -> None:
        path = write_summary_json(run, tmp_path / SUMMARY_FILENAME)

        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["dataset_counts"]["total"] == 5

    def test_writing_twice_produces_identical_bytes(
        self,
        run: EvaluationRun,
        tmp_path: Path,
    ) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"

        write_reports(run, first)
        write_reports(run, second)

        for name in (SUMMARY_FILENAME, CASES_FILENAME, REPORT_FILENAME):
            assert (first / name).read_bytes() == (second / name).read_bytes()


class TestMarkdownReport:
    def test_it_contains_every_required_section(
        self,
        run: EvaluationRun,
    ) -> None:
        report = build_markdown_report(run)

        for heading in (
            "# 离线 RAG 检索与拒答评估报告",
            "## 运行配置",
            "## 语料说明",
            "## 数据集分布",
            "## 检索指标",
            "## 相似度分布摘要",
            "## 阈值扫描（仅 calibration split）",
            "## 推荐阈值及选择规则",
            "## 当前阈值与推荐阈值对比",
            "## 测试集最终结果",
            "## 主要失败案例",
            "## 已知限制",
            "## 后续建议",
        ):
            assert heading in report

    def test_it_states_that_generation_quality_is_out_of_scope(
        self,
        run: EvaluationRun,
    ) -> None:
        report = build_markdown_report(run)

        assert "不评估生成答案的质量" in report
        assert "不会**自动写入生产配置" in report

    def test_it_reports_the_run_configuration_verbatim(
        self,
        run: EvaluationRun,
    ) -> None:
        report = build_markdown_report(run)

        assert "embedding_model     = fake-bigram-hash" in report
        assert "effective_top_k     = 5" in report
        assert "eval/corpus_manifest.json" in report

    def test_it_marks_the_weighting_as_a_business_choice(
        self,
        run: EvaluationRun,
    ) -> None:
        assert "本项目的业务选择" in build_markdown_report(run)

    def test_it_says_so_when_a_failure_category_is_empty(self) -> None:
        run = _fake_run(
            answerable_scores=[0.90],
            unanswerable_scores=[0.10],
        )

        assert "本类别没有失败案例。" in build_markdown_report(run)


def _fake_run(
    answerable_scores: list[float],
    unanswerable_scores: list[float],
    answerable_matched: tuple[int, int, int] = (1, 1, 1),
) -> EvaluationRun:
    """Assemble an EvaluationRun from handcrafted scores, without retrieval."""
    from src.evaluation.metrics import (
        compute_decision_metrics,
        compute_retrieval_metrics,
        group_retrieval_metrics,
        score_distributions,
    )
    from src.evaluation.runner import ChunkConfiguration, RunConfiguration
    from src.evaluation.threshold import generate_thresholds, scan_thresholds
    from tests.evaluation_helpers import make_case

    results = tuple(
        make_result(
            case_id=f"a-{index}",
            split="calibration" if index == 0 else "test",
            answerable=True,
            max_relevance_score=score,
            matched=answerable_matched,
            first_relevant_rank=1 if answerable_matched[0] else None,
        )
        for index, score in enumerate(answerable_scores)
    ) + tuple(
        make_result(
            case_id=f"u-{index}",
            split="calibration" if index == 0 else "test",
            answerable=False,
            max_relevance_score=score,
            category="out_of_scope_far",
        )
        for index, score in enumerate(unanswerable_scores)
    )
    cases = tuple(
        make_case(
            case_id=result.case_id,
            split=result.split,
            question=result.question,
            answerable=result.answerable,
            category=result.category,
            difficulty=result.difficulty,
        )
        for result in results
    )
    calibration = tuple(
        result for result in results if result.split == "calibration"
    )
    test = tuple(result for result in results if result.split == "test")
    if not test:
        test = calibration
    candidates = scan_thresholds(calibration, generate_thresholds(0.2, 0.6, 0.1))

    return EvaluationRun(
        configuration=RunConfiguration(
            manifest_path="eval/corpus_manifest.json",
            dataset_path="eval/dataset.jsonl",
            requested_top_k=5,
            effective_top_k=5,
            threshold_start=0.2,
            threshold_end=0.6,
            threshold_step=0.1,
            current_threshold=0.35,
            false_answer_weight=3.0,
            false_refusal_weight=1.0,
        ),
        chunk_configuration=ChunkConfiguration(
            chunk_size=300,
            chunk_overlap=50,
            document_count=2,
            chunk_count=2,
        ),
        embedding_model="fake-bigram-hash",
        cases=cases,
        results=results,
        retrieval_metrics=compute_retrieval_metrics(results),
        retrieval_metrics_by_category=group_retrieval_metrics(results, "category"),
        retrieval_metrics_by_difficulty=group_retrieval_metrics(
            results,
            "difficulty",
        ),
        retrieval_metrics_by_split=group_retrieval_metrics(results, "split"),
        calibration_candidates=candidates,
        recommended_threshold=0.35,
        calibration_metrics_current=compute_decision_metrics(calibration, 0.35),
        calibration_metrics_recommended=compute_decision_metrics(calibration, 0.35),
        test_metrics_current=compute_decision_metrics(test, 0.35),
        test_metrics_recommended=compute_decision_metrics(test, 0.35),
        score_distribution=score_distributions(results),
    )
