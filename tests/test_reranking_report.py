"""Tests for the A/B reranking report serialization (Fix 4 + Fix 3)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation.reranking_metrics import (
    BranchMetrics,
    CandidateMetrics,
    MetricDelta,
    MetricDeltas,
)
from src.evaluation.reranking_models import (
    CandidateResult,
    RerankingCaseResult,
    RerankerRunIdentity,
)
from src.evaluation.reranking_report import write_reranking_reports
from src.evaluation.reranking_runner import RerankingRun, RerankingRunConfiguration


def _metrics() -> BranchMetrics:
    return BranchMetrics(
        case_count=1,
        hit_at_1=1.0,
        hit_at_3=1.0,
        hit_at_5=1.0,
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        mean_reciprocal_rank=1.0,
    )


def _candidate_metrics() -> CandidateMetrics:
    return CandidateMetrics(
        case_count=1,
        hit_at_5=1.0,
        hit_at_10=1.0,
        hit_at_15=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        recall_at_15=1.0,
        mean_reciprocal_rank=1.0,
    )


def _deltas() -> MetricDeltas:
    return MetricDeltas(
        hit_at_1=MetricDelta(1.0, 1.0, 0.0, None),
        hit_at_3=MetricDelta(1.0, 1.0, 0.0, None),
        hit_at_5=MetricDelta(1.0, 1.0, 0.0, None),
        recall_at_1=MetricDelta(1.0, 1.0, 0.0, None),
        recall_at_3=MetricDelta(1.0, 1.0, 0.0, None),
        recall_at_5=MetricDelta(1.0, 1.0, 0.0, None),
        mean_reciprocal_rank=MetricDelta(1.0, 1.0, 0.0, None),
    )


def _make_run(
    *,
    candidate_recall_at_5: float | None = 0.0,
    decision_invariance_passed: bool = True,
    inconsistent_ids: tuple[str, ...] = (),
) -> "RerankingRun":  # type: ignore[name-defined]
    candidate = CandidateResult(
        first_relevant_rank=1,
        hit_at_5=True,
        hit_at_10=True,
        hit_at_15=True,
        recall_at_5=candidate_recall_at_5,
        recall_at_10=0.5,
        recall_at_15=0.0,
        reciprocal_rank=1.0,
    )
    result = RerankingCaseResult(
        case_id="t-001",
        split="calibration",
        category="direct",
        difficulty="easy",
        answerable=True,
        question="问题？",
        candidate=candidate,
        max_retrieval_score=0.5,
        vector_max_retrieval_score=0.5,
        reranked_max_retrieval_score=0.5,
        vector_decision_at_comparison=True,
        reranked_decision_at_comparison=True,
        vector_decision_at_recommended=True,
        reranked_decision_at_recommended=True,
        candidate_count=15,
        vector=None,
        reranked=None,
        reranker_applied=False,
        reranker_fallback=False,
        rank_change=None,
        improved=False,
        regressed=False,
        unchanged=False,
    )
    return RerankingRun(
        configuration=RerankingRunConfiguration(
            manifest_path="m",
            dataset_path="d",
            embedding_model="fake-embed",
            reranker_model="fake-reranker",
            candidate_top_k=15,
            final_top_k=5,
        ),
        cases=(),
        results=(result,),
        candidate_metrics=_candidate_metrics(),
        vector_metrics=_metrics(),
        reranked_metrics=_metrics(),
        metric_deltas=_deltas(),
        vector_metrics_by_category={},
        reranked_metrics_by_category={},
        vector_metrics_by_difficulty={},
        reranked_metrics_by_difficulty={},
        vector_metrics_by_split={},
        reranked_metrics_by_split={},
        improved_case_ids=(),
        regressed_case_ids=(),
        unchanged_case_ids=(),
        decision_invariance_passed=decision_invariance_passed,
        decision_invariance_inconsistent_case_ids=inconsistent_ids,
        reranker_applied_count=1,
        reranker_fallback_count=0,
        reranker_fallback_rate=0.0,
        reranker_identity=RerankerRunIdentity(
            backend="fake",
            real_model_run=False,
            model_name="fake-reranker",
        ),
        recommend_enable=False,
        recommendation_reasons=("示例原因",),
        latency_records=(),
    )


class TestCsvZeroValue:
    def test_zero_recall_not_written_as_empty(self, tmp_path: Path) -> None:
        run = _make_run(candidate_recall_at_5=0.0)
        write_reranking_reports(run, tmp_path)
        csv_path = tmp_path / "cases.csv"

        # UTF-8 BOM must be preserved.
        raw = csv_path.read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf"

        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            row = next(reader)

        assert header is not None
        assert "candidate_recall_at_5" in header
        # 0.0 must remain "0.0", not blank.
        assert row["candidate_recall_at_5"] == "0.0"
        # candidate_recall_at_15 was also 0.0
        assert row["candidate_recall_at_15"] == "0.0"
        # recall_at_10 = 0.5
        assert row["candidate_recall_at_10"] == "0.5"

    def test_missing_recall_written_as_empty(self, tmp_path: Path) -> None:
        run = _make_run(candidate_recall_at_5=None)
        write_reranking_reports(run, tmp_path)
        csv_path = tmp_path / "cases.csv"
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            row = next(csv.DictReader(f))
        assert row["candidate_recall_at_5"] == ""


class TestDecisionInvarianceReport:
    def test_failed_check_reports_inconsistent_ids(self, tmp_path: Path) -> None:
        run = _make_run(
            decision_invariance_passed=False,
            inconsistent_ids=("p-001", "p-010"),
        )
        write_reranking_reports(run, tmp_path)

        summary_text = (tmp_path / "summary.json").read_text(encoding="utf-8")
        summary = json.loads(summary_text)
        check = summary["decision_invariance_check"]
        assert check["passed"] is False
        assert check["inconsistent_case_ids"] == ["p-001", "p-010"]
        assert "必然一致" not in summary_text

        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "p-001" in report
        assert "p-010" in report
        assert "必然一致" not in report
        assert "不一致案例" in report


class TestRunProvenanceReport:
    def test_summary_exposes_backend_and_real_model_run(self, tmp_path: Path) -> None:
        write_reranking_reports(_make_run(), tmp_path)
        summary = json.loads(
            (tmp_path / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["reranker_backend"] == "fake"
        assert summary["real_model_run"] is False
        assert summary["reranker_model_revision"] is None
        assert "classification_note" in summary
        assert "unchanged" in summary["classification_note"]

    def test_report_md_lists_backend_and_real_model_run(
        self, tmp_path: Path
    ) -> None:
        write_reranking_reports(_make_run(), tmp_path)
        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Reranker backend" in report
        assert "Real model run" in report
        assert "两个分支都未命中" in report
        assert "未变案例" in report

    def test_cases_csv_has_independent_max_scores(self, tmp_path: Path) -> None:
        write_reranking_reports(_make_run(), tmp_path)
        with open(tmp_path / "cases.csv", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            row = next(reader)
        assert header is not None
        for col in (
            "candidate_max_retrieval_score",
            "vector_max_retrieval_score",
            "reranked_max_retrieval_score",
            "vector_decision_at_comparison",
            "reranked_decision_at_comparison",
        ):
            assert col in header
        assert row["candidate_max_retrieval_score"] == "0.5"
        assert row["vector_max_retrieval_score"] == "0.5"
        assert row["reranked_max_retrieval_score"] == "0.5"
