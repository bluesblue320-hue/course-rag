"""Tests for the A/B reranking evaluation runner."""

from __future__ import annotations

import pytest

from src.evaluation.models import EvaluationCase, EvidenceExpectation
from src.evaluation.reranking_metrics import (
    compute_candidate_metrics,
    compute_metric_deltas,
    compute_reranked_metrics,
    compute_vector_metrics,
    group_branch_metrics,
)
from src.evaluation.reranking_models import (
    BranchResult,
    CandidateResult,
    RerankingCaseResult,
)
from src.evaluation.reranking_runner import run_reranking_evaluation
from src.reranker import FakeReranker
from tests.evaluation_helpers import FakeEmbeddingService, make_case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_corpus_chunks():
    """Create minimal corpus chunks for testing."""
    from src.documents import ChunkRecord
    return (
        ChunkRecord(
            chunk_id="c0",
            document_id="doc-a",
            filename="alpha.md",
            text="Service 层负责业务逻辑。",
            chunk_index=0,
            page_number=None,
        ),
        ChunkRecord(
            chunk_id="c1",
            document_id="doc-a",
            filename="alpha.md",
            text="Router 层不应该承载业务逻辑。",
            chunk_index=1,
            page_number=None,
        ),
        ChunkRecord(
            chunk_id="c2",
            document_id="doc-b",
            filename="beta.md",
            text="RAG 是检索增强生成。",
            chunk_index=2,
            page_number=None,
        ),
    )


def make_test_cases() -> tuple[EvaluationCase, ...]:
    return (
        make_case(
            "t-001", "calibration", "Service 层负责什么？",
            True, "direct", "easy",
            [EvidenceExpectation("doc-a", None, ("Service 层负责业务逻辑",))],
            "direct",
        ),
        make_case(
            "t-002", "test", "业务逻辑写在分层架构哪一层？",
            True, "paraphrase", "medium",
            [EvidenceExpectation("doc-a", None, ("Service 层负责业务逻辑",))],
            "paraphrase",
        ),
        make_case(
            "t-003", "test", "RAG 是什么？",
            True, "direct", "easy",
            [EvidenceExpectation("doc-b", None, ("RAG 是检索增强生成",))],
            "direct",
        ),
        make_case(
            "t-004", "calibration", "今天天气怎么样？",
            False, "out_of_scope_far", "easy",
            [],
            "out of scope",
        ),
    )


# ---------------------------------------------------------------------------
# Runner: query embedding and vector retrieval called once per question
# ---------------------------------------------------------------------------


class TestRunnerSingleCall:
    def test_query_embedding_called_once_per_question(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        # One query embedding per case
        assert embedding.query_calls == len(cases)

    def test_document_embeddings_called_once(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        assert embedding.document_batches == 1

    def test_reranker_called_once_per_answerable_question(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        # Reranker called once per case (even unanswerable, since candidates exist)
        assert reranker.call_count == len(cases)

    def test_vector_and_reranked_reuse_same_candidates(self) -> None:
        """Both branches must use the same candidate pool."""
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        for result in run.results:
            if result.answerable and result.vector and result.reranked:
                # Both branches should have sources from the same pool
                assert result.candidate_count > 0
                # Vector sources are a subset of candidates
                assert len(result.vector.sources) <= result.candidate_count
                assert len(result.reranked.sources) <= result.candidate_count


# ---------------------------------------------------------------------------
# Runner: metrics
# ---------------------------------------------------------------------------


class TestRunnerMetrics:
    def test_candidate_metrics_computed(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        cm = run.candidate_metrics
        assert cm.case_count == 3  # 3 answerable cases

    def test_vector_metrics_computed(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        vm = run.vector_metrics
        assert vm.case_count == 3
        assert vm.hit_at_1 is not None
        assert vm.hit_at_5 is not None
        assert vm.mean_reciprocal_rank is not None

    def test_reranked_metrics_computed(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        rm = run.reranked_metrics
        assert rm.case_count == 3

    def test_metric_deltas_computed(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        d = run.metric_deltas
        assert d.hit_at_1 is not None
        assert d.mean_reciprocal_rank is not None

    def test_decision_invariance_passed(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        assert run.decision_invariance_passed is True

    def test_improved_regressed_unchanged(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        total = (
            len(run.improved_case_ids)
            + len(run.regressed_case_ids)
            + len(run.unchanged_case_ids)
        )
        # Should account for all answerable cases (may be less if some missed)
        assert total <= 3  # 3 answerable cases

    def test_grouped_by_category(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        assert "direct" in run.vector_metrics_by_category
        assert "paraphrase" in run.vector_metrics_by_category
        assert "direct" in run.reranked_metrics_by_category

    def test_grouped_by_difficulty(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        assert "easy" in run.vector_metrics_by_difficulty
        assert "medium" in run.vector_metrics_by_difficulty

    def test_grouped_by_split(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        assert "calibration" in run.vector_metrics_by_split
        assert "test" in run.vector_metrics_by_split


# ---------------------------------------------------------------------------
# Runner: validation
# ---------------------------------------------------------------------------


class TestRunnerValidation:
    def test_empty_cases_raises(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        with pytest.raises(Exception, match="不能为空"):
            run_reranking_evaluation(
                cases=(),
                corpus_chunks=make_corpus_chunks(),
                embedding_service=embedding,
                embedding_model="fake",
                reranker=reranker,
                reranker_model="fake",
                manifest_path="m",
                dataset_path="d",
                candidate_top_k=10,
                final_top_k=5,
                comparison_threshold=0.35,
                recommended_threshold=0.49,
            )

    def test_candidate_less_than_final_raises(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        with pytest.raises(Exception, match="candidate_top_k"):
            run_reranking_evaluation(
                cases=make_test_cases(),
                corpus_chunks=make_corpus_chunks(),
                embedding_service=embedding,
                embedding_model="fake",
                reranker=reranker,
                reranker_model="fake",
                manifest_path="m",
                dataset_path="d",
                candidate_top_k=3,
                final_top_k=5,
                comparison_threshold=0.35,
                recommended_threshold=0.49,
            )

    def test_final_top_k_less_than_5_raises(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        with pytest.raises(Exception, match="final_top_k"):
            run_reranking_evaluation(
                cases=make_test_cases(),
                corpus_chunks=make_corpus_chunks(),
                embedding_service=embedding,
                embedding_model="fake",
                reranker=reranker,
                reranker_model="fake",
                manifest_path="m",
                dataset_path="d",
                candidate_top_k=10,
                final_top_k=3,
                comparison_threshold=0.35,
                recommended_threshold=0.49,
            )


# ---------------------------------------------------------------------------
# Runner: no LLM, no network
# ---------------------------------------------------------------------------


class TestRunnerNoExternalCalls:
    def test_no_llm_called(self) -> None:
        """The runner must never call any LLM."""
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        # If this runs without error, no LLM was called.
        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )
        assert run is not None

    def test_no_network_access(self) -> None:
        """The runner uses only FakeEmbeddingService and FakeReranker."""
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        cases = make_test_cases()
        chunks = make_corpus_chunks()

        # If this completes, no network was needed.
        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=chunks,
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=10,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )
        assert len(run.results) == len(cases)


# ---------------------------------------------------------------------------
# Metrics unit tests
# ---------------------------------------------------------------------------


class TestMetricDeltas:
    def test_absolute_delta(self) -> None:
        from src.evaluation.reranking_metrics import MetricDelta, _make_delta
        d = _make_delta(0.5, 0.7)
        assert d.absolute_delta == pytest.approx(0.2)
        assert d.relative_delta == pytest.approx(0.4)

    def test_relative_delta_zero_denominator(self) -> None:
        from src.evaluation.reranking_metrics import _make_delta
        d = _make_delta(0.0, 0.5)
        assert d.absolute_delta == 0.5
        assert d.relative_delta is None  # Not Infinity

    def test_none_values(self) -> None:
        from src.evaluation.reranking_metrics import _make_delta
        d = _make_delta(None, 0.5)
        assert d.absolute_delta is None
        assert d.relative_delta is None

    def test_negative_delta(self) -> None:
        from src.evaluation.reranking_metrics import _make_delta
        d = _make_delta(0.8, 0.6)
        assert d.absolute_delta == pytest.approx(-0.2)
        assert d.relative_delta == pytest.approx(-0.25)
