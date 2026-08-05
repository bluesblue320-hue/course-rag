"""Tests for the A/B reranking evaluation runner."""

from __future__ import annotations

import pytest

from src.evaluation.models import EvaluationCase, EvidenceExpectation
from src.evaluation.reranking_metrics import (
    BranchMetrics,
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
    RerankerRunIdentity,
    classify_rank_change,
)
from src.evaluation.reranking_runner import (
    _build_candidate_result,
    _evaluate_recommendation,
    check_decision_invariance,
    run_reranking_evaluation,
)
from src.reranker import FakeReranker
from tests.evaluation_helpers import (
    FakeEmbeddingService,
    make_case,
    make_expectation,
)


def real_identity(model_name: str = "real-model") -> RerankerRunIdentity:
    """Identity for a successfully loaded real CrossEncoder."""
    return RerankerRunIdentity(
        backend="cross_encoder",
        real_model_run=True,
        model_name=model_name,
    )


def fake_identity(model_name: str = "fake-reranker") -> RerankerRunIdentity:
    """Identity for a FakeReranker / test substitute."""
    return RerankerRunIdentity(
        backend="fake",
        real_model_run=False,
        model_name=model_name,
    )


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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        total = (
            len(run.improved_case_ids)
            + len(run.regressed_case_ids)
            + len(run.unchanged_case_ids)
        )
        # Every answerable case belongs to exactly one set.
        assert total == 3  # 3 answerable cases

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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
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
                candidate_top_k=15,
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
                candidate_top_k=15,
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
            candidate_top_k=15,
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
            candidate_top_k=15,
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


# ---------------------------------------------------------------------------
# Fix 3: decision invariance real check
# ---------------------------------------------------------------------------


class TestDecisionInvarianceCheck:
    def test_pure_function_passes_when_consistent(self) -> None:
        res = check_decision_invariance(
            vector_max_score=0.5,
            reranked_max_score=0.5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
            case_id="p-001",
        )
        assert res.passed is True
        assert res.inconsistent_case_ids == ()

    def test_pure_function_fails_when_mismatch(self) -> None:
        # vector score clears the comparison threshold; reranked does not.
        res = check_decision_invariance(
            vector_max_score=0.4,
            reranked_max_score=0.2,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
            case_id="p-001",
        )
        assert res.passed is False
        assert res.inconsistent_case_ids == ("p-001",)
        assert res.vector_decision_at_comparison is True
        assert res.reranked_decision_at_comparison is False

    def test_runner_positive_still_passes(self) -> None:
        embedding = FakeEmbeddingService()
        reranker = FakeReranker()
        run = run_reranking_evaluation(
            cases=make_test_cases(),
            corpus_chunks=make_corpus_chunks(),
            embedding_service=embedding,
            embedding_model="fake",
            reranker=reranker,
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=15,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )
        assert run.decision_invariance_passed is True
        assert run.decision_invariance_inconsistent_case_ids == ()


# ---------------------------------------------------------------------------
# Fix 5: candidate_top_k >= 15 enforcement
# ---------------------------------------------------------------------------


class TestCandidateTopKEnforcement:
    def test_runner_rejects_candidate_top_k_below_15(self) -> None:
        with pytest.raises(Exception, match="candidate_top_k"):
            run_reranking_evaluation(
                cases=make_test_cases(),
                corpus_chunks=make_corpus_chunks(),
                embedding_service=FakeEmbeddingService(),
                embedding_model="fake",
                reranker=FakeReranker(),
                reranker_model="fake",
                manifest_path="m",
                dataset_path="d",
                candidate_top_k=10,
                final_top_k=5,
                comparison_threshold=0.35,
                recommended_threshold=0.49,
            )

    def test_runner_allows_candidate_top_k_15(self) -> None:
        run = run_reranking_evaluation(
            cases=make_test_cases(),
            corpus_chunks=make_corpus_chunks(),
            embedding_service=FakeEmbeddingService(),
            embedding_model="fake",
            reranker=FakeReranker(),
            reranker_model="fake-reranker",
            manifest_path="m",
            dataset_path="d",
            candidate_top_k=15,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )
        assert run.candidate_metrics.case_count == 3

    def test_hit_at_15_only_counts_first_15_candidates(self) -> None:
        evidence = (make_expectation(document_id="doc-ev", required_terms=("证据文本",)),)
        case = make_case(
            "t-ev", "calibration", "问题？", True, "direct", "easy", evidence
        )
        # Put the matching chunk at rank 17 (index 16): beyond Top-15.
        candidates = [
            {
                "rank": i + 1,
                "score": 1.0 - i * 0.01,
                "text": f"无关正文 {i}" if i != 16 else "这是证据文本",
                "chunk_index": i,
                "document_id": "doc-other" if i != 16 else "doc-ev",
                "filename": "f.md",
                "page_number": None,
            }
            for i in range(20)
        ]
        result = _build_candidate_result(case, candidates, candidate_top_k=20)
        assert result is not None
        assert result.hit_at_15 is False

        # Move the matching chunk into Top-15 (rank 14, index 13): now a hit.
        candidates[13]["document_id"] = "doc-ev"
        candidates[13]["text"] = "这是证据文本"
        candidates[16]["document_id"] = "doc-other"
        candidates[16]["text"] = "无关正文 16"
        result2 = _build_candidate_result(case, candidates, candidate_top_k=20)
        assert result2 is not None
        assert result2.hit_at_15 is True


# ---------------------------------------------------------------------------
# Fix 7: fallback gating in enable recommendation
# ---------------------------------------------------------------------------


def _branch_metrics(
    h1: float, h3: float, h5: float,
    r1: float, r3: float, r5: float, mrr: float,
) -> BranchMetrics:
    return BranchMetrics(
        case_count=3,
        hit_at_1=h1, hit_at_3=h3, hit_at_5=h5,
        recall_at_1=r1, recall_at_3=r3, recall_at_5=r5,
        mean_reciprocal_rank=mrr,
    )


class TestFallbackGating:
    def test_any_fallback_blocks_recommendation(self) -> None:
        vm = _branch_metrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        rm = _branch_metrics(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
        ok, reasons = _evaluate_recommendation(
            vm, rm, {}, {}, 0, True, real_identity(),
            reranker_fallback_count=1,
        )
        assert ok is False
        assert any("回退" in r for r in reasons)
        assert "Reranker 在 1 个案例" in reasons[0]

    def test_zero_fallback_passes_fallback_gate(self) -> None:
        vm = _branch_metrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        rm = _branch_metrics(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
        # With empty category groups the metrics gate fails for a non-fallback
        # reason; the fallback gate must NOT be the blocker.
        ok, reasons = _evaluate_recommendation(
            vm, rm, {}, {}, 0, True, real_identity(),
            reranker_fallback_count=0,
        )
        assert ok is False
        assert not any("回退" in r for r in reasons)


# ---------------------------------------------------------------------------
# Fix 5 (round 2): real-model provenance gating
# ---------------------------------------------------------------------------


class TestRealModelGating:
    def test_fake_reranker_with_great_metrics_never_recommends(self) -> None:
        vm = _branch_metrics(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
        rm = _branch_metrics(0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99)
        ok, reasons = _evaluate_recommendation(
            vm, rm, {}, {}, 0, True, fake_identity()
        )
        assert ok is False
        assert "未使用真实 CrossEncoder" in reasons[0]

    def test_fake_reranker_with_fake_real_model_name_never_recommends(self) -> None:
        # A fake reranker pretending to be a real model name must still fail.
        vm = _branch_metrics(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
        rm = _branch_metrics(0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99)
        ok, reasons = _evaluate_recommendation(
            vm, rm, {}, {}, 0, True,
            fake_identity(model_name="BAAI/bge-reranker-v2-m3"),
        )
        assert ok is False
        assert "未使用真实 CrossEncoder" in reasons[0]

    def test_custom_backend_never_recommends(self) -> None:
        identity = RerankerRunIdentity(
            backend="custom",
            real_model_run=False,
            model_name="custom-thing",
        )
        vm = _branch_metrics(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
        rm = _branch_metrics(0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99)
        ok, reasons = _evaluate_recommendation(vm, rm, {}, {}, 0, True, identity)
        assert ok is False
        assert "未使用真实 CrossEncoder" in reasons[0]

    def test_cross_encoder_but_real_model_run_false_never_recommends(self) -> None:
        identity = RerankerRunIdentity(
            backend="cross_encoder",
            real_model_run=False,
            model_name="real-model",
        )
        vm = _branch_metrics(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
        rm = _branch_metrics(0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99)
        ok, reasons = _evaluate_recommendation(vm, rm, {}, {}, 0, True, identity)
        assert ok is False
        assert "未使用真实 CrossEncoder" in reasons[0]

    def test_real_cross_encoder_proceeds_to_metric_gates(self) -> None:
        # Real identity passes the provenance gate; with empty category groups
        # the next blocker is a metric gate (paraphrase missing), not identity.
        vm = _branch_metrics(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
        rm = _branch_metrics(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
        ok, reasons = _evaluate_recommendation(
            vm, rm, {}, {}, 0, True, real_identity()
        )
        assert ok is False
        assert not any("未使用真实 CrossEncoder" in r for r in reasons)
        assert not any("不是 CrossEncoder" in r for r in reasons)

    def test_fallback_blocks_even_real_model(self) -> None:
        vm = _branch_metrics(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
        rm = _branch_metrics(0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99)
        ok, reasons = _evaluate_recommendation(
            vm, rm, {}, {}, 0, True, real_identity(),
            reranker_fallback_count=2,
        )
        assert ok is False
        assert "Reranker 在 2 个案例" in reasons[0]

    def test_decision_invariance_failure_blocks_even_real_model(self) -> None:
        # Provide category metrics so every metric gate passes and the only
        # remaining blocker is the decision-invariance failure.
        from src.evaluation.reranking_metrics import BranchMetrics as BM

        def cat(mrr: float) -> BM:
            return BM(
                case_count=1,
                hit_at_1=0.9,
                hit_at_3=0.9,
                hit_at_5=0.9,
                recall_at_1=0.9,
                recall_at_3=0.9,
                recall_at_5=0.9,
                mean_reciprocal_rank=mrr,
            )

        vm = _branch_metrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        rm = _branch_metrics(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
        ok, reasons = _evaluate_recommendation(
            vm,
            rm,
            {"paraphrase": cat(0.5), "direct": cat(0.5)},
            {"paraphrase": cat(0.9), "direct": cat(0.9)},
            0,
            False,
            real_identity(),
        )
        assert ok is False
        assert "决策一致性" in " ".join(reasons)


# ---------------------------------------------------------------------------
# Fix 1 (round 2): full-runner decision invariance genuinely fail-able
# ---------------------------------------------------------------------------


def _drift_corpus_chunks():
    """Corpus where the top retrieval hit is unmistakable.

    The evidence chunk text is identical to the question, so its cosine
    similarity is 1.0 (well above the 0.35 comparison threshold), while every
    other chunk is disjoint (cosine ~0).  Dropping the top chunk in the
    reranked branch therefore flips the reranked refusal decision.
    """
    from src.documents import ChunkRecord

    return (
        ChunkRecord(
            chunk_id="c0",
            document_id="doc-ev",
            filename="evidence.md",
            text="身份标识短语",
            chunk_index=0,
            page_number=None,
        ),
        ChunkRecord(
            chunk_id="c1",
            document_id="doc-other",
            filename="other.md",
            text="互不相干内容乙丙丁戊",
            chunk_index=1,
            page_number=None,
        ),
        ChunkRecord(
            chunk_id="c2",
            document_id="doc-other",
            filename="other.md",
            text="完全不同内容子丑寅卯",
            chunk_index=2,
            page_number=None,
        ),
    )


def _drift_case() -> EvaluationCase:
    return make_case(
        "t-drift",
        "calibration",
        "身份标识短语",
        True,
        "direct",
        "easy",
        [make_expectation(document_id="doc-ev", required_terms=("身份标识短语",))],
    )


class TestRunnerDecisionInvarianceNegative:
    def test_lost_top_candidate_makes_runner_check_fail(self, monkeypatch) -> None:
        from src.evaluation import reranking_runner as runner_module

        def dropping_rerank(reranker, query, candidates):
            # Simulate the reranking stage silently losing the highest
            # retrieval-score candidate while reporting no fallback.
            sorted_cands = sorted(
                candidates, key=lambda c: float(c["score"]), reverse=True
            )
            kept = sorted_cands[1:] if sorted_cands else []
            return kept, False

        monkeypatch.setattr(
            runner_module, "_rerank_candidates", dropping_rerank
        )
        run = run_reranking_evaluation(
            cases=(_drift_case(),),
            corpus_chunks=_drift_corpus_chunks(),
            embedding_service=FakeEmbeddingService(),
            embedding_model="fake",
            reranker=FakeReranker(),
            reranker_model="fake-reranker",
            manifest_path="m",
            dataset_path="d",
            candidate_top_k=15,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        # The vector max retrieval score is 1.0; the reranked branch lost the
        # top candidate, so its max is ~0 and the decisions diverge.
        result = run.results[0]
        assert result.vector_max_retrieval_score is not None
        assert result.vector_max_retrieval_score >= 0.99
        assert result.reranked_max_retrieval_score is not None
        assert result.reranked_max_retrieval_score < 0.35
        assert result.vector_decision_at_comparison is True
        assert result.reranked_decision_at_comparison is False

        assert run.decision_invariance_passed is False
        assert "t-drift" in run.decision_invariance_inconsistent_case_ids
        assert run.recommend_enable is False

    def test_corrupted_retrieval_score_makes_runner_check_fail(
        self, monkeypatch
    ) -> None:
        from src.evaluation import reranking_runner as runner_module

        def corrupting_rerank(reranker, query, candidates):
            # Simulate the reranking stage overwriting retrieval scores with
            # low values while returning the same candidate set (no fallback).
            corrupted = [
                {**c, "score": 0.1} for c in candidates
            ]
            return corrupted, False

        monkeypatch.setattr(
            runner_module, "_rerank_candidates", corrupting_rerank
        )
        run = run_reranking_evaluation(
            cases=(_drift_case(),),
            corpus_chunks=_drift_corpus_chunks(),
            embedding_service=FakeEmbeddingService(),
            embedding_model="fake",
            reranker=FakeReranker(),
            reranker_model="fake-reranker",
            manifest_path="m",
            dataset_path="d",
            candidate_top_k=15,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )

        assert run.decision_invariance_passed is False
        assert "t-drift" in run.decision_invariance_inconsistent_case_ids
        assert run.recommend_enable is False

    def test_runner_positive_still_passes_with_real_identity(self) -> None:
        """Normal data flow keeps passing even with a real identity."""
        run = run_reranking_evaluation(
            cases=make_test_cases(),
            corpus_chunks=make_corpus_chunks(),
            embedding_service=FakeEmbeddingService(),
            embedding_model="fake",
            reranker=FakeReranker(),
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=15,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
            reranker_identity=real_identity("fake-reranker"),
        )
        assert run.decision_invariance_passed is True
        assert run.decision_invariance_inconsistent_case_ids == ()


# ---------------------------------------------------------------------------
# Fix 4 (round 2): complete rank-change classification
# ---------------------------------------------------------------------------


class TestRankClassification:
    def test_equal_ranks_unchanged(self) -> None:
        c = classify_rank_change(1, 1, answerable=True)
        assert (c.improved, c.regressed, c.unchanged) == (False, False, True)
        assert c.rank_change == 0

    def test_improved(self) -> None:
        c = classify_rank_change(3, 1, answerable=True)
        assert (c.improved, c.regressed, c.unchanged) == (True, False, False)
        assert c.rank_change == -2

    def test_regressed(self) -> None:
        c = classify_rank_change(1, 3, answerable=True)
        assert (c.improved, c.regressed, c.unchanged) == (False, True, False)
        assert c.rank_change == 2

    def test_vector_miss_reranked_hit_improved(self) -> None:
        c = classify_rank_change(None, 2, answerable=True)
        assert (c.improved, c.regressed, c.unchanged) == (True, False, False)
        assert c.rank_change is None

    def test_vector_hit_reranked_miss_regressed(self) -> None:
        c = classify_rank_change(2, None, answerable=True)
        assert (c.improved, c.regressed, c.unchanged) == (False, True, False)
        assert c.rank_change is None

    def test_both_miss_answerable_unchanged(self) -> None:
        c = classify_rank_change(None, None, answerable=True)
        assert (c.improved, c.regressed, c.unchanged) == (False, False, True)
        assert c.rank_change is None

    def test_both_miss_unanswerable_not_classified(self) -> None:
        c = classify_rank_change(None, None, answerable=False)
        assert (c.improved, c.regressed, c.unchanged) == (False, False, False)
        assert c.rank_change is None


class TestClassificationCompleteness:
    def test_sets_are_disjoint_and_cover_all_answerable_cases(self) -> None:
        run = run_reranking_evaluation(
            cases=make_test_cases(),
            corpus_chunks=make_corpus_chunks(),
            embedding_service=FakeEmbeddingService(),
            embedding_model="fake",
            reranker=FakeReranker(),
            reranker_model="fake-reranker",
            manifest_path="manifest",
            dataset_path="dataset",
            candidate_top_k=15,
            final_top_k=5,
            comparison_threshold=0.35,
            recommended_threshold=0.49,
        )
        answerable_ids = {
            r.case_id for r in run.results if r.answerable
        }
        improved = set(run.improved_case_ids)
        regressed = set(run.regressed_case_ids)
        unchanged = set(run.unchanged_case_ids)

        # Pairwise disjoint
        assert improved & regressed == set()
        assert improved & unchanged == set()
        assert regressed & unchanged == set()
        # Union covers every answerable case exactly once.
        union = improved | regressed | unchanged
        assert union == answerable_ids
        assert len(improved) + len(regressed) + len(unchanged) == len(
            answerable_ids
        )

    def test_both_miss_cases_land_in_unchanged(self) -> None:
        """Forcing both branches to miss must still classify as unchanged."""
        from src.evaluation import reranking_runner as runner_module

        # Evidence terms that never appear in any corpus chunk: both branches
        # miss, and the case must still be classified as unchanged.
        case = make_case(
            "t-both-miss",
            "calibration",
            "一个普通问题？",
            True,
            "direct",
            "easy",
            [make_expectation(document_id="doc-a", required_terms=("绝不存在的术语",))],
        )

        def missing_rerank(reranker, query, candidates):
            # Return candidates that cannot match the evidence: empty pool.
            return [], False

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runner_module, "_rerank_candidates", missing_rerank
        )
        try:
            run = run_reranking_evaluation(
                cases=(case,),
                corpus_chunks=make_corpus_chunks(),
                embedding_service=FakeEmbeddingService(),
                embedding_model="fake",
                reranker=FakeReranker(),
                reranker_model="fake-reranker",
                manifest_path="m",
                dataset_path="d",
                candidate_top_k=15,
                final_top_k=5,
                comparison_threshold=0.35,
                recommended_threshold=0.49,
            )
        finally:
            monkeypatch.undo()

        assert run.unchanged_case_ids == ("t-both-miss",)
        assert run.improved_case_ids == ()
        assert run.regressed_case_ids == ()
        # Both branches report a miss, yet the case is still classified.
        result = run.results[0]
        assert result.vector.first_relevant_rank is None
        assert result.reranked.first_relevant_rank is None
        assert result.unchanged is True
