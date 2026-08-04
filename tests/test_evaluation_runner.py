"""End-to-end runner tests using a fake, fully offline embedding service."""

from pathlib import Path

import pytest

from src.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.evaluation.corpus import load_corpus, load_manifest
from src.evaluation.dataset import load_dataset
from src.evaluation.runner import (
    _build_result,
    _to_sources,
    calibration_split,
    resolve_effective_top_k,
    run_evaluation,
)

# Imported under an alias: pytest would otherwise collect the production
# helper ``test_split`` as a test function and fail on its ``cases`` argument.
from src.evaluation.runner import test_split as select_test_split
from src.exceptions import EvaluationError
from tests.evaluation_helpers import (
    MINI_DATASET_ROWS,
    FakeEmbeddingService,
    make_case,
    make_expectation,
    make_source,
    write_mini_corpus,
    write_mini_dataset,
)


@pytest.fixture()
def mini_run(tmp_path: Path):
    """Build one complete evaluation run over the two-document mini corpus."""
    manifest_path = write_mini_corpus(tmp_path)
    dataset_path = write_mini_dataset(tmp_path)
    documents = load_manifest(manifest_path, base_dir=tmp_path)
    corpus = load_corpus(documents)
    cases = load_dataset(dataset_path)
    embedding = FakeEmbeddingService()

    run = run_evaluation(
        corpus=corpus,
        cases=cases,
        embedding_service=embedding,
        embedding_model="fake-bigram-hash",
        manifest_path="eval/corpus_manifest.json",
        dataset_path="eval/dataset.jsonl",
        top_k=5,
        threshold_start=0.2,
        threshold_end=0.6,
        threshold_step=0.05,
    )
    return run, embedding, corpus, cases


class TestResolveEffectiveTopK:
    @pytest.mark.parametrize(("requested", "expected"), [(1, 5), (3, 5), (5, 5), (8, 8)])
    def test_it_never_drops_below_the_minimum_needed_for_hit_at_5(
        self,
        requested: int,
        expected: int,
    ) -> None:
        assert resolve_effective_top_k(requested) == expected

    @pytest.mark.parametrize("bad", [0, -1])
    def test_it_rejects_a_non_positive_top_k(self, bad: int) -> None:
        with pytest.raises(EvaluationError):
            resolve_effective_top_k(bad)

    @pytest.mark.parametrize("bad", [True, 1.5, "5", None])
    def test_it_rejects_a_non_integer_top_k(self, bad: object) -> None:
        with pytest.raises(EvaluationError):
            resolve_effective_top_k(bad)  # type: ignore[arg-type]


class TestToSources:
    def _raw(self, **overrides: object) -> dict[str, object]:
        raw: dict[str, object] = {
            "rank": 1,
            "score": 0.75,
            "text": "一段正文",
            "chunk_index": 0,
            "document_id": "doc-a",
            "filename": "doc-a.md",
            "page_number": None,
        }
        raw.update(overrides)
        return raw

    def test_it_converts_raw_search_results(self) -> None:
        sources = _to_sources([self._raw(), self._raw(rank=2, score=0.5)])

        assert [source.rank for source in sources] == [1, 2]
        assert sources[0].score == pytest.approx(0.75)
        assert sources[0].page_number is None

    def test_it_keeps_an_integer_page_number(self) -> None:
        sources = _to_sources([self._raw(page_number=3)])

        assert sources[0].page_number == 3

    def test_it_returns_an_empty_tuple_for_no_results(self) -> None:
        assert _to_sources([]) == ()

    @pytest.mark.parametrize("bad_score", ["0.5", None, True])
    def test_it_rejects_an_invalid_score(self, bad_score: object) -> None:
        with pytest.raises(EvaluationError):
            _to_sources([self._raw(score=bad_score)])

    def test_it_rejects_a_non_integer_page_number(self) -> None:
        with pytest.raises(EvaluationError):
            _to_sources([self._raw(page_number="3")])


class TestBuildResult:
    def test_it_records_retrieval_fields_for_an_answerable_case(self) -> None:
        case = make_case(
            answerable=True,
            expected_evidence=(make_expectation(required_terms=("结算服务",)),),
        )
        sources = (
            make_source(1, 0.9, text="与问题无关的内容"),
            make_source(2, 0.7, text="订单结算由结算服务负责"),
        )

        result = _build_result(case, sources)

        assert result.max_relevance_score == pytest.approx(0.9)
        assert result.first_relevant_rank == 2
        assert result.hit_at_1 is False
        assert result.hit_at_3 is True
        assert result.hit_at_5 is True
        assert result.reciprocal_rank == pytest.approx(0.5)
        assert result.total_evidence_count == 1

    def test_it_scores_a_complete_miss_as_zero_without_crashing(self) -> None:
        case = make_case(
            answerable=True,
            expected_evidence=(make_expectation(required_terms=("永不出现的短语",)),),
        )
        sources = (make_source(1, 0.4, text="别的内容"),)

        result = _build_result(case, sources)

        assert result.first_relevant_rank is None
        assert result.hit_at_5 is False
        assert result.reciprocal_rank == pytest.approx(0.0)

    def test_an_unanswerable_case_carries_no_retrieval_metric(self) -> None:
        case = make_case(answerable=False, category="out_of_scope_far")
        sources = (make_source(1, 0.61),)

        result = _build_result(case, sources)

        assert result.max_relevance_score == pytest.approx(0.61)
        assert result.hit_at_1 is None
        assert result.hit_at_3 is None
        assert result.hit_at_5 is None
        assert result.reciprocal_rank is None
        assert result.first_relevant_rank is None
        assert result.total_evidence_count == 0
        assert result.recall_at(5) is None

    def test_it_reports_no_score_when_nothing_was_retrieved(self) -> None:
        result = _build_result(make_case(answerable=False), ())

        assert result.max_relevance_score is None
        assert result.sources == ()


class TestRunEvaluation:
    def test_it_produces_one_result_per_case(self, mini_run) -> None:
        run, _embedding, _corpus, cases = mini_run

        assert len(run.results) == len(cases)
        assert [result.case_id for result in run.results] == [
            case.id for case in cases
        ]

    def test_it_embeds_documents_once_and_each_question_once(
        self,
        mini_run,
    ) -> None:
        run, embedding, _corpus, cases = mini_run

        assert embedding.document_batches == 1
        assert embedding.query_calls == len(cases)
        # The threshold sweep must reuse recorded scores, not re-embed.
        assert len(run.calibration_candidates) == 9

    def test_it_retrieves_the_expected_evidence_for_the_mini_corpus(
        self,
        mini_run,
    ) -> None:
        run, _embedding, _corpus, _cases = mini_run

        assert run.retrieval_metrics.case_count == 3
        assert run.retrieval_metrics.hit_at_5 == pytest.approx(1.0)
        assert run.retrieval_metrics.mean_reciprocal_rank is not None

    def test_it_records_the_production_chunk_configuration(
        self,
        mini_run,
    ) -> None:
        run, _embedding, corpus, _cases = mini_run

        assert run.chunk_configuration.chunk_size == DEFAULT_CHUNK_SIZE
        assert run.chunk_configuration.chunk_overlap == DEFAULT_CHUNK_OVERLAP
        assert run.chunk_configuration.document_count == 2
        assert run.chunk_configuration.chunk_count == len(corpus.chunks)

    def test_it_lifts_top_k_to_at_least_five(self, tmp_path: Path) -> None:
        manifest_path = write_mini_corpus(tmp_path)
        dataset_path = write_mini_dataset(tmp_path)
        corpus = load_corpus(load_manifest(manifest_path, base_dir=tmp_path))

        run = run_evaluation(
            corpus=corpus,
            cases=load_dataset(dataset_path),
            embedding_service=FakeEmbeddingService(),
            embedding_model="fake",
            manifest_path="eval/corpus_manifest.json",
            dataset_path="eval/dataset.jsonl",
            top_k=2,
            threshold_start=0.2,
            threshold_end=0.6,
            threshold_step=0.1,
        )

        assert run.configuration.requested_top_k == 2
        assert run.configuration.effective_top_k == 5

    def test_the_recommended_threshold_comes_from_calibration_only(
        self,
        mini_run,
    ) -> None:
        run, _embedding, _corpus, _cases = mini_run

        calibration_ids = {
            result.case_id
            for result in run.results
            if result.split == "calibration"
        }
        assert calibration_ids == {"mini-001", "mini-002", "mini-003"}
        assert all(
            candidate.metrics.case_count == len(calibration_ids)
            for candidate in run.calibration_candidates
        )
        assert run.test_metrics_recommended.case_count == 2

    def test_changing_only_test_cases_does_not_move_the_threshold(
        self,
        tmp_path: Path,
    ) -> None:
        """A leak-free selection must ignore everything in the test split."""

        def build(rows: tuple[dict[str, object], ...]) -> float:
            root = tmp_path / f"variant-{len(rows)}"
            root.mkdir(parents=True, exist_ok=True)
            manifest_path = write_mini_corpus(root)
            dataset_path = write_mini_dataset(root, rows)
            corpus = load_corpus(load_manifest(manifest_path, base_dir=root))
            run = run_evaluation(
                corpus=corpus,
                cases=load_dataset(dataset_path),
                embedding_service=FakeEmbeddingService(),
                embedding_model="fake",
                manifest_path="m",
                dataset_path="d",
                top_k=5,
                threshold_start=0.2,
                threshold_end=0.6,
                threshold_step=0.05,
            )
            return run.recommended_threshold

        baseline = build(MINI_DATASET_ROWS)
        # Drop one held-out test question; calibration is untouched.
        trimmed = tuple(
            row for row in MINI_DATASET_ROWS if row["id"] != "mini-005"
        )

        assert build(trimmed) == pytest.approx(baseline)

    def test_it_is_deterministic_across_runs(self, tmp_path: Path) -> None:
        manifest_path = write_mini_corpus(tmp_path)
        dataset_path = write_mini_dataset(tmp_path)
        corpus = load_corpus(load_manifest(manifest_path, base_dir=tmp_path))
        cases = load_dataset(dataset_path)

        def once() -> tuple:
            run = run_evaluation(
                corpus=corpus,
                cases=cases,
                embedding_service=FakeEmbeddingService(),
                embedding_model="fake",
                manifest_path="m",
                dataset_path="d",
                top_k=5,
                threshold_start=0.2,
                threshold_end=0.6,
                threshold_step=0.05,
            )
            return (
                run.recommended_threshold,
                tuple(result.max_relevance_score for result in run.results),
            )

        assert once() == once()

    def test_it_rejects_an_empty_dataset(self, tmp_path: Path) -> None:
        manifest_path = write_mini_corpus(tmp_path)
        corpus = load_corpus(load_manifest(manifest_path, base_dir=tmp_path))

        with pytest.raises(EvaluationError):
            run_evaluation(
                corpus=corpus,
                cases=(),
                embedding_service=FakeEmbeddingService(),
                embedding_model="fake",
                manifest_path="m",
                dataset_path="d",
                top_k=5,
                threshold_start=0.2,
                threshold_end=0.6,
                threshold_step=0.05,
            )

    def test_it_rejects_a_dataset_without_a_test_split(
        self,
        tmp_path: Path,
    ) -> None:
        manifest_path = write_mini_corpus(tmp_path)
        rows = tuple(
            row for row in MINI_DATASET_ROWS if row["split"] == "calibration"
        )
        dataset_path = write_mini_dataset(tmp_path, rows)
        corpus = load_corpus(load_manifest(manifest_path, base_dir=tmp_path))

        with pytest.raises(EvaluationError, match="test split"):
            run_evaluation(
                corpus=corpus,
                cases=load_dataset(dataset_path),
                embedding_service=FakeEmbeddingService(),
                embedding_model="fake",
                manifest_path="m",
                dataset_path="d",
                top_k=5,
                threshold_start=0.2,
                threshold_end=0.6,
                threshold_step=0.05,
            )

    def test_it_rejects_evidence_pointing_at_a_missing_document(
        self,
        tmp_path: Path,
    ) -> None:
        manifest_path = write_mini_corpus(tmp_path)
        corpus = load_corpus(load_manifest(manifest_path, base_dir=tmp_path))
        cases = (
            make_case(
                case_id="ghost",
                expected_evidence=(
                    make_expectation(document_id="mini-gamma"),
                ),
            ),
        )

        with pytest.raises(EvaluationError):
            run_evaluation(
                corpus=corpus,
                cases=cases,
                embedding_service=FakeEmbeddingService(),
                embedding_model="fake",
                manifest_path="m",
                dataset_path="d",
                top_k=5,
                threshold_start=0.2,
                threshold_end=0.6,
                threshold_step=0.05,
            )


class TestSplitHelpers:
    def test_it_separates_calibration_from_test(self, mini_run) -> None:
        _run, _embedding, _corpus, cases = mini_run

        assert [case.id for case in calibration_split(cases)] == [
            "mini-001",
            "mini-002",
            "mini-003",
        ]
        assert [case.id for case in select_test_split(cases)] == [
            "mini-004",
            "mini-005",
        ]
