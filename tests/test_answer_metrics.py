"""Tests for answer-quality metric computations (pure functions)."""

import math

import pytest

from src.evaluation.answer_metrics import (
    compute_aggregate_metrics,
    compute_case_metrics,
    fact_covered,
    fact_grounded_by_citation,
    forbidden_phrase_hits,
    group_case_metrics,
    normalize_answer_text,
    phrase_appears,
    summarize_latency,
)
from src.evaluation.answer_citations import parse_citations
from src.evaluation.answer_models import (
    AnswerAnnotation,
    AnswerResponse,
    AnswerSource,
    RequiredFact,
)
from src.evaluation.models import EvidenceExpectation
from src.exceptions import AnswerEvaluationError
from src.rag_service import INSUFFICIENT_CONTEXT_ANSWER
from tests.evaluation_helpers import make_case, make_expectation

FACT_ONE = RequiredFact(
    fact_id="fact-one",
    accepted_phrases=("事实陈述",),
    supporting_evidence_indexes=(1,),
)


def _annotation(
    facts: tuple[RequiredFact, ...] = (FACT_ONE,),
    forbidden: tuple[str, ...] = (),
    reference_answer: str = "事实陈述。",
) -> AnswerAnnotation:
    return AnswerAnnotation(
        case_id="c-001",
        reference_answer=reference_answer,
        required_facts=facts,
        forbidden_phrases=forbidden,
        notes="",
    )


def _response(
    answer: str = "事实陈述。",
    status: str = "answered",
    sources: tuple[AnswerSource, ...] | None = None,
    max_score: float | None = 0.6,
) -> AnswerResponse:
    if sources is None:
        sources = (
            AnswerSource(
                rank=1,
                score=0.6,
                text="事实陈述。",
                chunk_index=0,
                document_id="doc-a",
                filename="doc-a.md",
                page_number=None,
            ),
        )
    return AnswerResponse(
        case_id="c-001",
        answer_status=status,  # type: ignore[arg-type]
        answer=answer,
        sources=sources,
        max_relevance_score=max_score,
        relevance_threshold=0.35,
        retrieval_elapsed_ms=2.0,
        generation_elapsed_ms=100.0,
        total_elapsed_ms=102.0,
        reranker_applied=False,
        reranker_fallback=False,
    )


class TestNormalizeAnswerText:
    def test_nfkc_folding(self) -> None:
        assert normalize_answer_text("ＦａｓｔＡＰＩ") == "fastapi"

    def test_casefold(self) -> None:
        assert normalize_answer_text("Hello World") == "hello world"

    def test_whitespace_collapse(self) -> None:
        assert normalize_answer_text("  一\n  二  ") == "一二"

    def test_full_width_punctuation(self) -> None:
        assert normalize_answer_text("甲，乙。丙？") == "甲,乙.丙?"

    def test_chinese_stays_intact(self) -> None:
        assert normalize_answer_text("Router 层不应该承载业务逻辑") == (
            "router层不应该承载业务逻辑"
        )

    def test_cjk_adjacent_whitespace_is_removed(self) -> None:
        assert normalize_answer_text("业务 逻辑") == "业务逻辑"
        assert normalize_answer_text("业务逻辑 ") == "业务逻辑"
        assert normalize_answer_text(" 业务逻辑") == "业务逻辑"


class TestPhraseAppears:
    def test_substring_match(self) -> None:
        assert phrase_appears("业务逻辑", "Service 层负责业务逻辑。")

    def test_whitespace_insensitive(self) -> None:
        assert phrase_appears("业务 逻辑", "Service 层负责业务逻辑。")


class TestFactCovered:
    def test_any_accepted_phrase_matches(self) -> None:
        fact = RequiredFact(
            fact_id="f",
            accepted_phrases=("第一个表述", "第二个表述"),
            supporting_evidence_indexes=(1,),
        )
        covered, matched = fact_covered(fact, "答案是第二个表述。")
        assert covered
        assert matched == "第二个表述"

    def test_no_accepted_phrase_matches(self) -> None:
        fact = RequiredFact(
            fact_id="f",
            accepted_phrases=("完全不相关表述",),
            supporting_evidence_indexes=(1,),
        )
        covered, matched = fact_covered(fact, "答案是另一个内容。")
        assert not covered
        assert matched is None

    def test_nfkc_difference_does_not_block_match(self) -> None:
        fact = RequiredFact(
            fact_id="f",
            accepted_phrases=("ＦａｓｔＡＰＩ",),
            supporting_evidence_indexes=(1,),
        )
        covered, _ = fact_covered(fact, "答案是 FastAPI。")
        assert covered


class TestForbiddenPhraseHits:
    def test_hit_is_detected(self) -> None:
        hits = forbidden_phrase_hits(("错误说法",), "答案包含错误说法。")
        assert hits == ("错误说法",)

    def test_no_hit(self) -> None:
        hits = forbidden_phrase_hits(("错误说法",), "答案完全正确。")
        assert hits == ()


class TestCaseMetrics:
    def test_covered_fact_with_supporting_citation_is_grounded(self) -> None:
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                make_expectation(document_id="doc-a", required_terms=("事实陈述",)),
            ),
        )
        response = _response("事实陈述。[来源1]")
        metrics = compute_case_metrics(
            case,
            _annotation(),
            response,
            parse_citations(response.answer),
        )
        assert metrics.covered_fact_count == 1
        assert metrics.grounded_fact_count == 1
        assert metrics.facts[0].grounded_by_citation
        assert metrics.facts[0].supporting_citation_numbers == (1,)
        assert metrics.fact_coverage == 1.0
        assert metrics.grounded_fact_coverage == 1.0
        assert metrics.strict_pass

    def test_covered_fact_without_citation_is_not_grounded(self) -> None:
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                make_expectation(document_id="doc-a", required_terms=("事实陈述",)),
            ),
        )
        response = _response("事实陈述。")
        metrics = compute_case_metrics(
            case,
            _annotation(),
            response,
            parse_citations(response.answer),
        )
        assert metrics.covered_fact_count == 1
        assert metrics.grounded_fact_count == 0
        assert not metrics.facts[0].grounded_by_citation
        assert not metrics.strict_pass

    def test_covered_fact_but_wrong_source_is_not_grounded(self) -> None:
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                make_expectation(document_id="doc-a", required_terms=("事实陈述",)),
            ),
        )
        response = _response(
            "事实陈述。[来源1]",
            sources=(
                AnswerSource(
                    rank=1,
                    score=0.6,
                    text="完全不同的内容。",
                    chunk_index=0,
                    document_id="doc-b",
                    filename="doc-b.md",
                    page_number=None,
                ),
            ),
        )
        metrics = compute_case_metrics(
            case,
            _annotation(),
            response,
            parse_citations(response.answer),
        )
        assert metrics.covered_fact_count == 1
        assert metrics.grounded_fact_count == 0
        assert not metrics.strict_pass

    def test_multiple_supporting_evidence_indexes(self) -> None:
        fact = RequiredFact(
            fact_id="multi",
            accepted_phrases=("事实陈述",),
            supporting_evidence_indexes=(1, 2),
        )
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                make_expectation(document_id="doc-a", required_terms=("事实陈述",)),
                make_expectation(document_id="doc-b", required_terms=("另一个",)),
            ),
        )
        response = _response(
            "事实陈述。[来源2]",
            sources=(
                AnswerSource(
                    rank=1,
                    score=0.6,
                    text="其他内容。",
                    chunk_index=0,
                    document_id="doc-a",
                    filename="doc-a.md",
                    page_number=None,
                ),
                AnswerSource(
                    rank=2,
                    score=0.5,
                    text="事实陈述，同时包含另一个要点。",
                    chunk_index=1,
                    document_id="doc-b",
                    filename="doc-b.md",
                    page_number=None,
                ),
            ),
        )
        metrics = compute_case_metrics(
            case,
            _annotation(facts=(fact,)),
            response,
            parse_citations(response.answer),
        )
        assert metrics.grounded_fact_count == 1
        assert metrics.facts[0].supporting_citation_numbers == (2,)

    def test_page_number_must_match(self) -> None:
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                EvidenceExpectation(
                    document_id="doc-a",
                    page_number=3,
                    required_terms=("事实陈述",),
                ),
            ),
        )
        response = _response(
            "事实陈述。[来源1]",
            sources=(
                AnswerSource(
                    rank=1,
                    score=0.6,
                    text="事实陈述。",
                    chunk_index=0,
                    document_id="doc-a",
                    filename="doc-a.md",
                    page_number=2,
                ),
            ),
        )
        metrics = compute_case_metrics(
            case,
            _annotation(),
            response,
            parse_citations(response.answer),
        )
        assert metrics.grounded_fact_count == 0

    def test_document_id_mismatch_is_not_grounded(self) -> None:
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                make_expectation(document_id="doc-a", required_terms=("事实陈述",)),
            ),
        )
        response = _response(
            "事实陈述。[来源1]",
            sources=(
                AnswerSource(
                    rank=1,
                    score=0.6,
                    text="事实陈述。",
                    chunk_index=0,
                    document_id="other",
                    filename="other.md",
                    page_number=None,
                ),
            ),
        )
        metrics = compute_case_metrics(
            case,
            _annotation(),
            response,
            parse_citations(response.answer),
        )
        assert metrics.grounded_fact_count == 0

    def test_source_text_without_required_terms_is_not_grounded(self) -> None:
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                make_expectation(document_id="doc-a", required_terms=("必备术语",)),
            ),
        )
        response = _response(
            "事实陈述。[来源1]",
            sources=(
                AnswerSource(
                    rank=1,
                    score=0.6,
                    text="完全无关的段落内容，没有任何关键词。",
                    chunk_index=0,
                    document_id="doc-a",
                    filename="doc-a.md",
                    page_number=None,
                ),
            ),
        )
        metrics = compute_case_metrics(
            case,
            _annotation(),
            response,
            parse_citations(response.answer),
        )
        assert metrics.grounded_fact_count == 0

    def test_invalid_citation_number_is_recorded_not_fatal(self) -> None:
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                make_expectation(document_id="doc-a", required_terms=("事实陈述",)),
            ),
        )
        response = _response("事实陈述。[来源4]")
        metrics = compute_case_metrics(
            case,
            _annotation(),
            response,
            parse_citations(response.answer),
        )
        assert metrics.invalid_citation_occurrence_count == 1
        assert metrics.invalid_citation_numbers == (4,)
        assert metrics.citation_validity_rate == 0.0

    def test_malformed_citation_is_recorded(self) -> None:
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                make_expectation(document_id="doc-a", required_terms=("事实陈述",)),
            ),
        )
        response = _response("事实陈述。[来源A]")
        metrics = compute_case_metrics(
            case,
            _annotation(),
            response,
            parse_citations(response.answer),
        )
        assert metrics.malformed_citation_count == 1
        assert "[来源A]" in metrics.malformed_citation_fragments

    def test_no_citation_yields_null_validity_rate(self) -> None:
        case = make_case(
            "c-001",
            answerable=True,
            expected_evidence=(
                make_expectation(document_id="doc-a", required_terms=("事实陈述",)),
            ),
        )
        response = _response("事实陈述。")
        metrics = compute_case_metrics(
            case,
            _annotation(),
            response,
            parse_citations(response.answer),
        )
        assert metrics.citation_validity_rate is None


class TestDecisionMetrics:
    def test_true_answer(self) -> None:
        case = make_case("c-001", answerable=True)
        metrics = compute_case_metrics(
            case,
            _annotation(reference_answer="事实陈述。"),
            _response("事实陈述。[来源1]"),
            parse_citations("事实陈述。[来源1]"),
        )
        assert metrics.decision_correct
        assert metrics.answer_status == "answered"

    def test_false_refusal(self) -> None:
        case = make_case("c-001", answerable=True)
        metrics = compute_case_metrics(
            case,
            _annotation(reference_answer="事实陈述。"),
            _response(INSUFFICIENT_CONTEXT_ANSWER, status="insufficient_context"),
            parse_citations(INSUFFICIENT_CONTEXT_ANSWER),
        )
        assert not metrics.decision_correct

    def test_true_refusal(self) -> None:
        case = make_case(
            "c-001",
            answerable=False,
            expected_evidence=(),
            category="out_of_scope_far",
        )
        metrics = compute_case_metrics(
            case,
            _annotation(facts=(), reference_answer=INSUFFICIENT_CONTEXT_ANSWER),
            _response(INSUFFICIENT_CONTEXT_ANSWER, status="insufficient_context"),
            parse_citations(INSUFFICIENT_CONTEXT_ANSWER),
        )
        assert metrics.decision_correct
        assert metrics.strict_pass

    def test_false_answer(self) -> None:
        case = make_case(
            "c-001",
            answerable=False,
            expected_evidence=(),
            category="out_of_scope_far",
        )
        metrics = compute_case_metrics(
            case,
            _annotation(facts=(), reference_answer=INSUFFICIENT_CONTEXT_ANSWER),
            _response("编造的答案。", status="answered"),
            parse_citations("编造的答案。"),
        )
        assert not metrics.decision_correct


class TestStrictPass:
    def test_refusal_text_mismatch_fails(self) -> None:
        case = make_case(
            "c-001",
            answerable=False,
            expected_evidence=(),
            category="out_of_scope_far",
        )
        metrics = compute_case_metrics(
            case,
            _annotation(facts=(), reference_answer=INSUFFICIENT_CONTEXT_ANSWER),
            _response("资料不足。", status="insufficient_context"),
            parse_citations("资料不足。"),
        )
        assert metrics.answer_status == "insufficient_context"
        assert not metrics.strict_pass

    def test_answerable_requires_at_least_one_valid_citation(self) -> None:
        case = make_case("c-001", answerable=True)
        metrics = compute_case_metrics(
            case,
            _annotation(),
            _response("事实陈述。"),
            parse_citations("事实陈述。"),
        )
        assert metrics.covered_fact_count == 1
        assert not metrics.strict_pass


class TestAggregate:
    def _metrics_for(
        self,
        rows: list[tuple[bool, str, int, int, bool]],
    ):
        from src.evaluation.answer_models import CaseAnswerMetrics

        built: list[CaseAnswerMetrics] = []
        for answerable, status, covered, required, grounded in rows:
            fact_coverage = covered / required if required else None
            grounded_coverage = grounded / required if required else None
            built.append(
                CaseAnswerMetrics(
                    case_id=f"c-{len(built)+1:03d}",
                    split="test",
                    category="direct",
                    difficulty="easy",
                    answerable=answerable,
                    answer_status=status,
                    decision_correct=True,
                    question="问题？",
                    reference_answer="参考答案。",
                    answer="答案。",
                    required_fact_count=required,
                    covered_fact_count=covered,
                    grounded_fact_count=grounded,
                    fact_coverage=fact_coverage,
                    grounded_fact_coverage=grounded_coverage,
                    complete_fact_coverage=covered == required,
                    complete_grounded_fact_coverage=grounded == required,
                    forbidden_phrase_hits=(),
                    contradiction_free=True,
                    citation_occurrence_count=1,
                    unique_cited_source_count=1,
                    valid_citation_occurrence_count=1,
                    invalid_citation_occurrence_count=0,
                    invalid_citation_numbers=(),
                    malformed_citation_count=0,
                    malformed_citation_fragments=(),
                    citation_validity_rate=1.0,
                    facts=(),
                    max_relevance_score=0.6,
                    relevance_threshold=0.35,
                    retrieval_elapsed_ms=1.0,
                    generation_elapsed_ms=2.0,
                    total_elapsed_ms=3.0,
                    reranker_applied=False,
                    reranker_fallback=False,
                    strict_pass=covered == required and grounded == required,
                )
            )
        return tuple(built)

    def test_answerable_denominator_only(self) -> None:
        metrics = self._metrics_for(
            [
                (True, "answered", 1, 1, 1),
                (True, "answered", 1, 2, 1),
                (False, "insufficient_context", 0, 0, 0),
            ]
        )
        aggregate = compute_aggregate_metrics(metrics)
        assert aggregate.answerable_count == 2
        # (1.0 + 0.5) / 2
        assert aggregate.average_fact_coverage == pytest.approx(0.75)

    def test_false_refusal_counts_zero_coverage(self) -> None:
        from src.evaluation.answer_models import CaseAnswerMetrics

        base = self._metrics_for(
            [
                (True, "answered", 2, 2, 2),
                (True, "insufficient_context", 2, 2, 0),
            ]
        )
        # Simulate the real runner: a false refusal never contains the fact
        # phrases, so its per-case coverage is 0.0, not 1.0. The aggregation
        # must keep that 0.0 in the denominator instead of excluding it.
        refusal = CaseAnswerMetrics(
            **{
                **base[1].__dict__,
                "covered_fact_count": 0,
                "fact_coverage": 0.0,
            }
        )
        metrics = compute_aggregate_metrics((base[0], refusal))
        assert metrics.average_fact_coverage == pytest.approx(0.5)

    def test_citation_validity_uses_occurrence_weighting(self) -> None:
        from src.evaluation.answer_models import CaseAnswerMetrics

        base = self._metrics_for([(True, "answered", 1, 1, 1)])
        with_many = CaseAnswerMetrics(
            **{
                **base[0].__dict__,
                "citation_occurrence_count": 3,
                "valid_citation_occurrence_count": 2,
                "invalid_citation_occurrence_count": 1,
                "citation_validity_rate": 2 / 3,
            }
        )
        aggregate = compute_aggregate_metrics((base[0], with_many))
        # (1 + 2) / (1 + 3) = 0.75, not the mean of 1.0 and 0.667.
        assert aggregate.citation_validity_rate == pytest.approx(0.75)

    def test_no_citations_yields_null_validity(self) -> None:
        metrics = compute_aggregate_metrics(self._metrics_for([]))
        assert metrics.citation_validity_rate is None

    def test_no_nan_is_emitted(self) -> None:
        metrics = self._metrics_for([])
        aggregate = compute_aggregate_metrics(metrics)
        for value in (
            aggregate.decision_accuracy,
            aggregate.false_answer_rate,
            aggregate.false_refusal_rate,
            aggregate.average_fact_coverage,
            aggregate.strict_pass_rate,
        ):
            assert value is None or not math.isnan(value)

    def test_grouping_is_stable(self) -> None:
        metrics = self._metrics_for(
            [
                (True, "answered", 1, 1, 1),
                (True, "answered", 1, 1, 1),
                (False, "insufficient_context", 0, 0, 0),
            ]
        )
        grouped = group_case_metrics(metrics, "split")
        assert list(grouped.keys()) == ["test"]
        assert grouped["test"].case_count == 3


class TestLatency:
    def test_mean_median_p95_single_element(self) -> None:
        summary = summarize_latency([10.0])
        assert summary.count == 1
        assert summary.mean == 10.0
        assert summary.median == 10.0
        assert summary.p95 == 10.0
        assert summary.minimum == 10.0
        assert summary.maximum == 10.0

    def test_two_elements(self) -> None:
        summary = summarize_latency([1.0, 2.0])
        assert summary.median == 1.5
        assert summary.p95 == pytest.approx(1.95)

    def test_generation_zero_is_kept(self) -> None:
        summary = summarize_latency([0.0, 100.0])
        assert summary.count == 2
        assert summary.minimum == 0.0

    def test_invalid_inputs_are_dropped(self) -> None:
        summary = summarize_latency([1.0, math.nan, math.inf, -5.0, 2.0])
        assert summary.count == 2
        assert summary.minimum == 1.0
        assert summary.maximum == 2.0

    def test_empty_series(self) -> None:
        summary = summarize_latency([])
        assert summary.count == 0
        assert summary.mean is None
        assert summary.p95 is None
