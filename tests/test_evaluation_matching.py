"""Offline tests for evidence matching and text normalization."""

import pytest

from src.evaluation.matching import (
    compute_evidence_match,
    matches_expectation,
    normalize_text,
)
from src.evaluation.models import EvidenceExpectation

from evaluation_helpers import make_evaluation_case


def _expectation(
    document_id: str = "eval-doc-a",
    page_number: int | None = None,
    terms: tuple[str, ...] = ("alpha",),
) -> EvidenceExpectation:
    return EvidenceExpectation(
        document_id=document_id,
        page_number=page_number,
        required_terms=terms,
    )


def _source(
    rank: int,
    text: str,
    *,
    document_id: str = "eval-doc-a",
    page_number: int | None = None,
    score: float = 0.8,
) -> dict[str, object]:
    return {
        "rank": rank,
        "score": score,
        "text": text,
        "chunk_index": rank - 1,
        "document_id": document_id,
        "filename": "doc.md",
        "page_number": page_number,
    }


def test_normalize_text_casefolds_and_collapses_whitespace() -> None:
    assert normalize_text("  Alpha    Service  ") == "alpha service"
    assert normalize_text("中文  内容") == "中文 内容"
    assert normalize_text("Ｒｏｕｔｅｒ") == "router"


def test_normalize_text_keeps_chinese_content() -> None:
    assert normalize_text("业务逻辑") == "业务逻辑"


def test_document_id_must_match() -> None:
    assert matches_expectation("alpha 内容", None, _expectation()) is True
    case = make_evaluation_case("q001", expected_evidence=(_expectation(),))

    match = compute_evidence_match(
        [_source(1, "alpha 内容", document_id="eval-other")],
        case,
    )

    assert match.first_relevant_rank is None
    assert match.hit_at_5 is False


def test_page_number_null_matches_any_page() -> None:
    expectation = _expectation(page_number=None)
    assert matches_expectation("alpha", 1, expectation) is True
    assert matches_expectation("alpha", None, expectation) is True


def test_page_number_must_match_when_set() -> None:
    expectation = _expectation(page_number=2)
    assert matches_expectation("alpha", 2, expectation) is True
    assert matches_expectation("alpha", 3, expectation) is False
    assert matches_expectation("alpha", None, expectation) is False


def test_all_required_terms_must_be_present() -> None:
    expectation = _expectation(terms=("alpha", "service"))
    assert matches_expectation("alpha service 内容", None, expectation) is True
    assert matches_expectation("alpha 内容", None, expectation) is False


def test_terms_match_case_insensitively() -> None:
    expectation = _expectation(terms=("ROUTER",))
    assert matches_expectation("Router 层负责", None, expectation) is True


def test_terms_match_across_whitespace() -> None:
    expectation = _expectation(terms=("alpha 服务",))
    assert matches_expectation("alpha\n  服务\n内容", None, expectation) is True


def test_chinese_terms_match_exactly() -> None:
    expectation = _expectation(terms=("业务逻辑",))
    assert matches_expectation("这里讨论业务逻辑的写法", None, expectation) is True


def test_same_evidence_group_counts_once() -> None:
    case = make_evaluation_case(
        "q001",
        expected_evidence=(_expectation(terms=("alpha", "服务")),),
    )
    sources = [
        _source(1, "alpha 服务第一段"),
        _source(2, "alpha 服务第二段"),
        _source(3, "alpha 服务第三段"),
    ]

    match = compute_evidence_match(sources, case)

    assert match.matched_evidence_count_at_5 == 1
    assert match.first_relevant_rank == 1
    assert match.reciprocal_rank == 1.0


def test_multi_evidence_recall_counts_each_group_once() -> None:
    case = make_evaluation_case(
        "q001",
        category="multi_evidence",
        expected_evidence=(
            _expectation(document_id="eval-doc-a", terms=("alpha",)),
            _expectation(document_id="eval-doc-b", terms=("beta",)),
        ),
    )
    sources = [
        _source(1, "alpha 内容", document_id="eval-doc-a"),
        _source(2, "beta 内容", document_id="eval-doc-b"),
        _source(3, "gamma 内容", document_id="eval-doc-c"),
    ]

    match = compute_evidence_match(sources, case)

    assert match.matched_evidence_count_at_1 == 1
    assert match.matched_evidence_count_at_3 == 2
    assert match.total_evidence_count == 2
    assert match.first_relevant_rank == 1
    assert match.reciprocal_rank == 1.0


def test_evidence_miss_yields_zero_mrr() -> None:
    case = make_evaluation_case(
        "q001",
        expected_evidence=(_expectation(terms=("不存在的词",)),),
    )
    sources = [_source(1, "alpha 内容"), _source(2, "beta 内容")]

    match = compute_evidence_match(sources, case)

    assert match.first_relevant_rank is None
    assert match.reciprocal_rank == 0.0
    assert match.hit_at_5 is False


def test_unanswerable_case_has_null_retrieval_flags() -> None:
    case = make_evaluation_case("q001", answerable=False, expected_evidence=())
    sources = [_source(1, "alpha 内容")]

    match = compute_evidence_match(sources, case)

    assert match.hit_at_1 is None
    assert match.reciprocal_rank is None
    assert match.first_relevant_rank is None
    assert match.total_evidence_count == 0


def test_second_rank_match_sets_reciprocal_rank() -> None:
    case = make_evaluation_case(
        "q001",
        expected_evidence=(_expectation(terms=("beta",)),),
    )
    sources = [_source(1, "alpha 内容"), _source(2, "beta 内容")]

    match = compute_evidence_match(sources, case)

    assert match.first_relevant_rank == 2
    assert match.reciprocal_rank == pytest.approx(0.5)
    assert match.hit_at_1 is False
    assert match.hit_at_3 is True


def test_third_rank_match_sets_reciprocal_rank() -> None:
    case = make_evaluation_case(
        "q001",
        expected_evidence=(_expectation(terms=("gamma",)),),
    )
    sources = [
        _source(1, "alpha 内容"),
        _source(2, "beta 内容"),
        _source(3, "gamma 内容"),
    ]

    match = compute_evidence_match(sources, case)

    assert match.first_relevant_rank == 3
    assert match.reciprocal_rank == pytest.approx(1 / 3)
