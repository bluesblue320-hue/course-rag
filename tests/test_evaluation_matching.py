"""Test evidence matching between retrieved chunks and labelled expectations."""

import pytest

from src.evaluation.matching import (
    count_matched_evidence,
    evidence_matches,
    first_relevant_rank,
    matched_evidence_indexes,
    normalize_text,
    text_matches_expectation,
)
from tests.evaluation_helpers import make_expectation, make_source


def test_normalize_text_collapses_whitespace_and_case() -> None:
    assert normalize_text("  Hello   WORLD \n") == "hello world"


def test_normalize_text_applies_nfkc_folding() -> None:
    assert normalize_text("ＦａｓｔＡＰＩ") == "fastapi"


def test_normalize_text_keeps_chinese_characters_intact() -> None:
    assert normalize_text("依赖注入 让上层接收依赖") == "依赖注入 让上层接收依赖"


def test_normalize_text_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_text(123)  # type: ignore[arg-type]


def test_evidence_matches_requires_every_term() -> None:
    expectation = make_expectation(required_terms=("主键", "外键"))
    assert evidence_matches(
        expectation,
        make_source(1, 0.9, text="主键与外键共同保证一致性"),
    )
    assert not evidence_matches(
        expectation,
        make_source(1, 0.9, text="只提到了主键"),
    )


def test_evidence_matches_requires_same_document() -> None:
    expectation = make_expectation(document_id="doc-a", required_terms=("主键",))
    source = make_source(1, 0.9, text="主键唯一", document_id="doc-b")
    assert not evidence_matches(expectation, source)


def test_evidence_matches_ignores_page_when_expectation_has_none() -> None:
    expectation = make_expectation(page_number=None, required_terms=("主键",))
    assert evidence_matches(
        expectation,
        make_source(1, 0.9, text="主键唯一", page_number=7),
    )


def test_evidence_matches_enforces_page_when_expectation_sets_one() -> None:
    expectation = make_expectation(page_number=3, required_terms=("主键",))
    assert evidence_matches(
        expectation,
        make_source(1, 0.9, text="主键唯一", page_number=3),
    )
    assert not evidence_matches(
        expectation,
        make_source(1, 0.9, text="主键唯一", page_number=4),
    )


def test_evidence_matches_is_case_insensitive_for_latin_terms() -> None:
    expectation = make_expectation(required_terms=("FastAPI",))
    assert evidence_matches(
        expectation,
        make_source(1, 0.9, text="fastapi 使用 Depends 声明依赖"),
    )


def test_evidence_matches_rejects_empty_required_terms() -> None:
    expectation = make_expectation(required_terms=())
    assert not evidence_matches(expectation, make_source(1, 0.9, text="任意"))


def test_text_matches_expectation_ignores_document_and_page() -> None:
    expectation = make_expectation(
        document_id="doc-z",
        page_number=99,
        required_terms=("事务", "回滚"),
    )
    assert text_matches_expectation(expectation, "事务失败时回滚")
    assert not text_matches_expectation(expectation, "只提到事务")


def test_text_matches_expectation_rejects_empty_terms() -> None:
    assert not text_matches_expectation(
        make_expectation(required_terms=()),
        "任意正文",
    )


def test_matched_evidence_indexes_counts_each_group_once() -> None:
    expectations = (
        make_expectation(required_terms=("主键",)),
        make_expectation(required_terms=("外键",)),
    )
    sources = (
        make_source(1, 0.9, text="主键唯一标识一行"),
        make_source(2, 0.8, text="主键再次出现"),
        make_source(3, 0.7, text="外键指向主键"),
    )
    assert matched_evidence_indexes(expectations, sources, 2) == frozenset({0})
    assert matched_evidence_indexes(expectations, sources, 3) == frozenset({0, 1})


def test_matched_evidence_indexes_respects_the_k_cutoff() -> None:
    expectations = (make_expectation(required_terms=("外键",)),)
    sources = (
        make_source(1, 0.9, text="主键唯一"),
        make_source(2, 0.8, text="外键指向主键"),
    )
    assert matched_evidence_indexes(expectations, sources, 1) == frozenset()
    assert matched_evidence_indexes(expectations, sources, 2) == frozenset({0})


def test_matched_evidence_indexes_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        matched_evidence_indexes((make_expectation(),), (), 0)


def test_count_matched_evidence_returns_group_count() -> None:
    expectations = (
        make_expectation(required_terms=("主键",)),
        make_expectation(required_terms=("外键",)),
    )
    sources = (make_source(1, 0.9, text="主键与外键都出现"),)
    assert count_matched_evidence(expectations, sources, 5) == 2


def test_first_relevant_rank_returns_one_based_position() -> None:
    expectations = (make_expectation(required_terms=("索引",)),)
    sources = (
        make_source(1, 0.9, text="事务与回滚"),
        make_source(2, 0.8, text="索引加速查询"),
    )
    assert first_relevant_rank(expectations, sources) == 2


def test_first_relevant_rank_returns_none_without_any_match() -> None:
    expectations = (make_expectation(required_terms=("索引",)),)
    sources = (make_source(1, 0.9, text="事务与回滚"),)
    assert first_relevant_rank(expectations, sources) is None


def test_first_relevant_rank_returns_none_without_expectations() -> None:
    assert first_relevant_rank((), (make_source(1, 0.9),)) is None
