"""Tests for answer responses loading and strict validation."""

import json
import math
from pathlib import Path

import pytest

from src.evaluation.answer_responses import load_answer_responses
from src.exceptions import AnswerResponseValidationError
from tests.evaluation_helpers import make_case

SUFFICIENT_REFUSAL = (
    "当前课程资料中没有足够信息回答这个问题。请尝试换一种问法，"
    "或切换到语义检索查看最接近的课程原文。"
)


def _source(
    rank: int = 1,
    score: float = 0.6,
    text: str = "来源正文内容。",
    document_id: str = "doc-a",
    filename: str = "doc-a.md",
) -> dict[str, object]:
    return {
        "rank": rank,
        "score": score,
        "text": text,
        "chunk_index": 0,
        "document_id": document_id,
        "filename": filename,
        "page_number": None,
    }


def _response_row(
    case_id: str = "c-001",
    answer_status: str = "answered",
    answer: str = "这是一个回答。",
    sources: list[dict[str, object]] | None = None,
    max_relevance_score: float | None = None,
    relevance_threshold: float = 0.35,
    retrieval_elapsed_ms: float = 2.0,
    generation_elapsed_ms: float = 100.0,
    total_elapsed_ms: float = 102.0,
    reranker_applied: bool = False,
    reranker_fallback: bool = False,
) -> dict[str, object]:
    if sources is None:
        sources = [_source()]
    if max_relevance_score is None:
        max_relevance_score = sources[0]["score"] if sources else None
    return {
        "case_id": case_id,
        "answer_status": answer_status,
        "answer": answer,
        "sources": sources,
        "max_relevance_score": max_relevance_score,
        "relevance_threshold": relevance_threshold,
        "retrieval_elapsed_ms": retrieval_elapsed_ms,
        "generation_elapsed_ms": generation_elapsed_ms,
        "total_elapsed_ms": total_elapsed_ms,
        "reranker_applied": reranker_applied,
        "reranker_fallback": reranker_fallback,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _two_cases():
    return (
        make_case("c-001", answerable=True),
        make_case(
            "c-002",
            answerable=False,
            expected_evidence=(),
            category="out_of_scope_far",
        ),
    )


def _refusal_row() -> dict[str, object]:
    return _response_row(
        "c-002",
        answer_status="insufficient_context",
        answer=SUFFICIENT_REFUSAL,
    )


def _write_two_rows(
    tmp_path: Path,
    first: dict[str, object],
) -> Path:
    return _write(
        tmp_path / "responses.jsonl",
        [first, _refusal_row()],
    )


class TestLoadAnswerResponses:
    def test_full_valid_file_reorders_to_dataset_order(self, tmp_path: Path) -> None:
        cases = _two_cases()
        path = _write(
            tmp_path / "responses.jsonl",
            [
                _response_row("c-002", answer_status="insufficient_context", answer=SUFFICIENT_REFUSAL),
                _response_row("c-001"),
            ],
        )
        responses = load_answer_responses(path, cases)
        assert [response.case_id for response in responses] == ["c-001", "c-002"]

    def test_missing_case_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        path = _write(tmp_path / "responses.jsonl", [_response_row("c-001")])
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "缺少 case" in str(exc.value)

    def test_extra_case_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        path = _write(
            tmp_path / "responses.jsonl",
            [_response_row("c-001"), _response_row("c-002", answer_status="insufficient_context", answer=SUFFICIENT_REFUSAL), _response_row("c-999")],
        )
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "数据集中不存在" in str(exc.value)

    def test_duplicate_case_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        path = _write(
            tmp_path / "responses.jsonl",
            [_response_row("c-001"), _response_row("c-001")],
        )
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "重复" in str(exc.value)

    def test_invalid_answer_status_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001")
        row["answer_status"] = "maybe"
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "answer_status" in str(exc.value)

    def test_empty_answer_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001", answer="")
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "answer 不允许为空" in str(exc.value)

    def test_rank_not_starting_at_one_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001", sources=[_source(rank=2)])
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "连续递增" in str(exc.value)

    def test_rank_gap_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row(
            "c-001",
            sources=[_source(rank=1), _source(rank=3, text="另一个来源")],
        )
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "连续递增" in str(exc.value)

    def test_later_rank_gap_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row(
            "c-001",
            sources=[
                _source(rank=1),
                _source(rank=2, text="第二个来源"),
                _source(rank=4, text="第四个来源"),
            ],
        )
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "连续递增" in str(exc.value)

    def test_duplicate_rank_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row(
            "c-001",
            sources=[_source(rank=1), _source(rank=1, text="另一个来源")],
        )
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "连续递增" in str(exc.value)

    def test_bool_score_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001", sources=[_source(score=1)])
        row["sources"][0]["score"] = True  # type: ignore[index]
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "必须是数字" in str(exc.value)

    def test_nan_score_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001", sources=[_source(score=math.nan)])
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "NaN" in str(exc.value)

    def test_infinite_score_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001", sources=[_source(score=math.inf)])
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "Infinity" in str(exc.value)

    def test_out_of_range_score_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001", sources=[_source(score=1.5)])
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "不能大于" in str(exc.value)

    def test_invalid_page_number_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001", sources=[_source()])
        row["sources"][0]["page_number"] = 0  # type: ignore[index]
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "page_number" in str(exc.value)

    def test_negative_elapsed_is_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001", retrieval_elapsed_ms=-1.0)
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "不能小于" in str(exc.value)

    def test_bool_fields_with_int_values_are_rejected(self, tmp_path: Path) -> None:
        cases = _two_cases()
        row = _response_row("c-001", reranker_applied=1)  # type: ignore[arg-type]
        path = _write_two_rows(tmp_path, row)
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        assert "布尔值" in str(exc.value)

    def test_no_sources_with_null_max_score_is_accepted(self, tmp_path: Path) -> None:
        cases = (
            make_case(
                "c-001",
                answerable=False,
                expected_evidence=(),
                category="out_of_scope_far",
            ),
        )
        row = _response_row(
            "c-001",
            answer_status="insufficient_context",
            answer=SUFFICIENT_REFUSAL,
            sources=[],
            max_relevance_score=None,
        )
        path = _write(tmp_path / "responses.jsonl", [row])
        responses = load_answer_responses(path, cases)
        assert responses[0].max_relevance_score is None

    def test_error_carries_line_number_without_absolute_path(
        self, tmp_path: Path
    ) -> None:
        cases = _two_cases()
        path = tmp_path / "responses.jsonl"
        path.write_text("{bad json\n", encoding="utf-8")
        with pytest.raises(AnswerResponseValidationError) as exc:
            load_answer_responses(path, cases)
        message = str(exc.value)
        assert "第 1 行" in message
        assert str(tmp_path) not in message
        assert not any(
            f"{letter}:" in message
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )


class TestMaxRelevanceScoreSemantics:
    def _single_case(self) -> tuple:
        return (make_case("c-001", answerable=True),)

    def _write_and_load(
        self,
        tmp_path: Path,
        row: dict[str, object],
    ) -> None:
        path = _write(tmp_path / "responses.jsonl", [row])
        load_answer_responses(path, self._single_case())

    def test_no_reranker_equal_score_passes(self, tmp_path: Path) -> None:
        row = _response_row(
            "c-001",
            sources=[_source(rank=1, score=0.8), _source(rank=2, score=0.7, text="二")],
            max_relevance_score=0.8,
            reranker_applied=False,
        )
        self._write_and_load(tmp_path, row)

    def test_no_reranker_greater_max_is_rejected(self, tmp_path: Path) -> None:
        row = _response_row(
            "c-001",
            sources=[_source(rank=1, score=0.8), _source(rank=2, score=0.7, text="二")],
            max_relevance_score=0.9,
            reranker_applied=False,
        )
        with pytest.raises(AnswerResponseValidationError) as exc:
            self._write_and_load(tmp_path, row)
        assert "必须与最高检索分数一致" in str(exc.value)

    def test_reranker_applied_candidate_max_greater_passes(
        self, tmp_path: Path
    ) -> None:
        row = _response_row(
            "c-001",
            sources=[_source(rank=1, score=0.8), _source(rank=2, score=0.7, text="二")],
            max_relevance_score=0.9,
            reranker_applied=True,
        )
        self._write_and_load(tmp_path, row)

    def test_reranker_applied_exact_equal_passes(self, tmp_path: Path) -> None:
        row = _response_row(
            "c-001",
            sources=[_source(rank=1, score=0.9), _source(rank=2, score=0.7, text="二")],
            max_relevance_score=0.9,
            reranker_applied=True,
        )
        self._write_and_load(tmp_path, row)

    def test_reranker_applied_max_below_source_is_rejected(
        self, tmp_path: Path
    ) -> None:
        row = _response_row(
            "c-001",
            sources=[_source(rank=1, score=0.9), _source(rank=2, score=0.7, text="二")],
            max_relevance_score=0.8,
            reranker_applied=True,
        )
        with pytest.raises(AnswerResponseValidationError) as exc:
            self._write_and_load(tmp_path, row)
        assert "不能低于" in str(exc.value)

    def test_epsilon_difference_passes(self, tmp_path: Path) -> None:
        row = _response_row(
            "c-001",
            sources=[_source(rank=1, score=0.9000000001)],
            max_relevance_score=0.9,
            reranker_applied=False,
        )
        self._write_and_load(tmp_path, row)

    def test_reranker_fallback_with_false_applied_uses_strict_rule(
        self, tmp_path: Path
    ) -> None:
        # A fallback keeps the original vector order, so the final Top-K must
        # contain the highest vector candidate and scores must match.
        row = _response_row(
            "c-001",
            sources=[_source(rank=1, score=0.8)],
            max_relevance_score=0.9,
            reranker_applied=False,
            reranker_fallback=True,
        )
        with pytest.raises(AnswerResponseValidationError) as exc:
            self._write_and_load(tmp_path, row)
        assert "必须与最高检索分数一致" in str(exc.value)
