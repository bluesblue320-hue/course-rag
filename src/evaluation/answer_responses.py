"""Load and strictly validate answer responses for offline scoring.

A responses file records what the RAG system actually produced for every
case.  The loader rejects structural problems with line numbers and never
leaks absolute filesystem paths.
"""

import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.answer_models import (
    ANSWER_STATUSES,
    AnswerResponse,
    AnswerSource,
)
from src.evaluation.models import EvaluationCase
from src.exceptions import AnswerResponseValidationError

_REQUIRED_RESPONSE_FIELDS = (
    "case_id",
    "answer_status",
    "answer",
    "sources",
    "max_relevance_score",
    "relevance_threshold",
    "retrieval_elapsed_ms",
    "generation_elapsed_ms",
    "total_elapsed_ms",
    "reranker_applied",
    "reranker_fallback",
)
_REQUIRED_SOURCE_FIELDS = (
    "rank",
    "score",
    "text",
    "chunk_index",
    "document_id",
    "filename",
    "page_number",
)

# Tolerance when comparing max_relevance_score against source scores.
_SCORE_EPSILON = 1e-9


def _fail(line_number: int, message: str) -> AnswerResponseValidationError:
    return AnswerResponseValidationError(f"回答结果文件第 {line_number} 行: {message}")


def _validate_finite_number(
    value: object,
    line_number: int,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(line_number, f"{field_name} 必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise _fail(line_number, f"{field_name} 不允许 NaN 或 Infinity")
    if minimum is not None and number < minimum:
        raise _fail(line_number, f"{field_name} 不能小于 {minimum}")
    if maximum is not None and number > maximum:
        raise _fail(line_number, f"{field_name} 不能大于 {maximum}")
    return number


def _parse_source(raw: object, line_number: int) -> AnswerSource:
    if not isinstance(raw, dict):
        raise _fail(line_number, "sources 的元素必须是对象")
    for field in _REQUIRED_SOURCE_FIELDS:
        if field not in raw:
            raise _fail(line_number, f"source 缺少字段: {field}")

    rank = raw["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise _fail(line_number, "source 的 rank 必须是整数")
    if rank < 1:
        raise _fail(line_number, "source 的 rank 必须从 1 开始且大于 0")

    score = _validate_finite_number(
        raw["score"],
        line_number,
        "source 的 score",
        minimum=-1.0,
        maximum=1.0,
    )

    text = raw["text"]
    if not isinstance(text, str) or not text.strip():
        raise _fail(line_number, "source 的 text 不允许为空")

    chunk_index = raw["chunk_index"]
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
        raise _fail(line_number, "source 的 chunk_index 必须是非负整数")
    if chunk_index < 0:
        raise _fail(line_number, "source 的 chunk_index 不能为负数")

    document_id = raw["document_id"]
    if not isinstance(document_id, str) or not document_id.strip():
        raise _fail(line_number, "source 的 document_id 不允许为空")

    filename = raw["filename"]
    if not isinstance(filename, str):
        raise _fail(line_number, "source 的 filename 必须是字符串")

    page_number = raw["page_number"]
    if page_number is not None:
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise _fail(line_number, "source 的 page_number 必须是正整数或 null")
        if page_number < 1:
            raise _fail(line_number, "source 的 page_number 不能小于 1")

    return AnswerSource(
        rank=rank,
        score=score,
        text=text.strip(),
        chunk_index=chunk_index,
        document_id=document_id.strip(),
        filename=filename,
        page_number=page_number,
    )


def _parse_response(
    raw: object,
    line_number: int,
) -> AnswerResponse:
    if not isinstance(raw, dict):
        raise _fail(line_number, "每一行必须是 JSON 对象")
    for field in _REQUIRED_RESPONSE_FIELDS:
        if field not in raw:
            raise _fail(line_number, f"缺少字段: {field}")

    case_id = raw["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise _fail(line_number, "case_id 必须是非空字符串")

    answer_status = raw["answer_status"]
    if answer_status not in ANSWER_STATUSES:
        raise _fail(
            line_number,
            f"answer_status 非法: {answer_status!r}（仅允许 answered / insufficient_context）",
        )

    answer = raw["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise _fail(line_number, "answer 不允许为空")

    raw_sources = raw["sources"]
    if not isinstance(raw_sources, list):
        raise _fail(line_number, "sources 必须是数组")
    sources = tuple(_parse_source(item, line_number) for item in raw_sources)

    ranks = [source.rank for source in sources]
    expected_ranks = list(range(1, len(sources) + 1))
    if ranks != expected_ranks:
        raise _fail(
            line_number,
            "sources 的 rank 必须从 1 开始连续递增",
        )

    # Reranker flags must be validated before the max-score comparison below:
    # they decide which score rule applies, and coercing with bool() would
    # silently accept 1 / "false" / [].
    reranker_applied = raw["reranker_applied"]
    if not isinstance(reranker_applied, bool):
        raise _fail(line_number, "reranker_applied 必须是真正的布尔值")
    reranker_fallback = raw["reranker_fallback"]
    if not isinstance(reranker_fallback, bool):
        raise _fail(line_number, "reranker_fallback 必须是真正的布尔值")

    max_relevance_score = raw["max_relevance_score"]
    if max_relevance_score is not None:
        parsed_max_score = _validate_finite_number(
            max_relevance_score,
            line_number,
            "max_relevance_score",
            minimum=-1.0,
            maximum=1.0,
        )
        if sources:
            top_returned_score = max(source.score for source in sources)
            if reranker_applied:
                # The reranker may move the highest vector candidate out of
                # the final Top-K, so max_relevance_score (the candidate-pool
                # maximum, the refusal decision basis) may exceed the highest
                # returned source score.  It must never fall below any
                # returned source's retrieval score.
                if parsed_max_score + _SCORE_EPSILON < top_returned_score:
                    raise _fail(
                        line_number,
                        "max_relevance_score 不能低于任何最终来源的检索分数",
                    )
            else:
                # Without reranking (or when the reranker fell back and kept
                # the original vector order), the final Top-K contains the
                # highest vector candidate, so the scores must match.
                if abs(parsed_max_score - top_returned_score) > _SCORE_EPSILON:
                    raise _fail(
                        line_number,
                        "max_relevance_score 必须与最高检索分数一致",
                    )
    elif sources:
        raise _fail(line_number, "有 sources 时 max_relevance_score 不允许为 null")

    relevance_threshold = _validate_finite_number(
        raw["relevance_threshold"],
        line_number,
        "relevance_threshold",
        minimum=0.0,
        maximum=1.0,
    )

    retrieval_elapsed_ms = _validate_finite_number(
        raw["retrieval_elapsed_ms"],
        line_number,
        "retrieval_elapsed_ms",
        minimum=0.0,
    )
    generation_elapsed_ms = _validate_finite_number(
        raw["generation_elapsed_ms"],
        line_number,
        "generation_elapsed_ms",
        minimum=0.0,
    )
    total_elapsed_ms = _validate_finite_number(
        raw["total_elapsed_ms"],
        line_number,
        "total_elapsed_ms",
        minimum=0.0,
    )

    return AnswerResponse(
        case_id=case_id.strip(),
        answer_status=answer_status,  # type: ignore[arg-type]
        answer=answer,
        sources=sources,
        max_relevance_score=(
            None if max_relevance_score is None else parsed_max_score
        ),
        relevance_threshold=relevance_threshold,
        retrieval_elapsed_ms=retrieval_elapsed_ms,
        generation_elapsed_ms=generation_elapsed_ms,
        total_elapsed_ms=total_elapsed_ms,
        reranker_applied=reranker_applied,
        reranker_fallback=reranker_fallback,
    )


def load_answer_responses(
    responses_path: Path,
    cases: tuple[EvaluationCase, ...],
) -> tuple[AnswerResponse, ...]:
    """Parse one responses JSONL and reorder it to dataset order.

    Case coverage is exact: every dataset case must appear exactly once, and
    every response must reference a known dataset case.  Errors carry line
    numbers and never an absolute path.
    """
    responses_path = Path(responses_path)
    if not responses_path.is_file():
        raise AnswerResponseValidationError("回答结果文件不存在")

    try:
        raw_text = responses_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AnswerResponseValidationError("回答结果文件无法以 UTF-8 读取") from exc

    dataset_ids = [case.id for case in cases]
    dataset_id_set = set(dataset_ids)
    by_id: dict[str, AnswerResponse] = {}
    seen: set[str] = set()
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload: Any = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise _fail(line_number, "不是合法 JSON") from exc

        if not isinstance(payload, dict):
            raise _fail(line_number, "每一行必须是 JSON 对象")
        raw_case_id = payload.get("case_id")
        if not isinstance(raw_case_id, str) or not raw_case_id.strip():
            raise _fail(line_number, "case_id 必须是非空字符串")
        case_id = raw_case_id.strip()
        if case_id in seen:
            raise _fail(line_number, f"case_id 重复: {case_id}")
        seen.add(case_id)
        if case_id not in dataset_id_set:
            raise _fail(
                line_number,
                f"回答结果引用了数据集中不存在的 case_id: {case_id}",
            )
        by_id[case_id] = _parse_response(payload, line_number)

    if sorted(seen) != sorted(dataset_id_set):
        missing = sorted(dataset_id_set - seen)
        if missing:
            raise AnswerResponseValidationError(f"回答结果缺少 case: {missing}")

    return tuple(by_id[case_id] for case_id in dataset_ids)
