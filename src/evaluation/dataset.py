"""Load and strictly validate the labelled evaluation dataset."""

import json
from pathlib import Path
from typing import Any, cast

from src.documents import ChunkRecord
from src.evaluation.matching import text_matches_expectation
from src.evaluation.models import (
    CATEGORIES,
    DIFFICULTIES,
    MULTI_EVIDENCE_CATEGORY,
    SPLITS,
    EvaluationCase,
    EvidenceExpectation,
    Split,
)
from src.exceptions import DatasetValidationError

_REQUIRED_CASE_FIELDS = (
    "id",
    "split",
    "question",
    "answerable",
    "category",
    "difficulty",
    "expected_evidence",
    "notes",
)
_REQUIRED_EVIDENCE_FIELDS = ("document_id", "page_number", "required_terms")


def _fail(line_number: int, message: str) -> DatasetValidationError:
    return DatasetValidationError(f"数据集第 {line_number} 行: {message}")


def _parse_evidence(
    raw: object,
    line_number: int,
    known_document_ids: frozenset[str] | None,
) -> EvidenceExpectation:
    if not isinstance(raw, dict):
        raise _fail(line_number, "expected_evidence 的元素必须是对象")
    for field in _REQUIRED_EVIDENCE_FIELDS:
        if field not in raw:
            raise _fail(line_number, f"evidence 缺少字段: {field}")

    document_id = raw["document_id"]
    if not isinstance(document_id, str) or not document_id.strip():
        raise _fail(line_number, "evidence 的 document_id 必须是非空字符串")
    document_id = document_id.strip()
    if known_document_ids is not None and document_id not in known_document_ids:
        raise _fail(line_number, f"evidence 引用了未知 document_id: {document_id}")

    page_number = raw["page_number"]
    if page_number is not None:
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise _fail(line_number, "evidence 的 page_number 必须是整数或 null")
        if page_number < 1:
            raise _fail(line_number, "evidence 的 page_number 不能小于 1")

    required_terms = raw["required_terms"]
    if not isinstance(required_terms, list) or not required_terms:
        raise _fail(line_number, "evidence 的 required_terms 不能为空")
    parsed_terms: list[str] = []
    for term in required_terms:
        if not isinstance(term, str) or not term.strip():
            raise _fail(line_number, "evidence 的 required_terms 不允许空字符串")
        parsed_terms.append(term.strip())

    return EvidenceExpectation(
        document_id=document_id,
        page_number=page_number,
        required_terms=tuple(parsed_terms),
    )


def _parse_case(
    raw: object,
    line_number: int,
    known_document_ids: frozenset[str] | None,
) -> EvaluationCase:
    if not isinstance(raw, dict):
        raise _fail(line_number, "每一行必须是 JSON 对象")
    for field in _REQUIRED_CASE_FIELDS:
        if field not in raw:
            raise _fail(line_number, f"缺少字段: {field}")

    case_id = raw["id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise _fail(line_number, "id 必须是非空字符串")

    split = raw["split"]
    if split not in SPLITS:
        raise _fail(line_number, f"split 非法: {split!r}")

    question = raw["question"]
    if not isinstance(question, str) or not question.strip():
        raise _fail(line_number, "question 不能为空")

    answerable = raw["answerable"]
    if not isinstance(answerable, bool):
        raise _fail(line_number, "answerable 必须是布尔值")

    category = raw["category"]
    if category not in CATEGORIES:
        raise _fail(line_number, f"category 非法: {category!r}")

    difficulty = raw["difficulty"]
    if difficulty not in DIFFICULTIES:
        raise _fail(line_number, f"difficulty 非法: {difficulty!r}")

    notes = raw["notes"]
    if not isinstance(notes, str):
        raise _fail(line_number, "notes 必须是字符串")

    raw_evidence = raw["expected_evidence"]
    if not isinstance(raw_evidence, list):
        raise _fail(line_number, "expected_evidence 必须是数组")
    evidence = tuple(
        _parse_evidence(item, line_number, known_document_ids)
        for item in raw_evidence
    )

    if answerable and not evidence:
        raise _fail(line_number, "answerable=true 必须提供 expected_evidence")
    if not answerable and evidence:
        raise _fail(line_number, "answerable=false 不允许提供 expected_evidence")
    if category == MULTI_EVIDENCE_CATEGORY and len(evidence) < 2:
        raise _fail(line_number, "multi_evidence 问题至少需要两个 evidence group")
    if not answerable and category not in (
        "out_of_scope_far",
        "out_of_scope_near",
    ):
        raise _fail(line_number, "answerable=false 只能使用 out_of_scope_* 类别")
    if answerable and category in ("out_of_scope_far", "out_of_scope_near"):
        raise _fail(line_number, "out_of_scope_* 类别必须是 answerable=false")

    return EvaluationCase(
        id=case_id.strip(),
        split=cast(Split, split),
        question=question.strip(),
        answerable=answerable,
        category=category,
        difficulty=difficulty,
        expected_evidence=evidence,
        notes=notes,
    )


def load_dataset(
    dataset_path: Path,
    known_document_ids: frozenset[str] | None = None,
) -> tuple[EvaluationCase, ...]:
    """Parse a JSONL dataset and reject every structurally invalid record."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise DatasetValidationError("数据集文件不存在")

    try:
        raw_text = dataset_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetValidationError("数据集无法以 UTF-8 读取") from exc

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload: Any = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise _fail(line_number, "不是合法 JSON") from exc

        case = _parse_case(payload, line_number, known_document_ids)
        if case.id in seen_ids:
            raise _fail(line_number, f"id 重复: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise DatasetValidationError("数据集不能为空")
    return tuple(cases)


def validate_dataset_against_corpus(
    cases: tuple[EvaluationCase, ...],
    chunks: tuple[ChunkRecord, ...],
) -> None:
    """Check every labelled evidence group can be located in the corpus.

    This is a pure label-integrity check. It never calls an embedding model
    and never inspects retrieval output, so ground truth stays independent of
    what the system happens to return.
    """
    if not chunks:
        raise DatasetValidationError("语料没有任何 Chunk，无法校验标注")

    chunks_by_document: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    for case in cases:
        for position, expectation in enumerate(case.expected_evidence, start=1):
            candidates = chunks_by_document.get(expectation.document_id, [])
            if not candidates:
                raise DatasetValidationError(
                    f"用例 {case.id} 的第 {position} 组证据引用了语料中不存在的"
                    f" document_id: {expectation.document_id}"
                )
            matched = any(
                (
                    expectation.page_number is None
                    or expectation.page_number == chunk.page_number
                )
                and text_matches_expectation(expectation, chunk.text)
                for chunk in candidates
            )
            if not matched:
                raise DatasetValidationError(
                    f"用例 {case.id} 的第 {position} 组证据在语料中找不到匹配的"
                    f" Chunk: {list(expectation.required_terms)}"
                )


def count_by(
    cases: tuple[EvaluationCase, ...],
    attribute: str,
) -> dict[str, int]:
    """Return a deterministic count of cases grouped by one string attribute."""
    counts: dict[str, int] = {}
    for case in cases:
        value = getattr(case, attribute)
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def split_cases(
    cases: tuple[EvaluationCase, ...],
    split: str,
) -> tuple[EvaluationCase, ...]:
    """Return only the cases belonging to one split, preserving input order."""
    if split not in SPLITS:
        raise DatasetValidationError(f"split 非法: {split!r}")
    return tuple(case for case in cases if case.split == split)
