"""Load and validate evaluation datasets (JSONL) and corpus manifests."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.documents import ChunkRecord

from .models import (
    EvaluationCase,
    EvaluationDataError,
    EvidenceExpectation,
)

ALLOWED_SPLITS = ("calibration", "test")
ALLOWED_CATEGORIES = (
    "direct",
    "paraphrase",
    "multi_evidence",
    "out_of_scope_far",
    "out_of_scope_near",
)
ALLOWED_DIFFICULTIES = ("easy", "medium", "hard")


def _require(value: bool, message: str) -> None:
    if not value:
        raise EvaluationDataError(message)


def _parse_evidence(item: object, case_id: str) -> EvidenceExpectation:
    _require(
        isinstance(item, dict),
        f"case {case_id}: expected_evidence 中的每一项必须是对象",
    )
    document_id = item.get("document_id")
    page_number = item.get("page_number")
    required_terms = item.get("required_terms")
    _require(
        isinstance(document_id, str) and document_id.strip(),
        f"case {case_id}: evidence document_id 必须是非空字符串",
    )
    _require(
        page_number is None
        or (isinstance(page_number, int) and page_number >= 1),
        f"case {case_id}: evidence page_number 必须大于等于 1 或为 null",
    )
    _require(
        isinstance(required_terms, list)
        and len(required_terms) > 0
        and all(
            isinstance(term, str) and term.strip()
            for term in required_terms
        ),
        f"case {case_id}: required_terms 必须是非空字符串列表，不能包含空字符串",
    )
    return EvidenceExpectation(
        document_id=document_id,
        page_number=page_number,
        required_terms=tuple(str(term).strip() for term in required_terms),
    )


def _parse_case(line_number: int, raw: str) -> EvaluationCase:
    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(
            f"第 {line_number} 行不是合法的 JSON：{exc.msg}"
        ) from exc
    _require(
        isinstance(payload, dict),
        f"第 {line_number} 行必须是 JSON 对象",
    )
    case_id = payload.get("id")
    split = payload.get("split")
    question = payload.get("question")
    answerable = payload.get("answerable")
    category = payload.get("category")
    difficulty = payload.get("difficulty")
    evidence = payload.get("expected_evidence")
    notes = payload.get("notes")

    _require(
        isinstance(case_id, str) and case_id.strip(),
        f"第 {line_number} 行：id 必须是非空字符串",
    )
    _require(
        split in ALLOWED_SPLITS,
        f"case {case_id}: split 必须是 calibration 或 test",
    )
    _require(
        isinstance(question, str) and question.strip(),
        f"case {case_id}: question 不能为空",
    )
    _require(
        isinstance(answerable, bool),
        f"case {case_id}: answerable 必须是布尔值",
    )
    _require(
        category in ALLOWED_CATEGORIES,
        f"case {case_id}: category 非法：{category}",
    )
    _require(
        difficulty in ALLOWED_DIFFICULTIES,
        f"case {case_id}: difficulty 非法：{difficulty}",
    )
    _require(
        isinstance(notes, str),
        f"case {case_id}: notes 必须是字符串",
    )
    _require(
        isinstance(evidence, list),
        f"case {case_id}: expected_evidence 必须是列表",
    )

    parsed_evidence = tuple(
        _parse_evidence(item, case_id) for item in evidence
    )
    if answerable:
        _require(
            len(parsed_evidence) > 0,
            f"case {case_id}: answerable=true 但 expected_evidence 为空",
        )
    else:
        _require(
            len(parsed_evidence) == 0,
            f"case {case_id}: answerable=false 但 expected_evidence 非空",
        )
    if category == "multi_evidence":
        _require(
            len(parsed_evidence) >= 2,
            f"case {case_id}: multi_evidence 问题至少需要两个 evidence group",
        )

    return EvaluationCase(
        id=case_id,
        split=split,  # type: ignore[arg-type]
        question=question.strip(),
        answerable=answerable,
        category=category,
        difficulty=difficulty,
        expected_evidence=parsed_evidence,
        notes=notes.strip(),
    )


def load_dataset(dataset_path: Path) -> list[EvaluationCase]:
    """Load a JSONL dataset and validate its structure and labels."""
    dataset_path = Path(dataset_path)
    _require(
        dataset_path.is_file(),
        f"数据集文件不存在：{dataset_path}",
    )
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationDataError("数据集文件无法读取") from exc
    _require(len(lines) > 0, "数据集文件为空")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        case = _parse_case(line_number, stripped)
        _require(
            case.id not in seen_ids,
            f"重复的 case id：{case.id}",
        )
        seen_ids.add(case.id)
        cases.append(case)
    _require(len(cases) > 0, "数据集文件不包含任何问题")
    return cases


def validate_dataset_document_ids(
    cases: Iterable[EvaluationCase],
    document_ids: set[str],
) -> None:
    """Reject evidence groups that reference unknown corpus documents."""
    for case in cases:
        for evidence in case.expected_evidence:
            if evidence.document_id not in document_ids:
                raise EvaluationDataError(
                    f"case {case.id}: evidence 引用了不存在的 document_id："
                    f"{evidence.document_id}"
                )


def validate_dataset_integrity(
    cases: Iterable[EvaluationCase],
    chunks_by_document: dict[str, list[ChunkRecord]],
) -> None:
    """Verify every answerable evidence group is locatable in the corpus.

    This check only compares labels against chunk text and never runs
    embeddings, so it can run fully offline.
    """
    from .matching import normalize_text

    for case in cases:
        if not case.answerable:
            continue
        for index, evidence in enumerate(case.expected_evidence):
            chunks = chunks_by_document.get(evidence.document_id, [])
            located = False
            for chunk in chunks:
                if evidence.page_number is not None and (
                    chunk.page_number != evidence.page_number
                ):
                    continue
                chunk_text = normalize_text(chunk.text)
                if all(
                    normalize_text(term) in chunk_text
                    for term in evidence.required_terms
                ):
                    located = True
                    break
            if not located:
                raise EvaluationDataError(
                    f"case {case.id}: 第 {index + 1} 个 evidence group "
                    f"无法在语料 {evidence.document_id} 中定位"
                )


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a corpus manifest and return its raw documents list."""
    manifest_path = Path(manifest_path)
    _require(manifest_path.is_file(), f"manifest 文件不存在：{manifest_path}")
    try:
        payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError("manifest 不是合法的 JSON") from exc
    _require(
        isinstance(payload, dict) and isinstance(payload.get("documents"), list),
        "manifest 必须包含 documents 列表",
    )
    return payload
