"""Load and strictly validate answer-quality annotations.

Annotations are validated jointly against the retrieval dataset so that
case coverage, fact/answerable consistency, evidence mappings, and
reference-answer integrity are all checked in one pass.  Errors always
include a line number and never an absolute filesystem path.
"""

import json
from pathlib import Path
from typing import Any

from src.evaluation.answer_metrics import phrase_appears
from src.evaluation.answer_models import AnswerAnnotation, RequiredFact
from src.evaluation.models import EvaluationCase
from src.exceptions import AnswerAnnotationValidationError
from src.rag_service import INSUFFICIENT_CONTEXT_ANSWER

_REQUIRED_ANNOTATION_FIELDS = (
    "case_id",
    "reference_answer",
    "required_facts",
    "forbidden_phrases",
    "notes",
)
_REQUIRED_FACT_FIELDS = (
    "fact_id",
    "accepted_phrases",
    "supporting_evidence_indexes",
)

_REFERENCE_ANSWER_HINT = "reference_answer 供人工审计，不参与语义相似度评分。"


def _fail(line_number: int, message: str) -> AnswerAnnotationValidationError:
    return AnswerAnnotationValidationError(f"标注文件第 {line_number} 行: {message}")


def _parse_required_fact(
    raw: object,
    line_number: int,
    reference_answer: str,
    evidence_count: int,
) -> RequiredFact:
    if not isinstance(raw, dict):
        raise _fail(line_number, "required_facts 的元素必须是对象")
    for field in _REQUIRED_FACT_FIELDS:
        if field not in raw:
            raise _fail(line_number, f"required_fact 缺少字段: {field}")

    fact_id = raw["fact_id"]
    if not isinstance(fact_id, str) or not fact_id.strip():
        raise _fail(line_number, "fact_id 必须是非空字符串")

    accepted_phrases = raw["accepted_phrases"]
    if not isinstance(accepted_phrases, list) or not accepted_phrases:
        raise _fail(line_number, "accepted_phrases 必须是非空数组")
    parsed_phrases: list[str] = []
    for phrase in accepted_phrases:
        if not isinstance(phrase, str) or not phrase.strip():
            raise _fail(line_number, "accepted_phrases 不允许空字符串")
        cleaned_phrase = phrase.strip()
        if cleaned_phrase in parsed_phrases:
            raise _fail(line_number, f"accepted_phrases 重复: {cleaned_phrase}")
        parsed_phrases.append(cleaned_phrase)

    indexes = raw["supporting_evidence_indexes"]
    if not isinstance(indexes, list) or not indexes:
        raise _fail(line_number, "supporting_evidence_indexes 必须是非空数组")
    parsed_indexes: list[int] = []
    for index in indexes:
        if isinstance(index, bool) or not isinstance(index, int):
            raise _fail(line_number, "supporting_evidence_indexes 必须是整数")
        if index < 1:
            raise _fail(line_number, "supporting_evidence_indexes 不得小于 1")
        if index > evidence_count:
            raise _fail(
                line_number,
                f"supporting_evidence_indexes 越界: {index}（共 {evidence_count} 组证据）",
            )
        if index in parsed_indexes:
            raise _fail(line_number, f"supporting_evidence_indexes 重复: {index}")
        parsed_indexes.append(index)

    return RequiredFact(
        fact_id=fact_id.strip(),
        accepted_phrases=tuple(parsed_phrases),
        supporting_evidence_indexes=tuple(parsed_indexes),
    )


def _parse_annotation(
    raw: object,
    line_number: int,
    case: EvaluationCase,
) -> AnswerAnnotation:
    if not isinstance(raw, dict):
        raise _fail(line_number, "每一行必须是 JSON 对象")
    for field in _REQUIRED_ANNOTATION_FIELDS:
        if field not in raw:
            raise _fail(line_number, f"缺少字段: {field}")

    case_id = raw["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise _fail(line_number, "case_id 必须是非空字符串")
    case_id = case_id.strip()
    if case_id != case.id:
        raise _fail(
            line_number,
            f"case_id 与数据集不一致: 标注 {case_id!r}，数据集 {case.id!r}",
        )

    reference_answer = raw["reference_answer"]
    if not isinstance(reference_answer, str) or not reference_answer.strip():
        raise _fail(line_number, "reference_answer 不允许为空")
    reference_answer = reference_answer.strip()

    raw_facts = raw["required_facts"]
    if not isinstance(raw_facts, list):
        raise _fail(line_number, "required_facts 必须是数组")
    if case.answerable and not raw_facts:
        raise _fail(line_number, "answerable=true 的题目必须提供 required_facts")
    if not case.answerable and raw_facts:
        raise _fail(line_number, "answerable=false 的题目不允许提供 required_facts")

    evidence_count = len(case.expected_evidence)
    facts: list[RequiredFact] = []
    seen_fact_ids: set[str] = set()
    for raw_fact in raw_facts:
        fact = _parse_required_fact(
            raw_fact,
            line_number,
            reference_answer,
            evidence_count,
        )
        if fact.fact_id in seen_fact_ids:
            raise _fail(line_number, f"fact_id 在 case 内重复: {fact.fact_id}")
        seen_fact_ids.add(fact.fact_id)
        facts.append(fact)

    raw_forbidden = raw["forbidden_phrases"]
    if not isinstance(raw_forbidden, list):
        raise _fail(line_number, "forbidden_phrases 必须是数组")
    forbidden_phrases: list[str] = []
    for phrase in raw_forbidden:
        if not isinstance(phrase, str) or not phrase.strip():
            raise _fail(line_number, "forbidden_phrases 不允许空字符串")
        cleaned_phrase = phrase.strip()
        if cleaned_phrase in forbidden_phrases:
            raise _fail(line_number, f"forbidden_phrases 重复: {cleaned_phrase}")
        forbidden_phrases.append(cleaned_phrase)

    notes = raw["notes"]
    if not isinstance(notes, str):
        raise _fail(line_number, "notes 必须是字符串")

    return AnswerAnnotation(
        case_id=case_id,
        reference_answer=reference_answer,
        required_facts=tuple(facts),
        forbidden_phrases=tuple(forbidden_phrases),
        notes=notes,
    )


def _validate_reference_answer_integrity(
    annotation: AnswerAnnotation,
    line_number: int,
) -> None:
    """Check the reference answer satisfies the annotated fact phrases.

    This is an annotation-integrity guard, not a semantic similarity score:
    it only requires at least one accepted phrase per fact to appear in the
    normalized reference answer so typos and empty phrases fail fast.
    """
    for fact in annotation.required_facts:
        covered = any(
            phrase_appears(phrase, annotation.reference_answer)
            for phrase in fact.accepted_phrases
        )
        if not covered:
            raise _fail(
                line_number,
                f"fact {fact.fact_id} 的任一 accepted_phrase 未出现在 reference_answer 中",
            )

    for phrase in annotation.forbidden_phrases:
        if phrase_appears(phrase, annotation.reference_answer):
            raise _fail(
                line_number,
                f"forbidden_phrase 出现在 reference_answer 中: {phrase}",
            )


def _validate_evidence_coverage(
    annotation: AnswerAnnotation,
    evidence_count: int,
    line_number: int,
) -> None:
    """Check every expected evidence group is referenced by at least one fact."""
    covered_indexes: set[int] = set()
    for fact in annotation.required_facts:
        covered_indexes.update(fact.supporting_evidence_indexes)
    missing = sorted(
        index
        for index in range(1, evidence_count + 1)
        if index not in covered_indexes
    )
    if missing:
        raise _fail(
            line_number,
            f"expected_evidence 未被任何 fact 引用: {missing}",
        )


def load_answer_annotations(
    annotations_path: Path,
    cases: tuple[EvaluationCase, ...],
) -> tuple[AnswerAnnotation, ...]:
    """Parse annotations and validate them jointly against the dataset.

    The returned annotations follow dataset order.  Errors include the JSONL
    line number and never an absolute path.
    """
    annotations_path = Path(annotations_path)
    if not annotations_path.is_file():
        raise AnswerAnnotationValidationError("标注文件不存在")

    try:
        raw_text = annotations_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AnswerAnnotationValidationError("标注文件无法以 UTF-8 读取") from exc

    cases_by_id = {case.id: case for case in cases}
    parsed: list[tuple[int, AnswerAnnotation]] = []
    seen_case_ids: set[str] = set()
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
        if case_id in seen_case_ids:
            raise _fail(line_number, f"case_id 重复: {case_id}")
        seen_case_ids.add(case_id)

        case = cases_by_id.get(case_id)
        if case is None:
            raise _fail(
                line_number,
                f"标注引用了数据集中不存在的 case_id: {case_id}",
            )

        annotation = _parse_annotation(payload, line_number, case)
        if not case.answerable:
            if annotation.reference_answer != INSUFFICIENT_CONTEXT_ANSWER:
                raise _fail(
                    line_number,
                    "unanswerable 题目的 reference_answer 必须等于生产固定拒答文案",
                )
        _validate_reference_answer_integrity(annotation, line_number)
        _validate_evidence_coverage(
            annotation,
            len(case.expected_evidence),
            line_number,
        )
        parsed.append((line_number, annotation))

    dataset_ids = [case.id for case in cases]
    annotated_ids = [annotation.case_id for _, annotation in parsed]
    if sorted(dataset_ids) != sorted(annotated_ids):
        missing = sorted(set(dataset_ids) - set(annotated_ids))
        extra = sorted(set(annotated_ids) - set(dataset_ids))
        if missing:
            raise AnswerAnnotationValidationError(
                f"标注缺少 case: {missing}"
            )
        if extra:
            raise AnswerAnnotationValidationError(
                f"标注包含多余 case: {extra}"
            )

    by_id = {annotation.case_id: annotation for _, annotation in parsed}
    return tuple(by_id[case_id] for case_id in dataset_ids)


def validate_evidence_coverage_with_count(
    annotation: AnswerAnnotation,
    evidence_count: int,
) -> None:
    """Validate evidence coverage against an explicit evidence count."""
    covered_indexes: set[int] = set()
    for fact in annotation.required_facts:
        covered_indexes.update(fact.supporting_evidence_indexes)
    missing = sorted(
        index
        for index in range(1, evidence_count + 1)
        if index not in covered_indexes
    )
    if missing:
        raise AnswerAnnotationValidationError(
            f"case {annotation.case_id} 的 expected_evidence 未被任何 fact 引用: {missing}"
        )
