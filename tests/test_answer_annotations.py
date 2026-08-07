"""Tests for answer annotation loading and strict validation."""

import json
from pathlib import Path

import pytest

from src.evaluation.answer_annotations import (
    load_answer_annotations,
    validate_evidence_coverage_with_count,
)
from src.evaluation.answer_models import AnswerAnnotation, RequiredFact
from src.exceptions import AnswerAnnotationValidationError
from tests.evaluation_helpers import make_case


def _write_annotations(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _annotation_row(
    case_id: str = "c-001",
    reference_answer: str = "事实陈述。",
    required_facts: list[dict[str, object]] | None = None,
    forbidden_phrases: list[str] | None = None,
    notes: str = "",
) -> dict[str, object]:
    if required_facts is None:
        required_facts = [
            {
                "fact_id": "fact-one",
                "accepted_phrases": ["事实陈述"],
                "supporting_evidence_indexes": [1],
            }
        ]
    return {
        "case_id": case_id,
        "reference_answer": reference_answer,
        "required_facts": required_facts,
        "forbidden_phrases": forbidden_phrases or [],
        "notes": notes,
    }


class TestLoadAnswerAnnotations:
    def test_full_valid_file_returns_cases_in_dataset_order(
        self, tmp_path: Path
    ) -> None:
        cases = (
            make_case("a-001", answerable=True),
            make_case("a-002", answerable=False),
        )
        rows = [
            _annotation_row("a-001"),
            _annotation_row(
                "a-002",
                reference_answer=(
                    "当前课程资料中没有足够信息回答这个问题。请尝试换一种问法，"
                    "或切换到语义检索查看最接近的课程原文。"
                ),
                required_facts=[],
            ),
        ]
        path = _write_annotations(tmp_path / "annotations.jsonl", rows)
        annotations = load_answer_annotations(path, cases)
        assert [annotation.case_id for annotation in annotations] == [
            "a-001",
            "a-002",
        ]

    def test_annotation_ids_must_cover_all_dataset_ids(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        path = _write_annotations(
            tmp_path / "annotations.jsonl",
            [_annotation_row("a-999")],
        )
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "数据集中不存在" in str(exc.value)

    def test_missing_case_is_rejected(self, tmp_path: Path) -> None:
        cases = (
            make_case("a-001", answerable=True),
            make_case("a-002", answerable=False),
        )
        path = _write_annotations(
            tmp_path / "annotations.jsonl",
            [_annotation_row("a-001")],
        )
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "缺少 case" in str(exc.value)

    def test_extra_case_is_rejected(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        path = _write_annotations(
            tmp_path / "annotations.jsonl",
            [_annotation_row("a-001"), _annotation_row("a-002")],
        )
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "数据集中不存在" in str(exc.value)

    def test_duplicate_case_id_is_rejected(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        path = _write_annotations(
            tmp_path / "annotations.jsonl",
            [_annotation_row("a-001"), _annotation_row("a-001")],
        )
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "重复" in str(exc.value)

    def test_answerable_without_required_facts_is_rejected(
        self, tmp_path: Path
    ) -> None:
        cases = (make_case("a-001", answerable=True),)
        row = _annotation_row("a-001", required_facts=[])
        path = _write_annotations(tmp_path / "annotations.jsonl", [row])
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "required_facts" in str(exc.value)

    def test_unanswerable_with_required_facts_is_rejected(
        self, tmp_path: Path
    ) -> None:
        cases = (make_case("a-001", answerable=False),)
        row = _annotation_row(
            "a-001",
            reference_answer=(
                "当前课程资料中没有足够信息回答这个问题。请尝试换一种问法，"
                "或切换到语义检索查看最接近的课程原文。"
            ),
            required_facts=[
                {
                    "fact_id": "fact-one",
                    "accepted_phrases": ["事实陈述"],
                    "supporting_evidence_indexes": [],
                }
            ],
        )
        path = _write_annotations(tmp_path / "annotations.jsonl", [row])
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "不允许提供 required_facts" in str(exc.value)

    def test_supporting_index_zero_is_rejected(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        row = _annotation_row(
            "a-001",
            required_facts=[
                {
                    "fact_id": "fact-one",
                    "accepted_phrases": ["事实陈述"],
                    "supporting_evidence_indexes": [0],
                }
            ],
        )
        path = _write_annotations(tmp_path / "annotations.jsonl", [row])
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "不得小于 1" in str(exc.value)

    def test_supporting_index_out_of_range_is_rejected(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        row = _annotation_row(
            "a-001",
            required_facts=[
                {
                    "fact_id": "fact-one",
                    "accepted_phrases": ["事实陈述"],
                    "supporting_evidence_indexes": [2],
                }
            ],
        )
        path = _write_annotations(tmp_path / "annotations.jsonl", [row])
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "越界" in str(exc.value)

    def test_duplicate_fact_id_is_rejected(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        row = _annotation_row(
            "a-001",
            required_facts=[
                {
                    "fact_id": "same",
                    "accepted_phrases": ["事实陈述"],
                    "supporting_evidence_indexes": [1],
                },
                {
                    "fact_id": "same",
                    "accepted_phrases": ["另一事实"],
                    "supporting_evidence_indexes": [1],
                },
            ],
        )
        path = _write_annotations(tmp_path / "annotations.jsonl", [row])
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "fact_id 在 case 内重复" in str(exc.value)

    def test_empty_accepted_phrase_is_rejected(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        row = _annotation_row(
            "a-001",
            required_facts=[
                {
                    "fact_id": "fact-one",
                    "accepted_phrases": [""],
                    "supporting_evidence_indexes": [1],
                }
            ],
        )
        path = _write_annotations(tmp_path / "annotations.jsonl", [row])
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "accepted_phrases 不允许空字符串" in str(exc.value)

    def test_empty_reference_answer_is_rejected(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        row = _annotation_row("a-001", reference_answer="")
        path = _write_annotations(tmp_path / "annotations.jsonl", [row])
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "reference_answer 不允许为空" in str(exc.value)

    def test_forbidden_phrase_in_reference_answer_is_rejected(
        self, tmp_path: Path
    ) -> None:
        cases = (make_case("a-001", answerable=True),)
        row = _annotation_row(
            "a-001",
            reference_answer="事实陈述，但这里写了矛盾内容。",
            forbidden_phrases=["矛盾内容"],
        )
        path = _write_annotations(tmp_path / "annotations.jsonl", [row])
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "forbidden_phrase 出现在 reference_answer" in str(exc.value)

    def test_uncovered_evidence_group_is_rejected(self, tmp_path: Path) -> None:
        cases = (
            make_case(
                "a-001",
                answerable=True,
                expected_evidence=(
                    make_expectation_helper("doc-a"),
                    make_expectation_helper("doc-b"),
                ),
            ),
        )
        row = _annotation_row("a-001")
        path = _write_annotations(tmp_path / "annotations.jsonl", [row])
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "未被任何 fact 引用" in str(exc.value)

    def test_invalid_json_reports_line_number(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        path = tmp_path / "annotations.jsonl"
        path.write_text("{not valid json}\n", encoding="utf-8")
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        assert "第 1 行" in str(exc.value)

    def test_error_never_leaks_absolute_path(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        path = _write_annotations(
            tmp_path / "annotations.jsonl",
            [_annotation_row("a-999")],
        )
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            load_answer_annotations(path, cases)
        message = str(exc.value)
        assert str(tmp_path) not in message
        # Windows drive-letter prefixes (e.g. "C:\\") must never appear.
        assert not any(f"{letter}:" in message for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        cases = (make_case("a-001", answerable=True),)
        path = tmp_path / "annotations.jsonl"
        path.write_text(
            json.dumps(_annotation_row("a-001"), ensure_ascii=False)
            + "\n\n\n",
            encoding="utf-8",
        )
        annotations = load_answer_annotations(path, cases)
        assert len(annotations) == 1


class TestValidateEvidenceCoverageWithCount:
    def test_evidence_count_validation_works(self) -> None:
        annotation = AnswerAnnotation(
            case_id="c-001",
            reference_answer="事实陈述。",
            required_facts=(
                RequiredFact(
                    fact_id="fact-one",
                    accepted_phrases=("事实陈述",),
                    supporting_evidence_indexes=(1,),
                ),
            ),
            forbidden_phrases=(),
            notes="",
        )
        validate_evidence_coverage_with_count(annotation, 1)

    def test_evidence_count_missing_group_is_rejected(self) -> None:
        annotation = AnswerAnnotation(
            case_id="c-001",
            reference_answer="事实陈述。",
            required_facts=(
                RequiredFact(
                    fact_id="fact-one",
                    accepted_phrases=("事实陈述",),
                    supporting_evidence_indexes=(1,),
                ),
            ),
            forbidden_phrases=(),
            notes="",
        )
        with pytest.raises(AnswerAnnotationValidationError) as exc:
            validate_evidence_coverage_with_count(annotation, 2)
        assert "未被任何 fact 引用" in str(exc.value)


def make_expectation_helper(document_id: str):
    from tests.evaluation_helpers import make_expectation

    return make_expectation(document_id=document_id)
