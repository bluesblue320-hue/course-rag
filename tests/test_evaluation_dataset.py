"""Offline tests for evaluation dataset loading and validation."""

import json
from pathlib import Path

import pytest

from src.evaluation.dataset import (
    load_dataset,
    validate_dataset_document_ids,
    validate_dataset_integrity,
)
from src.evaluation.models import EvaluationDataError

from evaluation_helpers import make_case, write_corpus, write_dataset


def _base_case(**overrides: object) -> dict[str, object]:
    case = make_case("q001")
    case.update(overrides)
    return case


def test_loads_a_valid_dataset(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path,
        [
            make_case("q001"),
            make_case("q002", question="beta 数据库是什么？", evidence=[("eval-doc-b", ("beta",))]),
        ],
    )

    cases = load_dataset(path)

    assert [case.id for case in cases] == ["q001", "q002"]
    assert cases[0].split == "calibration"
    assert cases[0].expected_evidence[0].required_terms == ("alpha",)


def test_rejects_empty_dataset(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="空"):
        load_dataset(path)


def test_rejects_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("{ not json\n", encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="JSON"):
        load_dataset(path)


def test_rejects_non_object_line(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text('["array"]\n', encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="对象"):
        load_dataset(path)


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [make_case("q001"), make_case("q001")])

    with pytest.raises(EvaluationDataError, match="重复"):
        load_dataset(path)


def test_rejects_empty_question(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [_base_case(question="   ")])

    with pytest.raises(EvaluationDataError, match="question"):
        load_dataset(path)


def test_rejects_invalid_split(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [_base_case(split="training")])

    with pytest.raises(EvaluationDataError, match="split"):
        load_dataset(path)


def test_rejects_invalid_category(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [_base_case(category="unknown")])

    with pytest.raises(EvaluationDataError, match="category"):
        load_dataset(path)


def test_rejects_invalid_difficulty(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [_base_case(difficulty="expert")])

    with pytest.raises(EvaluationDataError, match="difficulty"):
        load_dataset(path)


def test_rejects_answerable_without_evidence(tmp_path: Path) -> None:
    case = make_case("q001")
    case["expected_evidence"] = []
    path = write_dataset(tmp_path, [case])

    with pytest.raises(EvaluationDataError, match="answerable"):
        load_dataset(path)


def test_rejects_unanswerable_with_evidence(tmp_path: Path) -> None:
    case = make_case("q001", answerable=False)
    case["expected_evidence"] = [
        {"document_id": "eval-doc-a", "page_number": None, "required_terms": ["alpha"]}
    ]
    path = write_dataset(tmp_path, [case])

    with pytest.raises(EvaluationDataError, match="answerable"):
        load_dataset(path)


def test_rejects_unknown_document_id(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path,
        [make_case("q001", evidence=[("eval-missing", ("alpha",))])],
    )
    cases = load_dataset(path)

    with pytest.raises(EvaluationDataError, match="document_id"):
        validate_dataset_document_ids(cases, {"eval-doc-a"})


def test_rejects_empty_required_terms(tmp_path: Path) -> None:
    case = _base_case()
    case["expected_evidence"] = [
        {"document_id": "eval-doc-a", "page_number": None, "required_terms": []}
    ]
    path = write_dataset(tmp_path, [case])

    with pytest.raises(EvaluationDataError, match="required_terms"):
        load_dataset(path)


def test_rejects_blank_required_term(tmp_path: Path) -> None:
    case = _base_case()
    case["expected_evidence"] = [
        {
            "document_id": "eval-doc-a",
            "page_number": None,
            "required_terms": ["alpha", "   "],
        }
    ]
    path = write_dataset(tmp_path, [case])

    with pytest.raises(EvaluationDataError, match="required_terms"):
        load_dataset(path)


def test_rejects_zero_page_number(tmp_path: Path) -> None:
    case = _base_case()
    case["expected_evidence"] = [
        {"document_id": "eval-doc-a", "page_number": 0, "required_terms": ["alpha"]}
    ]
    path = write_dataset(tmp_path, [case])

    with pytest.raises(EvaluationDataError, match="page_number"):
        load_dataset(path)


def test_rejects_multi_evidence_category_with_one_group(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path,
        [
            _base_case(
                category="multi_evidence",
                evidence=[("eval-doc-a", ("alpha",))],
            )
        ],
    )

    with pytest.raises(EvaluationDataError, match="evidence group"):
        load_dataset(path)


def test_integrity_fails_when_evidence_is_not_located(tmp_path: Path) -> None:
    manifest = write_corpus(
        tmp_path,
        {"eval-doc-a": ["alpha 内容"], "eval-doc-b": ["beta 内容"]},
    )
    from src.evaluation.corpus import load_corpus

    chunks = load_corpus(manifest, base_dir=tmp_path)
    cases = load_dataset(
        write_dataset(
            tmp_path,
            [make_case("q001", evidence=[("eval-doc-b", ("不存在的词",))])],
        )
    )

    with pytest.raises(EvaluationDataError, match="无法在语料"):
        validate_dataset_integrity(cases, chunks)


def test_integrity_passes_when_evidence_is_located(tmp_path: Path) -> None:
    manifest = write_corpus(
        tmp_path,
        {"eval-doc-a": ["alpha 服务内容"], "eval-doc-b": ["beta 数据库内容"]},
    )
    from src.evaluation.corpus import load_corpus

    chunks = load_corpus(manifest, base_dir=tmp_path)
    cases = load_dataset(
        write_dataset(
            tmp_path,
            [make_case("q001"), make_case("q002", evidence=[("eval-doc-b", ("beta",))])],
        )
    )

    validate_dataset_integrity(cases, chunks)
    validate_dataset_document_ids(cases, set(chunks.keys()))


def test_multi_evidence_case_parses_both_groups(tmp_path: Path) -> None:
    case = make_case(
        "q001",
        category="multi_evidence",
        evidence=[("eval-doc-a", ("alpha",)), ("eval-doc-b", ("beta",))],
    )
    path = write_dataset(tmp_path, [case])

    cases = load_dataset(path)

    assert len(cases[0].expected_evidence) == 2
    assert [group.document_id for group in cases[0].expected_evidence] == [
        "eval-doc-a",
        "eval-doc-b",
    ]


def test_manifest_validation_rejects_absolute_and_traversal_paths(
    tmp_path: Path,
) -> None:
    from src.evaluation.corpus import load_corpus

    outside = tmp_path.parent / "outside.md"
    outside.write_text("alpha 内容", encoding="utf-8")

    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "evil",
                        "path": str(outside),
                        "filename": "outside.md",
                        "content_type": "text/markdown",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDataError, match="绝对路径"):
        load_corpus(manifest_path)

    traversal_manifest = tmp_path / "corpus_manifest.json"
    traversal_manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "evil",
                        "path": "../outside.md",
                        "filename": "outside.md",
                        "content_type": "text/markdown",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDataError, match="越出"):
        load_corpus(traversal_manifest)
