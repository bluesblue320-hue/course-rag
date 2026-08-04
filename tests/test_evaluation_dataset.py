"""Test dataset parsing, strict validation, and the shipped dataset itself."""

import json
from pathlib import Path

import pytest

from src.documents import ChunkRecord
from src.evaluation.corpus import load_corpus, load_manifest
from src.evaluation.dataset import (
    count_by,
    load_dataset,
    split_cases,
    validate_dataset_against_corpus,
)
from src.exceptions import DatasetValidationError
from tests.evaluation_helpers import (
    MINI_DATASET_ROWS,
    make_case,
    make_expectation,
    write_mini_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPOSITORY_ROOT / "eval" / "dataset.jsonl"
MANIFEST_PATH = REPOSITORY_ROOT / "eval" / "corpus_manifest.json"


def _write(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "x-001",
        "split": "calibration",
        "question": "一个问题？",
        "answerable": True,
        "category": "direct",
        "difficulty": "easy",
        "expected_evidence": [
            {
                "document_id": "doc-a",
                "page_number": None,
                "required_terms": ["关键短语"],
            }
        ],
        "notes": "",
    }
    row.update(overrides)
    return row


def test_load_dataset_parses_a_valid_file(tmp_path: Path) -> None:
    cases = load_dataset(_write(tmp_path, [_valid_row()]))
    assert len(cases) == 1
    assert cases[0].id == "x-001"
    assert cases[0].expected_evidence[0].required_terms == ("关键短语",)


def test_load_dataset_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps(_valid_row(), ensure_ascii=False) + "\n\n   \n",
        encoding="utf-8",
    )
    assert len(load_dataset(path)) == 1


def test_load_dataset_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError):
        load_dataset(tmp_path / "missing.jsonl")


def test_load_dataset_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_load_dataset_rejects_broken_json(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_load_dataset_rejects_non_object_line(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    rows = [_valid_row(), _valid_row(question="另一个问题？")]
    with pytest.raises(DatasetValidationError, match="id 重复"):
        load_dataset(_write(tmp_path, rows))


@pytest.mark.parametrize(
    "overrides",
    [
        {"question": "   "},
        {"split": "train"},
        {"category": "unknown"},
        {"difficulty": "impossible"},
        {"answerable": "yes"},
        {"id": ""},
        {"notes": 1},
        {"expected_evidence": {}},
    ],
)
def test_load_dataset_rejects_invalid_fields(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(DatasetValidationError):
        load_dataset(_write(tmp_path, [_valid_row(**overrides)]))


def test_load_dataset_rejects_missing_field(tmp_path: Path) -> None:
    row = _valid_row()
    del row["notes"]
    with pytest.raises(DatasetValidationError, match="缺少字段"):
        load_dataset(_write(tmp_path, [row]))


def test_load_dataset_rejects_answerable_without_evidence(tmp_path: Path) -> None:
    row = _valid_row(expected_evidence=[])
    with pytest.raises(DatasetValidationError, match="必须提供 expected_evidence"):
        load_dataset(_write(tmp_path, [row]))


def test_load_dataset_rejects_unanswerable_with_evidence(tmp_path: Path) -> None:
    row = _valid_row(answerable=False, category="out_of_scope_far")
    with pytest.raises(DatasetValidationError, match="不允许提供 expected_evidence"):
        load_dataset(_write(tmp_path, [row]))


def test_load_dataset_rejects_unanswerable_with_in_scope_category(
    tmp_path: Path,
) -> None:
    row = _valid_row(answerable=False, expected_evidence=[], category="direct")
    with pytest.raises(DatasetValidationError, match="out_of_scope"):
        load_dataset(_write(tmp_path, [row]))


def test_load_dataset_rejects_answerable_out_of_scope_category(
    tmp_path: Path,
) -> None:
    row = _valid_row(category="out_of_scope_near")
    with pytest.raises(DatasetValidationError, match="必须是 answerable=false"):
        load_dataset(_write(tmp_path, [row]))


def test_load_dataset_rejects_multi_evidence_with_one_group(tmp_path: Path) -> None:
    row = _valid_row(category="multi_evidence")
    with pytest.raises(DatasetValidationError, match="至少需要两个"):
        load_dataset(_write(tmp_path, [row]))


@pytest.mark.parametrize(
    "evidence",
    [
        [{"document_id": "", "page_number": None, "required_terms": ["x"]}],
        [{"document_id": "doc-a", "page_number": 0, "required_terms": ["x"]}],
        [{"document_id": "doc-a", "page_number": True, "required_terms": ["x"]}],
        [{"document_id": "doc-a", "page_number": None, "required_terms": []}],
        [{"document_id": "doc-a", "page_number": None, "required_terms": [" "]}],
        [{"document_id": "doc-a", "page_number": None}],
        ["not-an-object"],
    ],
)
def test_load_dataset_rejects_invalid_evidence(
    tmp_path: Path,
    evidence: list[object],
) -> None:
    with pytest.raises(DatasetValidationError):
        load_dataset(_write(tmp_path, [_valid_row(expected_evidence=evidence)]))


def test_load_dataset_rejects_unknown_document_id(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="未知 document_id"):
        load_dataset(
            _write(tmp_path, [_valid_row()]),
            known_document_ids=frozenset({"doc-b"}),
        )


def test_load_dataset_error_message_carries_the_line_number(
    tmp_path: Path,
) -> None:
    rows = [_valid_row(), _valid_row(id="x-002", split="train")]
    with pytest.raises(DatasetValidationError, match="第 2 行"):
        load_dataset(_write(tmp_path, rows))


def _chunk(text: str, document_id: str = "doc-a") -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id=document_id,
        filename="doc-a.md",
        text=text,
        chunk_index=0,
        page_number=None,
    )


def test_validate_dataset_against_corpus_accepts_locatable_evidence() -> None:
    cases = (make_case(expected_evidence=(make_expectation(),)),)
    validate_dataset_against_corpus(cases, (_chunk("这里出现了关键短语"),))


def test_validate_dataset_against_corpus_rejects_empty_corpus() -> None:
    with pytest.raises(DatasetValidationError, match="没有任何 Chunk"):
        validate_dataset_against_corpus((make_case(),), ())


def test_validate_dataset_against_corpus_rejects_unknown_document() -> None:
    cases = (make_case(expected_evidence=(make_expectation(document_id="doc-z"),)),)
    with pytest.raises(DatasetValidationError, match="不存在的 document_id"):
        validate_dataset_against_corpus(cases, (_chunk("关键短语"),))


def test_validate_dataset_against_corpus_rejects_unlocatable_terms() -> None:
    cases = (
        make_case(expected_evidence=(make_expectation(required_terms=("缺失短语",)),)),
    )
    with pytest.raises(DatasetValidationError, match="找不到匹配的"):
        validate_dataset_against_corpus(cases, (_chunk("这里只有关键短语"),))


def test_validate_dataset_against_corpus_rejects_page_mismatch() -> None:
    cases = (make_case(expected_evidence=(make_expectation(page_number=2),)),)
    with pytest.raises(DatasetValidationError):
        validate_dataset_against_corpus(cases, (_chunk("关键短语"),))


def test_validate_dataset_against_corpus_skips_unanswerable_cases() -> None:
    cases = (make_case(case_id="u-1", answerable=False, category="out_of_scope_far"),)
    validate_dataset_against_corpus(cases, (_chunk("与问题无关的正文"),))


def test_count_by_returns_sorted_counts() -> None:
    cases = (
        make_case(case_id="a", category="direct"),
        make_case(case_id="b", category="paraphrase"),
        make_case(case_id="c", category="direct"),
    )
    assert count_by(cases, "category") == {"direct": 2, "paraphrase": 1}


def test_split_cases_filters_and_preserves_order() -> None:
    cases = (
        make_case(case_id="a", split="calibration"),
        make_case(case_id="b", split="test"),
        make_case(case_id="c", split="calibration"),
    )
    assert [case.id for case in split_cases(cases, "calibration")] == ["a", "c"]


def test_split_cases_rejects_unknown_split() -> None:
    with pytest.raises(DatasetValidationError):
        split_cases((make_case(),), "train")


def test_write_mini_dataset_round_trips(tmp_path: Path) -> None:
    path = write_mini_dataset(tmp_path)
    assert len(load_dataset(path)) == len(MINI_DATASET_ROWS)


def test_shipped_dataset_matches_the_planned_distribution() -> None:
    cases = load_dataset(DATASET_PATH)
    assert len(cases) == 60
    assert count_by(cases, "split") == {"calibration": 42, "test": 18}
    assert count_by(cases, "category") == {
        "direct": 15,
        "multi_evidence": 6,
        "out_of_scope_far": 12,
        "out_of_scope_near": 12,
        "paraphrase": 15,
    }
    assert sum(1 for case in cases if case.answerable) == 36
    assert sum(1 for case in cases if not case.answerable) == 24


def test_shipped_dataset_has_both_splits_in_every_category() -> None:
    cases = load_dataset(DATASET_PATH)
    for category in ("direct", "paraphrase", "multi_evidence"):
        splits = {case.split for case in cases if case.category == category}
        assert splits == {"calibration", "test"}


def test_shipped_dataset_evidence_exists_in_the_shipped_corpus() -> None:
    documents = load_manifest(MANIFEST_PATH, REPOSITORY_ROOT)
    corpus = load_corpus(documents)
    cases = load_dataset(DATASET_PATH, corpus.document_ids)
    validate_dataset_against_corpus(cases, corpus.chunks)
