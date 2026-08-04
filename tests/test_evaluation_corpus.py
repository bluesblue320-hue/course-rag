"""Test manifest parsing, path safety, and corpus chunking."""

import json
from pathlib import Path

import pytest

from src.evaluation.corpus import (
    build_default_loaders,
    load_corpus,
    load_manifest,
)
from src.exceptions import CorpusValidationError
from tests.evaluation_helpers import write_mini_corpus

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPOSITORY_ROOT / "eval" / "corpus_manifest.json"


def _write_manifest(root: Path, documents: object) -> Path:
    manifest_path = root / "eval" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"documents": documents}, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def test_build_default_loaders_covers_the_production_extensions() -> None:
    assert set(build_default_loaders()) == {".txt", ".md", ".pdf"}


def test_load_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    manifest_path = write_mini_corpus(tmp_path)
    documents = load_manifest(manifest_path, tmp_path)
    assert [document.document_id for document in documents] == [
        "mini-alpha",
        "mini-beta",
    ]
    assert documents[0].relative_path == "eval/corpus/alpha.md"
    assert documents[0].resolved_path.is_file()


def test_load_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CorpusValidationError, match="不存在"):
        load_manifest(tmp_path / "nope.json", tmp_path)


def test_load_manifest_rejects_broken_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "eval" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{oops", encoding="utf-8")
    with pytest.raises(CorpusValidationError, match="合法 JSON"):
        load_manifest(manifest_path, tmp_path)


def test_load_manifest_rejects_non_object_root(tmp_path: Path) -> None:
    manifest_path = tmp_path / "eval" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("[]", encoding="utf-8")
    with pytest.raises(CorpusValidationError, match="顶层"):
        load_manifest(manifest_path, tmp_path)


def test_load_manifest_rejects_empty_documents(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [])
    with pytest.raises(CorpusValidationError, match="非空的 documents"):
        load_manifest(manifest_path, tmp_path)


def test_load_manifest_rejects_missing_entry_field(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [{"document_id": "a", "path": "eval/corpus/alpha.md", "filename": "a.md"}],
    )
    with pytest.raises(CorpusValidationError, match="缺少字段"):
        load_manifest(manifest_path, tmp_path)


def test_load_manifest_rejects_duplicate_document_id(tmp_path: Path) -> None:
    write_mini_corpus(tmp_path)
    entry = {
        "document_id": "dup",
        "path": "eval/corpus/alpha.md",
        "filename": "alpha.md",
        "content_type": "text/markdown",
    }
    manifest_path = _write_manifest(tmp_path, [entry, dict(entry)])
    with pytest.raises(CorpusValidationError, match="重复"):
        load_manifest(manifest_path, tmp_path)


def test_load_manifest_rejects_absolute_path(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "document_id": "a",
                "path": "/etc/passwd",
                "filename": "passwd",
                "content_type": "text/plain",
            }
        ],
    )
    with pytest.raises(CorpusValidationError, match="绝对路径"):
        load_manifest(manifest_path, tmp_path)


def test_load_manifest_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "document_id": "a",
                "path": "eval/../../secret.md",
                "filename": "secret.md",
                "content_type": "text/markdown",
            }
        ],
    )
    with pytest.raises(CorpusValidationError, match=r"\.\."):
        load_manifest(manifest_path, tmp_path)


def test_load_manifest_rejects_path_outside_the_eval_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "outside.md").write_text("外部文件", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "document_id": "a",
                "path": "outside.md",
                "filename": "outside.md",
                "content_type": "text/markdown",
            }
        ],
    )
    with pytest.raises(CorpusValidationError, match="必须位于评估目录内"):
        load_manifest(manifest_path, tmp_path)


def test_load_manifest_rejects_missing_corpus_file(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "document_id": "a",
                "path": "eval/corpus/ghost.md",
                "filename": "ghost.md",
                "content_type": "text/markdown",
            }
        ],
    )
    with pytest.raises(CorpusValidationError, match="语料文件不存在"):
        load_manifest(manifest_path, tmp_path)


def test_load_corpus_chunks_with_the_production_pipeline(tmp_path: Path) -> None:
    documents = load_manifest(write_mini_corpus(tmp_path), tmp_path)
    corpus = load_corpus(documents)
    assert corpus.document_ids == frozenset({"mini-alpha", "mini-beta"})
    assert len(corpus.chunks) >= 2
    assert all(chunk.page_number is None for chunk in corpus.chunks)
    assert {chunk.document_id for chunk in corpus.chunks} == corpus.document_ids


def test_load_corpus_rejects_empty_document_list() -> None:
    with pytest.raises(CorpusValidationError, match="至少一个文档"):
        load_corpus(())


def test_load_corpus_rejects_unsupported_extension(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "eval" / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "note.rtf").write_text("正文", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "document_id": "a",
                "path": "eval/corpus/note.rtf",
                "filename": "note.rtf",
                "content_type": "application/rtf",
            }
        ],
    )
    documents = load_manifest(manifest_path, tmp_path)
    with pytest.raises(CorpusValidationError, match="不受支持"):
        load_corpus(documents)


def test_load_corpus_reports_unparsable_documents(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "eval" / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "blank.md").write_text("   \n\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "document_id": "a",
                "path": "eval/corpus/blank.md",
                "filename": "blank.md",
                "content_type": "text/markdown",
            }
        ],
    )
    documents = load_manifest(manifest_path, tmp_path)
    with pytest.raises(CorpusValidationError):
        load_corpus(documents)


def test_shipped_corpus_produces_enough_chunks_for_top_five() -> None:
    corpus = load_corpus(load_manifest(MANIFEST_PATH, REPOSITORY_ROOT))
    assert len(corpus.documents) == 3
    assert len(corpus.chunks) > 5
    for document_id in corpus.document_ids:
        count = sum(1 for chunk in corpus.chunks if chunk.document_id == document_id)
        assert count >= 5
