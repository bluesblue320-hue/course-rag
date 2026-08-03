"""Offline tests for the text, markdown, and PDF document loaders."""

from pathlib import Path

import pytest

from src.document_loaders import (
    MarkdownDocumentLoader,
    PdfDocumentLoader,
    TextDocumentLoader,
)
from src.exceptions import DocumentParseError, EmptyDocumentError

from pdf_helpers import make_encrypted_pdf, make_text_pdf


def test_text_loader_reads_valid_utf8_txt(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("  第一段内容\n\n第二段内容。  ", encoding="utf-8")

    loaded = TextDocumentLoader().load(path)

    assert len(loaded.pages) == 1
    assert loaded.pages[0].page_number is None
    assert loaded.pages[0].text == "第一段内容\n\n第二段内容。"
    assert loaded.text_length == len(loaded.pages[0].text)


def test_markdown_loader_reads_valid_markdown(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# 标题\n\n正文内容。", encoding="utf-8")

    loaded = MarkdownDocumentLoader().load(path)

    assert len(loaded.pages) == 1
    assert loaded.pages[0].page_number is None
    assert loaded.pages[0].text == "# 标题\n\n正文内容。"


def test_pdf_loader_reads_text_pdf_with_one_based_pages(tmp_path: Path) -> None:
    path = tmp_path / "course.pdf"
    path.write_bytes(make_text_pdf(["Page one content", "Page two content"]))

    loaded = PdfDocumentLoader().load(path)

    assert [page.page_number for page in loaded.pages] == [1, 2]
    assert loaded.pages[0].text == "Page one content"
    assert loaded.pages[1].text == "Page two content"
    assert loaded.text_length == 16 + 16


def test_pdf_loader_skips_blank_pages_but_keeps_physical_numbers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "course.pdf"
    path.write_bytes(make_text_pdf(["Page one content", "", "Page three"]))

    loaded = PdfDocumentLoader().load(path)

    assert [page.page_number for page in loaded.pages] == [1, 3]


def test_text_loader_rejects_empty_txt(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n\t  ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        TextDocumentLoader().load(path)


def test_markdown_loader_rejects_empty_markdown(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        MarkdownDocumentLoader().load(path)


def test_pdf_loader_rejects_pdf_without_text(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_text_pdf(["", " "]))

    with pytest.raises(EmptyDocumentError):
        PdfDocumentLoader().load(path)


def test_pdf_loader_rejects_corrupted_pdf(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is definitely not a pdf file")

    with pytest.raises(DocumentParseError):
        PdfDocumentLoader().load(path)


def test_pdf_loader_rejects_encrypted_pdf(tmp_path: Path) -> None:
    path = tmp_path / "locked.pdf"
    path.write_bytes(make_encrypted_pdf())

    with pytest.raises(DocumentParseError, match="加密"):
        PdfDocumentLoader().load(path)


def test_text_loader_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe\x00invalid")

    with pytest.raises(DocumentParseError):
        TextDocumentLoader().load(path)


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentParseError):
        TextDocumentLoader().load(tmp_path / "missing.txt")
