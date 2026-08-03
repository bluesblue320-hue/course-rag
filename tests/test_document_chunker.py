"""Offline tests for chunk records carrying document metadata."""

import numpy as np

from src.document_chunker import chunk_document
from src.documents import LoadedDocument, LoadedPage


def _make_document(pages: list[tuple[int | None, str]]) -> LoadedDocument:
    loaded_pages = tuple(
        LoadedPage(page_number=page_number, text=text)
        for page_number, text in pages
    )
    return LoadedDocument(
        pages=loaded_pages,
        text_length=sum(len(text) for _, text in pages),
    )


def test_txt_chunks_have_null_page_number() -> None:
    document = _make_document([(None, "x" * 700)])

    chunks = chunk_document(document, "doc-1", "notes.txt")

    assert len(chunks) == 3
    assert all(chunk.page_number is None for chunk in chunks)


def test_markdown_chunks_have_null_page_number() -> None:
    document = _make_document([(None, "# 标题\n\n" + "y" * 700)])

    chunks = chunk_document(document, "doc-2", "notes.md")

    assert len(chunks) == 3
    assert all(chunk.page_number is None for chunk in chunks)


def test_pdf_chunks_keep_page_numbers() -> None:
    document = _make_document([(1, "z" * 700), (2, "w" * 700)])

    chunks = chunk_document(document, "doc-3", "course.pdf")

    assert [chunk.page_number for chunk in chunks] == [1, 1, 1, 2, 2, 2]


def test_chunk_indexes_are_continuous_within_a_document() -> None:
    document = _make_document([(1, "a" * 700), (2, "b" * 700)])

    chunks = chunk_document(document, "doc-4", "course.pdf")

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3, 4, 5]


def test_chunk_ids_are_unique() -> None:
    document = _make_document([(1, "a" * 700), (2, "b" * 700)])

    chunks = chunk_document(document, "doc-5", "course.pdf")

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_empty_pages_do_not_produce_chunks() -> None:
    document = _make_document([(1, "   \n "), (2, "only this page")])

    chunks = chunk_document(document, "doc-6", "course.pdf")

    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].filename == "course.pdf"
    assert chunks[0].document_id == "doc-6"
    assert chunks[0].text == "only this page"


def test_chunk_texts_are_trimmed() -> None:
    document = _make_document([(None, "  \n\n  leading text  \n\n  ")])

    chunks = chunk_document(document, "doc-7", "notes.txt")

    assert chunks[0].text == chunks[0].text.strip()


def test_chunk_records_are_frozen_and_numpy_compatible() -> None:
    document = _make_document([(None, "numpy vector")])
    chunks = chunk_document(document, "doc-8", "notes.txt")

    assert np.asarray([chunk.text for chunk in chunks]).shape[0] == 1
