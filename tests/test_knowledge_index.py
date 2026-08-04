"""Offline tests for atomic knowledge index replacement and metadata."""

import threading

import numpy as np
import pytest

from src.documents import ChunkRecord
from src.knowledge_index import KnowledgeIndex


def _chunk(
    document_id: str,
    filename: str,
    text: str,
    chunk_index: int,
    page_number: int | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{document_id}-{chunk_index}",
        document_id=document_id,
        filename=filename,
        text=text,
        chunk_index=chunk_index,
        page_number=page_number,
    )


def _first_chunks() -> list[ChunkRecord]:
    return [
        _chunk("doc-a", "a.txt", "alpha content", 0),
        _chunk("doc-a", "a.txt", "alpha more", 1),
    ]


def _second_chunks() -> list[ChunkRecord]:
    return [
        _chunk("doc-b", "b.pdf", "beta page one", 0, page_number=1),
        _chunk("doc-b", "b.pdf", "beta page two", 1, page_number=2),
    ]


def _embeddings(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        if "alpha" in text:
            rows.append([1.0, 0.0])
        elif "beta" in text:
            rows.append([0.0, 1.0])
        else:
            rows.append([0.5, 0.5])
    return np.asarray(rows)


def test_search_returns_document_metadata() -> None:
    index = KnowledgeIndex(_first_chunks(), _embeddings([c.text for c in _first_chunks()]))

    results = index.search(np.array([1.0, 0.0]), top_k=1)

    assert results == [
        {
            "rank": 1,
            "score": pytest.approx(1.0),
            "text": "alpha content",
            "chunk_index": 0,
            "document_id": "doc-a",
            "filename": "a.txt",
            "page_number": None,
        }
    ]


def test_search_keeps_rank_score_text_and_chunk_index() -> None:
    index = KnowledgeIndex(_first_chunks(), _embeddings([c.text for c in _first_chunks()]))

    results = index.search(np.array([1.0, 0.0]), top_k=2)

    assert [result["rank"] for result in results] == [1, 2]
    assert all("score" in result for result in results)
    assert all("text" in result for result in results)
    assert all("chunk_index" in result for result in results)


def test_replace_swaps_index_for_later_reads() -> None:
    first = _first_chunks()
    index = KnowledgeIndex(first, _embeddings([c.text for c in first]))

    second = _second_chunks()
    index.replace(second, _embeddings([c.text for c in second]))

    assert index.chunk_count == 2
    results = index.search(np.array([0.0, 1.0]), top_k=2)
    assert results[0]["document_id"] == "doc-b"
    assert results[0]["filename"] == "b.pdf"
    assert results[0]["page_number"] == 1
    assert results[1]["page_number"] == 2


def test_failed_build_keeps_old_index_unchanged() -> None:
    first = _first_chunks()
    index = KnowledgeIndex(first, _embeddings([c.text for c in first]))

    broken = _first_chunks() + [
        _chunk("doc-c", "c.txt", "zero vector", 2)
    ]
    bad_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="零向量"):
        index.replace(broken, bad_embeddings)

    assert index.chunk_count == 2
    results = index.search(np.array([1.0, 0.0]), top_k=1)
    assert results[0]["document_id"] == "doc-a"


def test_replacing_with_empty_chunks_clears_index() -> None:
    first = _first_chunks()
    index = KnowledgeIndex(first, _embeddings([c.text for c in first]))

    index.replace([], None)

    assert index.chunk_count == 0
    assert index.search(np.array([1.0, 0.0]), top_k=3) == []


def test_empty_index_has_clear_behavior() -> None:
    index = KnowledgeIndex()

    assert index.chunk_count == 0
    assert index.search(np.array([1.0, 0.0]), top_k=3) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_non_positive_top_k(top_k: int) -> None:
    index = KnowledgeIndex()

    with pytest.raises(ValueError, match="top_k 必须大于 0"):
        index.search(np.array([1.0, 0.0]), top_k=top_k)


def test_concurrent_readers_never_see_a_half_built_index() -> None:
    first = _first_chunks()
    index = KnowledgeIndex(first, _embeddings([c.text for c in first]))
    observed: list[int] = []
    barrier = threading.Barrier(3)

    def reader() -> None:
        barrier.wait()
        for _ in range(50):
            results = index.search(np.array([1.0, 0.0]), top_k=1)
            if results:
                observed.append(results[0]["chunk_index"])

    def writer() -> None:
        barrier.wait()
        for iteration in range(50):
            chunks = [
                _chunk("doc-d", "d.txt", f"dynamic {iteration}", 0),
                _chunk("doc-d", "d.txt", f"dynamic {iteration} tail", 1),
            ]
            index.replace(
                chunks,
                _embeddings([chunk.text for chunk in chunks]),
            )

    threads = [
        threading.Thread(target=reader),
        threading.Thread(target=reader),
        threading.Thread(target=writer),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert observed
    assert set(observed) <= {0, 1}
