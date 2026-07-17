import pytest

from src.chunker import split_text


def test_split_text_creates_fixed_length_chunks() -> None:
    result = split_text("abcdefghij", chunk_size=4, chunk_overlap=0)

    assert result == ["abcd", "efgh", "ij"]


def test_split_text_preserves_the_requested_overlap() -> None:
    result = split_text("abcdefghij", chunk_size=4, chunk_overlap=1)

    assert result == ["abcd", "defg", "ghij"]
    assert result[0][-1:] == result[1][:1]
    assert result[1][-1:] == result[2][:1]


def test_split_text_returns_one_chunk_for_short_text() -> None:
    assert split_text("短文本", chunk_size=10, chunk_overlap=2) == ["短文本"]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_split_text_rejects_non_positive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size 必须大于 0"):
        split_text("text", chunk_size=chunk_size, chunk_overlap=0)


def test_split_text_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap 不能小于 0"):
        split_text("text", chunk_size=4, chunk_overlap=-1)


@pytest.mark.parametrize("chunk_overlap", [4, 5])
def test_split_text_rejects_overlap_not_smaller_than_chunk_size(
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError, match="chunk_overlap 必须小于 chunk_size"):
        split_text("text", chunk_size=4, chunk_overlap=chunk_overlap)


def test_split_text_skips_whitespace_only_chunks() -> None:
    result = split_text("abc   xyz", chunk_size=3, chunk_overlap=0)

    assert result == ["abc", "xyz"]


def test_split_text_returns_empty_list_for_whitespace_only_text() -> None:
    assert split_text("   ", chunk_size=2, chunk_overlap=1) == []
