"""Split text into simple fixed-length overlapping chunks."""

# Named so that offline evaluation can report the exact production settings
# instead of duplicating literals that could silently drift apart.
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Return non-blank chunks produced by a fixed-size sliding window."""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Advancing by less than chunk_size keeps context shared by neighbors.
        start += step

    return chunks
