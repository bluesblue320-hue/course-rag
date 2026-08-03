"""Convert parsed pages into metadata-bearing chunk records."""

from uuid import uuid4

from src.chunker import split_text
from src.documents import ChunkRecord, LoadedDocument


def chunk_document(
    document: LoadedDocument,
    document_id: str,
    filename: str,
) -> list[ChunkRecord]:
    """Return one ChunkRecord per split page chunk with continuous indexes."""
    chunks: list[ChunkRecord] = []
    chunk_index = 0
    for page in document.pages:
        for text in split_text(page.text):
            chunks.append(
                ChunkRecord(
                    chunk_id=uuid4().hex,
                    document_id=document_id,
                    filename=filename,
                    text=text,
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                )
            )
            chunk_index += 1
    return chunks
