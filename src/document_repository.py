"""Persist document metadata in a JSON file with atomic writes."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from src.documents import DocumentRecord, sort_documents
from src.exceptions import DocumentMetadataError


class DocumentRepository:
    """Store DocumentRecord lists in one JSON file on the local filesystem."""

    def __init__(self, json_path: Path) -> None:
        self._json_path = json_path

    def list_documents(self) -> list[DocumentRecord]:
        """Return all stored documents in stable display order."""
        if not self._json_path.is_file():
            return []
        try:
            raw_text = self._json_path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except (OSError, ValueError, TypeError) as exc:
            raise DocumentMetadataError("文档元数据损坏，无法读取") from exc

        if not isinstance(payload, dict) or not isinstance(
            payload.get("documents"),
            list,
        ):
            raise DocumentMetadataError("文档元数据损坏，无法读取")

        documents: list[DocumentRecord] = []
        for item in payload["documents"]:
            if not isinstance(item, dict):
                raise DocumentMetadataError("文档元数据损坏，无法读取")
            try:
                stored_filename = item.get("stored_filename")
                documents.append(
                    DocumentRecord(
                        document_id=str(item["document_id"]),
                        original_filename=str(item["original_filename"]),
                        stored_filename=(
                            None
                            if stored_filename is None
                            else str(stored_filename)
                        ),
                        content_type=str(item["content_type"]),
                        size_bytes=int(item["size_bytes"]),
                        text_length=int(item["text_length"]),
                        chunk_count=int(item["chunk_count"]),
                        created_at=str(item["created_at"]),
                        is_builtin=bool(item["is_builtin"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DocumentMetadataError("文档元数据损坏，无法读取") from exc

        return sort_documents(documents)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one document by id or None when it does not exist."""
        for record in self.list_documents():
            if record.document_id == document_id:
                return record
        return None

    def save_documents(self, documents: list[DocumentRecord]) -> None:
        """Write the full document list atomically via a temp file."""
        payload = {"documents": [asdict(record) for record in documents]}
        self._json_path.parent.mkdir(parents=True, exist_ok=True)

        file_descriptor, temp_name = tempfile.mkstemp(
            dir=str(self._json_path.parent),
            prefix="documents-",
            suffix=".tmp",
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self._json_path)
        except OSError as exc:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise DocumentMetadataError("文档元数据保存失败") from exc
