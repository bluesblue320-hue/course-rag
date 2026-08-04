"""Load the controlled evaluation corpus with production loaders only."""

import json
from dataclasses import dataclass
from pathlib import Path

from src.document_chunker import chunk_document
from src.document_loaders import (
    DocumentLoader,
    MarkdownDocumentLoader,
    PdfDocumentLoader,
    TextDocumentLoader,
)
from src.documents import ChunkRecord
from src.exceptions import (
    CorpusValidationError,
    DocumentParseError,
    EmptyDocumentError,
)

_REQUIRED_FIELDS = ("document_id", "path", "filename", "content_type")


def build_default_loaders() -> dict[str, DocumentLoader]:
    """Return the same loader mapping the production application registers."""
    return {
        ".txt": TextDocumentLoader(),
        ".md": MarkdownDocumentLoader(),
        ".pdf": PdfDocumentLoader(),
    }


@dataclass(frozen=True)
class CorpusDocument:
    """Describe one manifest entry resolved to a safe on-disk path."""

    document_id: str
    relative_path: str
    resolved_path: Path
    filename: str
    content_type: str


@dataclass(frozen=True)
class LoadedCorpus:
    """Describe the parsed corpus and its chunks in a stable order."""

    documents: tuple[CorpusDocument, ...]
    chunks: tuple[ChunkRecord, ...]

    @property
    def document_ids(self) -> frozenset[str]:
        return frozenset(document.document_id for document in self.documents)


def _read_manifest_payload(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.is_file():
        raise CorpusValidationError("语料清单文件不存在")
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CorpusValidationError("语料清单无法以 UTF-8 读取") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorpusValidationError("语料清单不是合法 JSON") from exc

    if not isinstance(payload, dict):
        raise CorpusValidationError("语料清单顶层必须是 JSON 对象")
    return payload


def _validate_relative_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise CorpusValidationError("语料条目的 path 必须是非空字符串")

    candidate = Path(raw_path.strip().replace("\\", "/"))
    if candidate.is_absolute() or raw_path.strip().startswith("/"):
        raise CorpusValidationError("语料条目的 path 不允许使用绝对路径")
    if candidate.drive or candidate.root:
        raise CorpusValidationError("语料条目的 path 不允许使用绝对路径")
    if any(part == ".." for part in candidate.parts):
        raise CorpusValidationError("语料条目的 path 不允许包含 ..")
    return candidate


def load_manifest(
    manifest_path: Path,
    base_dir: Path | None = None,
) -> tuple[CorpusDocument, ...]:
    """Parse the manifest and resolve every path inside the evaluation folder.

    ``base_dir`` is the directory relative manifest paths are resolved against
    and defaults to the current working directory. Every resolved path must
    stay inside the manifest's own directory, which blocks path traversal and
    keeps the corpus reproducible for every contributor.
    """
    manifest_path = Path(manifest_path)
    payload = _read_manifest_payload(manifest_path)

    entries = payload.get("documents")
    if not isinstance(entries, list) or not entries:
        raise CorpusValidationError("语料清单必须包含非空的 documents 数组")

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    allowed_root = manifest_path.resolve().parent

    documents: list[CorpusDocument] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise CorpusValidationError("语料条目必须是 JSON 对象")
        for field in _REQUIRED_FIELDS:
            if field not in entry:
                raise CorpusValidationError(f"语料条目缺少字段: {field}")

        document_id = entry["document_id"]
        if not isinstance(document_id, str) or not document_id.strip():
            raise CorpusValidationError("语料条目的 document_id 必须是非空字符串")
        document_id = document_id.strip()
        if document_id in seen_ids:
            raise CorpusValidationError(f"语料条目的 document_id 重复: {document_id}")
        seen_ids.add(document_id)

        filename = entry["filename"]
        if not isinstance(filename, str) or not filename.strip():
            raise CorpusValidationError("语料条目的 filename 必须是非空字符串")
        content_type = entry["content_type"]
        if not isinstance(content_type, str) or not content_type.strip():
            raise CorpusValidationError("语料条目的 content_type 必须是非空字符串")

        relative_path = _validate_relative_path(entry["path"])
        resolved = (root / relative_path).resolve()
        if resolved != allowed_root and allowed_root not in resolved.parents:
            raise CorpusValidationError(
                f"语料文件必须位于评估目录内: {relative_path.as_posix()}"
            )
        if not resolved.is_file():
            raise CorpusValidationError(
                f"语料文件不存在: {relative_path.as_posix()}"
            )

        documents.append(
            CorpusDocument(
                document_id=document_id,
                relative_path=relative_path.as_posix(),
                resolved_path=resolved,
                filename=filename.strip(),
                content_type=content_type.strip(),
            )
        )

    return tuple(documents)


def load_corpus(
    documents: tuple[CorpusDocument, ...],
    loaders: dict[str, DocumentLoader] | None = None,
) -> LoadedCorpus:
    """Parse and chunk every corpus document with the production pipeline."""
    if not documents:
        raise CorpusValidationError("语料清单必须包含至少一个文档")

    active_loaders = loaders if loaders is not None else build_default_loaders()
    chunks: list[ChunkRecord] = []
    for document in documents:
        suffix = document.resolved_path.suffix.lower()
        loader = active_loaders.get(suffix)
        if loader is None:
            raise CorpusValidationError(
                f"语料文件类型不受支持: {document.relative_path}"
            )
        try:
            loaded = loader.load(document.resolved_path)
        except (DocumentParseError, EmptyDocumentError) as exc:
            raise CorpusValidationError(
                f"语料文件解析失败: {document.relative_path}"
            ) from exc

        document_chunks = chunk_document(
            loaded,
            document.document_id,
            document.filename,
        )
        if not document_chunks:
            raise CorpusValidationError(
                f"语料文件没有产生任何 Chunk: {document.relative_path}"
            )
        chunks.extend(document_chunks)

    return LoadedCorpus(documents=documents, chunks=tuple(chunks))
