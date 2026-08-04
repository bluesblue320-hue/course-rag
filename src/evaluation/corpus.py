"""Load evaluation corpus documents with the production loaders."""

from pathlib import Path
from typing import Any

from src.document_chunker import chunk_document
from src.document_loaders import (
    DocumentLoader,
    MarkdownDocumentLoader,
    PdfDocumentLoader,
    TextDocumentLoader,
)
from src.documents import ChunkRecord

from .models import CorpusDocument, EvaluationDataError

_ALLOWED_SUFFIXES = {
    ".txt": TextDocumentLoader,
    ".md": MarkdownDocumentLoader,
    ".pdf": PdfDocumentLoader,
}


def _require(value: bool, message: str) -> None:
    if not value:
        raise EvaluationDataError(message)


def _load_manifest_documents(manifest_path: Path) -> list[dict[str, Any]]:
    from .dataset import load_manifest

    payload = load_manifest(manifest_path)
    documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload["documents"]):
        _require(
            isinstance(item, dict),
            f"manifest 第 {index + 1} 项必须是对象",
        )
        document_id = item.get("document_id")
        path_value = item.get("path")
        filename = item.get("filename")
        content_type = item.get("content_type")
        _require(
            isinstance(document_id, str) and document_id.strip(),
            f"manifest 第 {index + 1} 项：document_id 必须是非空字符串",
        )
        _require(
            document_id not in seen_ids,
            f"manifest: 重复的 document_id：{document_id}",
        )
        _require(
            isinstance(path_value, str) and path_value.strip(),
            f"manifest 第 {index + 1} 项：path 必须是非空字符串",
        )
        _require(
            isinstance(filename, str) and filename.strip(),
            f"manifest 第 {index + 1} 项：filename 必须是非空字符串",
        )
        _require(
            isinstance(content_type, str) and content_type.strip(),
            f"manifest 第 {index + 1} 项：content_type 必须是非空字符串",
        )
        seen_ids.add(document_id)
        documents.append(
            {
                "document_id": document_id,
                "path": path_value,
                "filename": filename,
                "content_type": content_type,
            }
        )
    _require(len(documents) > 0, "manifest 不包含任何文档")
    return documents


def _resolve_document_path(
    manifest_path: Path,
    raw_path: str,
    document_id: str,
    base_dir: Path,
) -> Path:
    eval_root = manifest_path.resolve().parent
    _require(
        not Path(raw_path).is_absolute(),
        f"manifest: {document_id} 的 path 不允许是绝对路径",
    )
    candidate = (base_dir / raw_path).resolve()
    _require(
        candidate.is_relative_to(eval_root),
        f"manifest: {document_id} 的 path 越出了评估目录",
    )
    _require(
        candidate.is_file(),
        f"manifest: {document_id} 的文件不存在：{raw_path}",
    )
    return candidate


def _loader_for(path: Path, document_id: str) -> DocumentLoader:
    suffix = path.suffix.lower()
    loader_class = _ALLOWED_SUFFIXES.get(suffix)
    _require(
        loader_class is not None,
        f"manifest: {document_id} 的文件类型不受支持：{suffix}",
    )
    return loader_class()


def load_corpus(
    manifest_path: Path,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    base_dir: Path | None = None,
) -> dict[str, list[ChunkRecord]]:
    """Load and chunk every manifest document using production code.

    Manifest paths are resolved against ``base_dir`` (the current working
    directory by default) and must stay inside the manifest's directory.
    """
    manifest_path = Path(manifest_path)
    resolution_base = (
        Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    )
    if chunk_size <= 0:
        raise EvaluationDataError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise EvaluationDataError("chunk_overlap 必须介于 0 和 chunk_size 之间")

    chunks_by_document: dict[str, list[ChunkRecord]] = {}
    for item in _load_manifest_documents(manifest_path):
        document_id = item["document_id"]
        path = _resolve_document_path(
            manifest_path,
            item["path"],
            document_id,
            resolution_base,
        )
        loader = _loader_for(path, document_id)
        try:
            loaded = loader.load(path)
        except Exception as exc:
            raise EvaluationDataError(
                f"manifest: 无法解析文档 {document_id}"
            ) from exc
        chunks = chunk_document(loaded, document_id, item["filename"])
        _require(
            len(chunks) > 0,
            f"manifest: 文档 {document_id} 没有产生任何 Chunk",
        )
        chunks_by_document[document_id] = chunks
    return chunks_by_document
