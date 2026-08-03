"""Parse uploaded documents into typed page structures."""

from pathlib import Path
from typing import Protocol

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from src.documents import LoadedDocument, LoadedPage
from src.exceptions import DocumentParseError, EmptyDocumentError


class DocumentLoader(Protocol):
    """Protocol describing how one file format is parsed into pages."""

    def load(self, file_path: Path) -> LoadedDocument:
        """Return the parsed pages of one document file."""
        ...


class TextDocumentLoader:
    """Parse a UTF-8 text file into a single unnumbered page."""

    def load(self, file_path: Path) -> LoadedDocument:
        """Read strict UTF-8 text and reject blank or invalid content."""
        if not file_path.is_file():
            raise DocumentParseError("文档文件不存在")

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("文档文本不是有效的 UTF-8 编码") from exc
        except OSError as exc:
            raise DocumentParseError("文档文件无法读取") from exc

        text = text.strip()
        if not text:
            raise EmptyDocumentError("文档内容为空")

        return LoadedDocument(
            pages=(LoadedPage(page_number=None, text=text),),
            text_length=len(text),
        )


class MarkdownDocumentLoader(TextDocumentLoader):
    """Parse a Markdown file exactly like plain text for this stage."""


class PdfDocumentLoader:
    """Parse a text-based PDF into one page per physical page."""

    def load(self, file_path: Path) -> LoadedDocument:
        """Extract text page by page, starting page numbers at one."""
        if not file_path.is_file():
            raise DocumentParseError("文档文件不存在")

        try:
            reader = PdfReader(str(file_path))
        except (PdfReadError, OSError, ValueError) as exc:
            raise DocumentParseError("PDF 文件无法解析") from exc

        try:
            page_count = len(reader.pages)
        except (PdfReadError, FileNotDecryptedError) as exc:
            raise DocumentParseError("PDF 文件已加密且无法读取") from exc

        pages: list[LoadedPage] = []
        for index in range(page_count):
            try:
                text = reader.pages[index].extract_text() or ""
            except (PdfReadError, FileNotDecryptedError) as exc:
                raise DocumentParseError("PDF 文件已加密且无法读取") from exc
            stripped_text = text.strip()
            if stripped_text:
                pages.append(
                    LoadedPage(page_number=index + 1, text=stripped_text)
                )

        if not pages:
            raise EmptyDocumentError("PDF 文件不包含可提取的文本")

        return LoadedDocument(
            pages=tuple(pages),
            text_length=sum(len(page.text) for page in pages),
        )
