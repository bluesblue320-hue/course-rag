"""Application-specific exceptions for the RAG pipeline."""


class RagError(RuntimeError):
    """Base exception for the RAG pipeline."""


class GenerationConfigurationError(RagError):
    """Raised when LLM configuration is incomplete."""


class GenerationError(RagError):
    """Raised when answer generation fails."""


class RagConfigurationError(Exception):
    """Raised when RAG-specific configuration is invalid."""


class DocumentError(RagError):
    """Base exception for document ingestion and management."""


class UnsupportedDocumentTypeError(DocumentError):
    """Raised when an uploaded file extension or content type is not allowed."""


class DocumentParseError(DocumentError):
    """Raised when an uploaded file cannot be parsed as its declared type."""


class EmptyDocumentError(DocumentError):
    """Raised when an uploaded file contains no extractable text."""


class UploadTooLargeError(DocumentError):
    """Raised when an uploaded file exceeds the configured size limit."""


class DocumentNotFoundError(DocumentError):
    """Raised when a document id does not exist."""


class BuiltinDocumentDeletionError(DocumentError):
    """Raised when an attempt deletes the built-in knowledge document."""


class DocumentIngestionError(DocumentError):
    """Raised when an upload or deletion cannot be completed safely."""


class DocumentMetadataError(DocumentError):
    """Raised when the document metadata store is corrupted or unreadable."""


class UploadConfigurationError(DocumentError):
    """Raised when MAX_UPLOAD_BYTES is not a positive integer."""
