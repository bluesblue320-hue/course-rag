"""Application-specific exceptions for the RAG pipeline."""


class RagError(RuntimeError):
    """Base exception for the RAG pipeline."""


class GenerationConfigurationError(RagError):
    """Raised when LLM configuration is incomplete."""


class GenerationError(RagError):
    """Raised when answer generation fails."""
