"""Two-stage retrieval service: vector candidate pool + optional reranking.

This module orchestrates the retrieval flow without containing business
logic itself:

1. **Vector retrieval** – the existing ``KnowledgeIndex.search`` produces a
   candidate pool of ``candidate_top_k`` chunks ranked by cosine similarity.
2. **Optional reranking** – when a reranker is available, the candidate pool
   is re-scored and re-sorted.  When it is unavailable or fails, the original
   vector order is preserved (safe fallback).
3. **Final truncation** – only ``final_top_k`` chunks are returned to the
   caller.

Key invariants:

* ``retrieval_score`` (cosine similarity) is **never** overwritten by
  ``rerank_score``.
* ``max_retrieval_score`` is taken from the candidate pool, not from the
  reranked results, so the refusal decision is unaffected by reranking.
* ``reranker_fallback`` is reported so callers and health checks can observe
  degradation.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.reranker import (
    RerankInput,
    RerankScore,
    RerankerConfig,
    RerankerProtocol,
    safe_rerank,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk after two-stage retrieval.

    ``retrieval_rank`` and ``retrieval_score`` come from vector retrieval
    and are immutable across reranking.  ``final_rank`` is the display
    order after optional reranking.  ``rerank_score`` is ``None`` when the
    reranker was not applied or fell back.
    """

    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    text: str

    retrieval_rank: int
    retrieval_score: float

    final_rank: int
    rerank_score: float | None


@dataclass(frozen=True)
class RetrievalOutcome:
    """The complete result of one retrieval request.

    ``max_retrieval_score`` is the highest cosine similarity in the
    candidate pool.  It is the score used for the refusal decision and is
    **not** affected by reranking.
    """

    sources: tuple[RetrievedChunk, ...]
    candidate_count: int
    reranker_applied: bool
    reranker_fallback: bool
    max_retrieval_score: float | None


# ---------------------------------------------------------------------------
# Retriever protocol (what KnowledgeIndex satisfies)
# ---------------------------------------------------------------------------


class RetrieverProtocol(Protocol):
    """Describe the search surface the retrieval service depends on."""

    def search(
        self,
        query_embedding: object,
        top_k: int,
    ) -> list[dict[str, object]]:
        ...


# ---------------------------------------------------------------------------
# Stable tie-breaker
# ---------------------------------------------------------------------------


def _stable_sort_key(
    rerank_score: float | None,
    retrieval_rank: int,
    document_id: str,
    chunk_index: int,
) -> tuple[float, int, str, int]:
    """Produce a deterministic sort key for final ranking.

    Order of precedence:
    1. ``rerank_score`` descending (higher is better)
    2. ``retrieval_rank`` ascending (earlier vector hit wins ties)
    3. ``document_id`` ascending
    4. ``chunk_index`` ascending
    """
    # Negate rerank_score so that sorting ascending gives descending scores.
    # When rerank_score is None (fallback), use -retrieval_score so the
    # original vector order is preserved.
    if rerank_score is not None:
        primary = -rerank_score
    else:
        primary = float(retrieval_rank)
    return (primary, retrieval_rank, document_id, chunk_index)


# ---------------------------------------------------------------------------
# Retrieval service
# ---------------------------------------------------------------------------


class RetrievalService:
    """Orchestrate vector retrieval and optional reranking.

    The service is constructed once at application startup.  When the
    reranker is disabled or failed to load, ``_reranker`` is ``None`` and
    the service behaves exactly like the original vector-only pipeline.
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        config: RerankerConfig,
        reranker: RerankerProtocol | None = None,
    ) -> None:
        self._retriever = retriever
        self._config = config
        # Only hold a reranker when enabled AND successfully loaded.
        if config.enabled and reranker is not None:
            self._reranker: RerankerProtocol | None = reranker
        else:
            self._reranker = None

    @property
    def reranker_enabled(self) -> bool:
        return self._config.enabled

    @property
    def reranker_ready(self) -> bool:
        return self._reranker is not None

    @property
    def reranker_model(self) -> str | None:
        if self._reranker is not None:
            return getattr(self._reranker, "model_name", None)
        return None

    @property
    def candidate_top_k(self) -> int:
        return self._config.candidate_top_k

    def retrieve(
        self,
        query_embedding: object,
        query_text: str,
        final_top_k: int,
    ) -> RetrievalOutcome:
        """Run two-stage retrieval and return the outcome.

        ``query_text`` is needed for the reranker (it scores query-chunk
        pairs, not embeddings).  ``final_top_k`` controls how many chunks
        are returned to the caller.
        """
        if final_top_k <= 0:
            raise ValueError("final_top_k 必须大于 0")

        # Stage 1: vector retrieval (always runs)
        candidate_count = max(
            self._config.candidate_top_k if self._reranker is not None
            else final_top_k,
            final_top_k,
        )
        raw_results = self._retriever.search(
            query_embedding,
            top_k=candidate_count,
        )

        if not raw_results:
            return RetrievalOutcome(
                sources=(),
                candidate_count=0,
                reranker_applied=False,
                reranker_fallback=False,
                max_retrieval_score=None,
            )

        max_retrieval_score = _extract_score(raw_results[0])

        # Stage 2: optional reranking
        reranker_applied = False
        reranker_fallback = False
        rerank_scores: dict[str, float] = {}

        if self._reranker is not None:
            candidates = [
                RerankInput(
                    candidate_id=str(r["chunk_index"]),
                    text=str(r["text"]),
                )
                for r in raw_results
            ]
            scores, fallback = safe_rerank(
                self._reranker,
                query_text,
                candidates,
            )
            if fallback or not scores:
                reranker_fallback = True
            else:
                reranker_applied = True
                rerank_scores = {s.candidate_id: s.score for s in scores}

        # Build final ordered list
        chunks_with_scores: list[tuple[dict[str, object], float | None]] = []
        for raw in raw_results:
            cid = str(raw["chunk_index"])
            rscore = rerank_scores.get(cid) if reranker_applied else None
            chunks_with_scores.append((raw, rscore))

        # Sort by stable tie-breaker
        chunks_with_scores.sort(
            key=lambda pair: _stable_sort_key(
                pair[1],
                int(pair[0]["rank"]),
                str(pair[0]["document_id"]),
                int(pair[0]["chunk_index"]),
            )
        )

        # Truncate to final_top_k and assign final ranks
        truncated = chunks_with_scores[:final_top_k]
        sources: list[RetrievedChunk] = []
        for final_rank, (raw, rscore) in enumerate(truncated, start=1):
            sources.append(
                RetrievedChunk(
                    document_id=str(raw["document_id"]),
                    filename=str(raw["filename"]),
                    page_number=_extract_page_number(raw),
                    chunk_index=int(raw["chunk_index"]),
                    text=str(raw["text"]),
                    retrieval_rank=int(raw["rank"]),
                    retrieval_score=_extract_score(raw),
                    final_rank=final_rank,
                    rerank_score=rscore,
                )
            )

        return RetrievalOutcome(
            sources=tuple(sources),
            candidate_count=len(raw_results),
            reranker_applied=reranker_applied,
            reranker_fallback=reranker_fallback,
            max_retrieval_score=max_retrieval_score,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_score(raw: dict[str, object]) -> float:
    """Extract and validate a retrieval score from a raw result dict."""
    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("检索结果的分数无效")
    result = float(score)
    if not math.isfinite(result):
        raise ValueError("检索结果的分数不是有限数")
    return result


def _extract_page_number(raw: dict[str, object]) -> int | None:
    """Extract an optional page number from a raw result dict."""
    page_number = raw.get("page_number")
    if page_number is None:
        return None
    if not isinstance(page_number, int) or isinstance(page_number, bool):
        raise ValueError("检索结果的页码无效")
    return page_number


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_reranker(config: RerankerConfig) -> RerankerProtocol | None:
    """Build a reranker instance from configuration.

    Returns ``None`` when disabled.  Raises ``RuntimeError`` when the model
    cannot be loaded; the caller should catch this and proceed without a
    reranker (safe degradation).
    """
    if not config.enabled:
        return None
    if not config.model_name:
        return None
    # Lazy construction; any import or load failure is caught by the caller.
    from src.reranker import CrossEncoderReranker

    return CrossEncoderReranker(config.model_name)
