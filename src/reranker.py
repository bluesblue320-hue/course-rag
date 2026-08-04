"""Optional second-stage reranker abstraction for the RAG pipeline.

The reranker sits *after* vector retrieval and *before* final ranking.  It
receives a larger candidate pool (e.g. Top-15) and re-scores each
``(query, chunk_text)`` pair with a CrossEncoder model, producing a
``rerank_score`` that is semantically distinct from the vector
``retrieval_score``.

Design rules enforced here:

* ``CrossEncoder`` is imported lazily so ``--help``, tests, and disabled
  mode never pay the model-loading cost.
* Model loading is **offline only** (``local_files_only=True``): a model that
  is not already cached locally fails clearly instead of silently downloading
  from the network.  This keeps the project's offline / air-gapped posture.
* Scores are validated as finite floats; ``NaN`` / ``Infinity`` / non-numeric
  values cause the caller to fall back to vector-only ordering.
* The protocol guarantees every input candidate is returned exactly once.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


RERANKER_MODEL_ERROR = "RAG_RERANKER_MODEL 在启用 Reranker 时不能为空"
RERANKER_CANDIDATE_TOP_K_ERROR = (
    "RAG_RERANKER_CANDIDATE_TOP_K 必须是大于 0 的整数"
)
RERANKER_ENABLED_ERROR = "RAG_RERANKER_ENABLED 必须是布尔值 (true/false)"

DEFAULT_CANDIDATE_TOP_K = 15
MAX_CANDIDATE_TOP_K = 100
MIN_CANDIDATE_TOP_K = 5


@dataclass(frozen=True)
class RerankInput:
    """One candidate chunk presented to the reranker.

    ``candidate_id`` must be unique within a single ``rerank`` call so the
    caller can map scores back to the original retrieval results.
    """

    candidate_id: str
    text: str


@dataclass(frozen=True)
class RerankScore:
    """The reranker's relevance score for one candidate.

    The score is **not** assumed to be in ``[0, 1]``; it may be logits or
    any other model-specific output.  Sorting is always the caller's job.
    """

    candidate_id: str
    score: float


class RerankerProtocol(Protocol):
    """Describe the contract every reranker implementation must satisfy."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankInput],
    ) -> tuple[RerankScore, ...]:
        """Return exactly one score per candidate, preserving no ordering."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankerConfig:
    """Resolved reranker configuration.

    ``enabled`` gates model loading and request-time reranking.
    ``model_name`` is the CrossEncoder model identifier (empty when disabled).
    ``candidate_top_k`` is the vector retrieval pool size fed to the reranker.
    """

    enabled: bool
    model_name: str
    candidate_top_k: int


def _parse_bool(value: str) -> bool:
    """Parse a strict boolean string (true/false, case-insensitive)."""
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise ValueError(RERANKER_ENABLED_ERROR)


def resolve_reranker_config(
    *,
    explicit_enabled: object | None = None,
    explicit_model: object | None = None,
    explicit_candidate_top_k: object | None = None,
    final_top_k: int = 5,
) -> RerankerConfig:
    """Resolve and validate reranker config from explicit values or env.

    When ``RAG_RERANKER_ENABLED`` is false (the default), the model name and
    candidate top-k are still parsed so the health endpoint can report them,
    but no model is loaded.
    """
    if explicit_enabled is not None:
        enabled = _parse_bool(str(explicit_enabled))
    else:
        raw_enabled = os.getenv("RAG_RERANKER_ENABLED", "false")
        enabled = _parse_bool(raw_enabled)

    if explicit_model is not None:
        model_name = str(explicit_model).strip()
    else:
        model_name = os.getenv("RAG_RERANKER_MODEL", "").strip()

    if explicit_candidate_top_k is not None:
        candidate_top_k = int(explicit_candidate_top_k)  # type: ignore[arg-type]
    else:
        raw_top_k = os.getenv("RAG_RERANKER_CANDIDATE_TOP_K", "")
        if raw_top_k.strip():
            candidate_top_k = int(raw_top_k)
        else:
            candidate_top_k = DEFAULT_CANDIDATE_TOP_K

    if not isinstance(candidate_top_k, int) or isinstance(
        candidate_top_k, bool
    ):
        raise ValueError(RERANKER_CANDIDATE_TOP_K_ERROR)
    if candidate_top_k < MIN_CANDIDATE_TOP_K:
        raise ValueError(
            f"RAG_RERANKER_CANDIDATE_TOP_K 必须大于等于 {MIN_CANDIDATE_TOP_K}"
        )
    if candidate_top_k > MAX_CANDIDATE_TOP_K:
        raise ValueError(
            f"RAG_RERANKER_CANDIDATE_TOP_K 不能超过 {MAX_CANDIDATE_TOP_K}"
        )
    if candidate_top_k < final_top_k:
        raise ValueError(
            "RAG_RERANKER_CANDIDATE_TOP_K 必须大于等于 final_top_k"
        )

    if enabled and not model_name:
        raise ValueError(RERANKER_MODEL_ERROR)

    return RerankerConfig(
        enabled=enabled,
        model_name=model_name,
        candidate_top_k=candidate_top_k,
    )


# ---------------------------------------------------------------------------
# Score validation helpers
# ---------------------------------------------------------------------------


def _to_finite_float(value: Any) -> float:
    """Convert a model output value to a finite Python float.

    Raises ``ValueError`` for NaN, Infinity, or non-numeric types so the
    caller can trigger a safe fallback.
    """
    if isinstance(value, bool):
        raise ValueError("reranker 分数不能是布尔值")
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        # numpy scalar fallback
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("reranker 分数不是数字") from exc
    if not math.isfinite(result):
        raise ValueError("reranker 分数不是有限数")
    return result


def _validate_rerank_scores(
    candidates: Sequence[RerankInput],
    scores: Sequence[RerankScore],
) -> tuple[RerankScore, ...]:
    """Validate that scores are complete, unique, and finite.

    Returns the validated tuple.  Raises ``ValueError`` on any violation so
    the caller can fall back to vector-only ordering.
    """
    if len(scores) != len(candidates):
        raise ValueError(
            f"reranker 返回数量不正确: 期望 {len(candidates)}, 实际 {len(scores)}"
        )

    expected_ids = {c.candidate_id for c in candidates}
    seen_ids: set[str] = set()
    validated: list[RerankScore] = []

    for score in scores:
        cid = score.candidate_id
        if cid in seen_ids:
            raise ValueError(f"reranker 返回了重复的 candidate_id: {cid}")
        if cid not in expected_ids:
            raise ValueError(f"reranker 返回了未知的 candidate_id: {cid}")
        seen_ids.add(cid)
        validated.append(
            RerankScore(
                candidate_id=cid,
                score=_to_finite_float(score.score),
            )
        )

    return tuple(validated)


# ---------------------------------------------------------------------------
# CrossEncoder implementation
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Score (query, chunk) pairs with a sentence-transformers CrossEncoder.

    The model is loaded in ``__init__`` but the ``CrossEncoder`` class itself
    is imported lazily inside ``__init__`` so module import never touches the
    heavy runtime.
    """

    def __init__(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("CrossEncoder 模型名称不能为空")
        # Lazy import: the sentence_transformers package is only needed when
        # a reranker is actually instantiated, never at module import time.
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "无法加载 sentence-transformers；请确认已安装该依赖"
            ) from exc

        self._model_name = model_name
        # Offline-only loading: if the model is not already cached locally,
        # sentence-transformers must fail loudly instead of reaching out to
        # Hugging Face.  This is a hard security/offline requirement.
        self._model = CrossEncoder(model_name, local_files_only=True)

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankInput],
    ) -> tuple[RerankScore, ...]:
        """Score every candidate against the query in one batch."""
        if not candidates:
            return ()

        pairs = [(query, c.text) for c in candidates]
        raw_scores = self._model.predict(pairs)

        scores = [
            RerankScore(
                candidate_id=candidates[i].candidate_id,
                score=raw_scores[i],
            )
            for i in range(len(candidates))
        ]
        return _validate_rerank_scores(candidates, scores)


# ---------------------------------------------------------------------------
# Fake reranker (for testing)
# ---------------------------------------------------------------------------


class FakeReranker:
    """Deterministic reranker for offline tests.

    Accepts an optional ``score_map`` that maps candidate_id to an explicit
    score.  Candidates not in the map receive a score based on their position
    (higher position = lower score, so the original order is preserved by
    default).
    """

    def __init__(
        self,
        score_map: dict[str, float] | None = None,
        fail: bool = False,
    ) -> None:
        self._score_map = dict(score_map) if score_map else {}
        self._fail = fail
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-reranker"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankInput],
    ) -> tuple[RerankScore, ...]:
        self.call_count += 1
        if self._fail:
            raise RuntimeError("FakeReranker 模拟失败")

        scores: list[RerankScore] = []
        for i, candidate in enumerate(candidates):
            if candidate.candidate_id in self._score_map:
                score = self._score_map[candidate.candidate_id]
            else:
                # Preserve original order by default: earlier = higher score.
                score = float(len(candidates) - i)
            scores.append(
                RerankScore(
                    candidate_id=candidate.candidate_id,
                    score=score,
                )
            )
        return _validate_rerank_scores(candidates, scores)


# ---------------------------------------------------------------------------
# Safe rerank wrapper
# ---------------------------------------------------------------------------


def safe_rerank(
    reranker: RerankerProtocol,
    query: str,
    candidates: Sequence[RerankInput],
) -> tuple[tuple[RerankScore, ...], bool]:
    """Call the reranker and fall back gracefully on any failure.

    Returns ``(scores, fallback)`` where ``fallback=True`` means the reranker
    could not produce valid scores and the caller should use the original
    vector retrieval order instead.

    On fallback, an empty tuple is returned so the caller knows to skip
    reranking entirely.
    """
    try:
        scores = reranker.rerank(query, candidates)
        return scores, False
    except Exception:
        return (), True
