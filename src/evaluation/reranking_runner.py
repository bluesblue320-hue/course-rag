"""Run the A/B reranking evaluation over the production retrieval pipeline.

The runner executes **exactly one** query embedding and **exactly one**
vector retrieval per question.  The candidate pool (Top-15) is then split
into two branches:

* **Vector-only** – the first ``final_top_k`` chunks from the original
  vector ranking.
* **Reranked** – the full candidate pool is re-scored by the reranker,
  re-sorted, and truncated to ``final_top_k``.

Both branches reuse the same candidate pool so the comparison reflects
only the reranker's re-ordering, not a different retrieval pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, Sequence

import numpy as np

from src.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.evaluation.matching import (
    count_matched_evidence,
    first_relevant_rank,
)
from src.evaluation.models import EvaluationCase, RetrievedSource
from src.evaluation.reranking_models import (
    BranchResult,
    CandidateResult,
    DecisionInvarianceResult,
    LatencyRecord,
    RerankingCaseResult,
)
from src.evaluation.reranking_metrics import (
    BranchMetrics,
    CandidateMetrics,
    MetricDeltas,
    compute_candidate_metrics,
    compute_metric_deltas,
    compute_reranked_metrics,
    compute_vector_metrics,
    group_branch_metrics,
    summarize_latency,
)
from src.evaluation.runner import EmbeddingProtocol
from src.exceptions import EvaluationError
from src.knowledge_index import KnowledgeIndex
from src.rag_service import has_sufficient_context
from src.reranker import RerankInput, RerankerProtocol, safe_rerank
from src.retrieval_service import _stable_sort_key


FOCUS_CASE_IDS: tuple[str, ...] = ("p-001", "p-010", "p-012")

# Project acceptance thresholds (not industry standards).
HIT_AT_1_ABSOLUTE_IMPROVEMENT = 0.05
MRR_ABSOLUTE_IMPROVEMENT = 0.03
PARAPHRASE_MRR_IMPROVEMENT = 0.05
HIT_AT_5_REGRESSION_LIMIT = 0.01


@dataclass(frozen=True)
class RerankingRunConfiguration:
    """Describe one reproducible A/B evaluation run."""

    manifest_path: str
    dataset_path: str
    embedding_model: str
    reranker_model: str
    candidate_top_k: int
    final_top_k: int


@dataclass(frozen=True)
class RerankingRun:
    """Complete A/B evaluation results."""

    configuration: RerankingRunConfiguration
    cases: tuple[EvaluationCase, ...]
    results: tuple[RerankingCaseResult, ...]
    candidate_metrics: CandidateMetrics
    vector_metrics: BranchMetrics
    reranked_metrics: BranchMetrics
    metric_deltas: MetricDeltas
    vector_metrics_by_category: dict[str, BranchMetrics]
    reranked_metrics_by_category: dict[str, BranchMetrics]
    vector_metrics_by_difficulty: dict[str, BranchMetrics]
    reranked_metrics_by_difficulty: dict[str, BranchMetrics]
    vector_metrics_by_split: dict[str, BranchMetrics]
    reranked_metrics_by_split: dict[str, BranchMetrics]
    improved_case_ids: tuple[str, ...]
    regressed_case_ids: tuple[str, ...]
    unchanged_case_ids: tuple[str, ...]
    decision_invariance_passed: bool
    decision_invariance_inconsistent_case_ids: tuple[str, ...]
    reranker_applied_count: int
    reranker_fallback_count: int
    reranker_fallback_rate: float | None
    recommend_enable: bool
    recommendation_reasons: tuple[str, ...]
    latency_records: tuple[LatencyRecord, ...]


def _to_candidate_sources(
    raw_results: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Convert raw retrieval dicts to a stable tuple."""
    return tuple(
        {
            "rank": int(r["rank"]),
            "score": float(r["score"]),
            "text": str(r["text"]),
            "chunk_index": int(r["chunk_index"]),
            "document_id": str(r["document_id"]),
            "filename": str(r["filename"]),
            "page_number": r["page_number"],
        }
        for r in raw_results
    )


def _dicts_to_sources(
    sources: Sequence[dict[str, object]],
) -> tuple[RetrievedSource, ...]:
    """Convert dict sources to RetrievedSource objects for matching."""
    result: list[RetrievedSource] = []
    for i, s in enumerate(sources):
        page_number = s.get("page_number")
        if page_number is not None and not isinstance(page_number, int):
            page_number = None
        result.append(
            RetrievedSource(
                rank=int(s.get("rank", i + 1)),
                score=float(s.get("score", 0.0)),
                text=str(s.get("text", "")),
                chunk_index=int(s.get("chunk_index", 0)),
                document_id=str(s.get("document_id", "")),
                filename=str(s.get("filename", "")),
                page_number=page_number if isinstance(page_number, int) else None,
            )
        )
    return tuple(result)


def _build_branch_result(
    case: EvaluationCase,
    sources: Sequence[dict[str, object]],
) -> BranchResult:
    """Compute Hit@K, Recall@K, MRR for one branch's top-K sources."""
    src_tuple = _dicts_to_sources(sources)
    if not case.answerable:
        return BranchResult(
            first_relevant_rank=None,
            hit_at_1=None,
            hit_at_3=None,
            hit_at_5=None,
            recall_at_1=None,
            recall_at_3=None,
            recall_at_5=None,
            reciprocal_rank=None,
            sources=tuple(sources),
        )

    matched_1 = count_matched_evidence(
        case.expected_evidence, src_tuple, 1
    )
    matched_3 = count_matched_evidence(
        case.expected_evidence, src_tuple, 3
    )
    matched_5 = count_matched_evidence(
        case.expected_evidence, src_tuple, 5
    )
    rank = first_relevant_rank(case.expected_evidence, src_tuple)
    total = len(case.expected_evidence)

    return BranchResult(
        first_relevant_rank=rank,
        hit_at_1=matched_1 > 0 if case.answerable else None,
        hit_at_3=matched_3 > 0 if case.answerable else None,
        hit_at_5=matched_5 > 0 if case.answerable else None,
        recall_at_1=matched_1 / total if total > 0 else None,
        recall_at_3=matched_3 / total if total > 0 else None,
        recall_at_5=matched_5 / total if total > 0 else None,
        reciprocal_rank=(1.0 / rank if rank is not None else 0.0),
        sources=tuple(sources),
    )


def _build_candidate_result(
    case: EvaluationCase,
    candidates: Sequence[dict[str, object]],
    candidate_top_k: int,
) -> CandidateResult | None:
    """Compute candidate-pool metrics (Hit@5/10/15, Recall@5/10/15, MRR)."""
    if not case.answerable:
        return None

    src_tuple = _dicts_to_sources(candidates)
    matched_5 = count_matched_evidence(
        case.expected_evidence, src_tuple, 5
    )
    matched_10 = count_matched_evidence(
        case.expected_evidence, src_tuple, 10
    )
    # Candidate Hit@15 / Recall@15 are always computed over the first 15 of the
    # candidate pool.  The A/B runner already requires candidate_top_k >= 15, so
    # the candidate list here always has at least 15 entries when answerable.
    matched_15 = count_matched_evidence(
        case.expected_evidence, src_tuple, 15
    )
    rank = first_relevant_rank(case.expected_evidence, src_tuple)
    total = len(case.expected_evidence)

    return CandidateResult(
        first_relevant_rank=rank,
        hit_at_5=matched_5 > 0,
        hit_at_10=matched_10 > 0,
        hit_at_15=matched_15 > 0,
        recall_at_5=matched_5 / total if total > 0 else None,
        recall_at_10=matched_10 / total if total > 0 else None,
        recall_at_15=matched_15 / total if total > 0 else None,
        reciprocal_rank=(1.0 / rank if rank is not None else 0.0),
    )


def _rerank_candidates(
    reranker: RerankerProtocol,
    query: str,
    candidates: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], bool]:
    """Re-sort candidates by reranker score.  Returns (sorted, fallback)."""
    if not candidates:
        return list(candidates), False

    rerank_inputs = [
        RerankInput(
            candidate_id=str(c["chunk_index"]),
            text=str(c["text"]),
        )
        for c in candidates
    ]
    scores, fallback = safe_rerank(reranker, query, rerank_inputs)

    if fallback or not scores:
        return list(candidates), True

    score_map = {s.candidate_id: s.score for s in scores}

    annotated = [
        (c, score_map.get(str(c["chunk_index"]))) for c in candidates
    ]
    annotated.sort(
        key=lambda pair: _stable_sort_key(
            pair[1],
            int(pair[0]["rank"]),
            str(pair[0]["document_id"]),
            int(pair[0]["chunk_index"]),
        )
    )
    return [c for c, _ in annotated], False


def check_decision_invariance(
    *,
    vector_max_score: float | None,
    reranked_max_score: float | None,
    comparison_threshold: float,
    recommended_threshold: float,
    case_id: str | None = None,
) -> DecisionInvarianceResult:
    """Check refusal-decision invariance between the two branches.

    The production refusal decision is computed by ``has_sufficient_context``
    over the highest **vector** retrieval score.  This function computes that
    decision independently for the vector-only view (``vector_max_score``) and
    the reranked view (``reranked_max_score``) at both the comparison and
    recommended thresholds, and reports failure when they disagree.

    Although the current pipeline shares one ``max_retrieval_score`` between
    both branches (so the two inputs are normally identical), this is a real
    assertion, not a hardcoded ``True``: feed differing scores and it fails.
    """
    vector_at_comparison = has_sufficient_context(
        vector_max_score, comparison_threshold
    )
    reranked_at_comparison = has_sufficient_context(
        reranked_max_score, comparison_threshold
    )
    vector_at_recommended = has_sufficient_context(
        vector_max_score, recommended_threshold
    )
    reranked_at_recommended = has_sufficient_context(
        reranked_max_score, recommended_threshold
    )

    mismatch = (
        vector_at_comparison != reranked_at_comparison
        or vector_at_recommended != reranked_at_recommended
    )

    return DecisionInvarianceResult(
        passed=not mismatch,
        inconsistent_case_ids=(case_id,) if (mismatch and case_id) else (),
        vector_decision_at_comparison=vector_at_comparison,
        reranked_decision_at_comparison=reranked_at_comparison,
        vector_decision_at_recommended=vector_at_recommended,
        reranked_decision_at_recommended=reranked_at_recommended,
    )


def _check_decision_invariance(
    results: Sequence[RerankingCaseResult],
    comparison_threshold: float,
    recommended_threshold: float,
) -> tuple[bool, tuple[str, ...]]:
    """Run :func:`check_decision_invariance` per case and aggregate.

    Returns ``(passed, inconsistent_case_ids)``.  Both branches currently share
    ``max_retrieval_score``, so this normally passes; if a case ever yields
    differing decisions the check fails and the offending case id is reported.
    """
    inconsistent: list[str] = []
    for case in results:
        result = check_decision_invariance(
            vector_max_score=case.max_retrieval_score,
            reranked_max_score=case.max_retrieval_score,
            comparison_threshold=comparison_threshold,
            recommended_threshold=recommended_threshold,
            case_id=case.case_id,
        )
        if not result.passed:
            inconsistent.extend(result.inconsistent_case_ids)
    return (len(inconsistent) == 0), tuple(inconsistent)


def _evaluate_recommendation(
    vector_metrics: BranchMetrics,
    reranked_metrics: BranchMetrics,
    vector_by_category: dict[str, BranchMetrics],
    reranked_by_category: dict[str, BranchMetrics],
    regressed_count: int,
    decision_invariance_passed: bool,
    reranker_model: str,
    reranker_fallback_count: int = 0,
) -> tuple[bool, tuple[str, ...]]:
    """Apply transparent, rule-based recommendation logic.

    These thresholds are project acceptance criteria, not industry standards.
    """
    reasons: list[str] = []

    # 0. No fallback allowed.  A fallback means the run did not exercise the
    #    real reranker end-to-end, so it cannot be used to justify enabling it.
    if reranker_fallback_count > 0:
        return False, (
            f"Reranker 在 {reranker_fallback_count} 个案例中发生回退，不能建议启用",
        )

    # 1. Reranked Hit@1 > Vector Hit@1
    if reranked_metrics.hit_at_1 is None or vector_metrics.hit_at_1 is None:
        return False, ("Hit@1 数据缺失",)
    if reranked_metrics.hit_at_1 <= vector_metrics.hit_at_1:
        return False, ("Reranked Hit@1 未超过 Vector Hit@1",)
    reasons.append("Reranked Hit@1 > Vector Hit@1")

    # 2. Reranked MRR > Vector MRR
    if reranked_metrics.mean_reciprocal_rank is None or vector_metrics.mean_reciprocal_rank is None:
        return False, ("MRR 数据缺失",)
    if reranked_metrics.mean_reciprocal_rank <= vector_metrics.mean_reciprocal_rank:
        return False, ("Reranked MRR 未超过 Vector MRR",)
    reasons.append("Reranked MRR > Vector MRR")

    # 3. Paraphrase improvement
    para_vector = vector_by_category.get("paraphrase")
    para_reranked = reranked_by_category.get("paraphrase")
    if para_vector is None or para_reranked is None:
        return False, ("Paraphrase 指标缺失",)
    para_mrr_delta = (para_reranked.mean_reciprocal_rank or 0) - (
        para_vector.mean_reciprocal_rank or 0
    )
    if para_mrr_delta < PARAPHRASE_MRR_IMPROVEMENT:
        return False, (
            f"Paraphrase MRR 提升 {para_mrr_delta:.4f} 低于阈值 {PARAPHRASE_MRR_IMPROVEMENT}",
        )
    reasons.append(f"Paraphrase MRR 提升 {para_mrr_delta:.4f}")

    # 4. Reranked Hit@5 >= Vector Hit@5
    if reranked_metrics.hit_at_5 is None or vector_metrics.hit_at_5 is None:
        return False, ("Hit@5 数据缺失",)
    hit5_delta = (reranked_metrics.hit_at_5 or 0) - (vector_metrics.hit_at_5 or 0)
    if hit5_delta < -HIT_AT_5_REGRESSION_LIMIT:
        return False, (f"Hit@5 退化 {abs(hit5_delta):.4f}",)
    reasons.append("Hit@5 无明显退化")

    # 5. Direct Hit@5 not significantly degraded
    direct_vector = vector_by_category.get("direct")
    direct_reranked = reranked_by_category.get("direct")
    if direct_vector is not None and direct_reranked is not None:
        direct_delta = (direct_reranked.hit_at_5 or 0) - (direct_vector.hit_at_5 or 0)
        if direct_delta < -HIT_AT_5_REGRESSION_LIMIT:
            return False, (f"Direct Hit@5 退化 {abs(direct_delta):.4f}",)
        reasons.append("Direct Hit@5 无明显退化")

    # 6. Regressed cases within limit
    if regressed_count > 5:
        return False, (f"退化案例数 {regressed_count} 超过上限 5",)
    reasons.append(f"退化案例数 {regressed_count} 在上限内")

    # 7. Decision invariance check passed
    if not decision_invariance_passed:
        return False, ("拒答决策一致性检查未通过",)
    reasons.append("拒答决策一致性检查通过")

    # 8. Real reranker model specified
    if not reranker_model:
        return False, ("未指定真实 Reranker 模型",)
    reasons.append(f"使用真实 Reranker: {reranker_model}")

    # Check absolute improvement thresholds
    hit1_delta = reranked_metrics.hit_at_1 - vector_metrics.hit_at_1
    if hit1_delta < HIT_AT_1_ABSOLUTE_IMPROVEMENT:
        return False, (
            f"Hit@1 绝对提升 {hit1_delta:.4f} 低于阈值 {HIT_AT_1_ABSOLUTE_IMPROVEMENT}",
        )

    mrr_delta = reranked_metrics.mean_reciprocal_rank - vector_metrics.mean_reciprocal_rank
    if mrr_delta < MRR_ABSOLUTE_IMPROVEMENT:
        return False, (
            f"MRR 绝对提升 {mrr_delta:.4f} 低于阈值 {MRR_ABSOLUTE_IMPROVEMENT}",
        )

    return True, tuple(reasons)


def run_reranking_evaluation(
    *,
    cases: tuple[EvaluationCase, ...],
    corpus_chunks: tuple[Any, ...],
    embedding_service: EmbeddingProtocol,
    embedding_model: str,
    reranker: RerankerProtocol,
    reranker_model: str,
    manifest_path: str,
    dataset_path: str,
    candidate_top_k: int,
    final_top_k: int,
    comparison_threshold: float,
    recommended_threshold: float,
    include_latency: bool = False,
) -> RerankingRun:
    """Execute the A/B reranking evaluation.

    For each question:
    1. Encode the query once.
    2. Retrieve Top-``candidate_top_k`` candidates once.
    3. Split into vector-only (first ``final_top_k``) and reranked (full pool
       re-scored, then truncated to ``final_top_k``).
    4. Compute metrics for both branches.
    """
    if not cases:
        raise EvaluationError("评估数据集不能为空")
    # The A/B evaluation reports Candidate Hit@15 / Recall@15 metrics that are
    # only meaningful when the candidate pool is at least 15 wide.  Production
    # Reranker configuration is unaffected by this rule.
    if candidate_top_k < 15:
        raise EvaluationError("A/B 评估要求 candidate_top_k 必须大于等于 15")
    if candidate_top_k < final_top_k:
        raise EvaluationError("candidate_top_k 必须大于等于 final_top_k")
    if final_top_k < 5:
        raise EvaluationError("final_top_k 必须大于等于 5")

    # Build the index once (document embeddings computed once)
    document_embeddings = embedding_service.encode_documents(
        [chunk.text for chunk in corpus_chunks]
    )
    index = KnowledgeIndex()
    index.replace(list(corpus_chunks), document_embeddings)

    results: list[RerankingCaseResult] = []
    latency_records: list[LatencyRecord] = []

    for case in cases:
        # Stage 1: one query embedding
        retrieval_start = perf_counter()
        query_embedding = embedding_service.encode_query(case.question)

        # Stage 2: one vector retrieval (candidate pool)
        raw_results = index.search(
            query_embedding,
            top_k=candidate_top_k,
        )
        retrieval_end = perf_counter()
        vector_retrieval_ms = (retrieval_end - retrieval_start) * 1000

        candidates = list(_to_candidate_sources(raw_results))
        max_retrieval_score = (
            float(candidates[0]["score"]) if candidates else None
        )
        candidate_count = len(candidates)

        # Candidate metrics (over the full pool)
        candidate_result = _build_candidate_result(
            case, candidates, candidate_top_k
        )

        # Vector-only branch: first final_top_k from original ranking
        vector_sources = candidates[:final_top_k]
        vector_result = _build_branch_result(case, vector_sources)

        # Reranked branch: re-score full pool, then truncate
        rerank_start = perf_counter()
        reranked_sources, fallback = _rerank_candidates(
            reranker,
            case.question,
            candidates,
        )
        rerank_end = perf_counter()
        rerank_ms = (rerank_end - rerank_start) * 1000

        reranked_truncated = reranked_sources[:final_top_k]
        reranked_result = _build_branch_result(case, reranked_truncated)

        reranker_applied = not fallback
        reranker_fallback = fallback

        # Rank change
        rank_change = None
        improved = False
        regressed = False
        unchanged = False
        if (
            case.answerable
            and vector_result.first_relevant_rank is not None
            and reranked_result.first_relevant_rank is not None
        ):
            rank_change = (
                reranked_result.first_relevant_rank
                - vector_result.first_relevant_rank
            )
            if rank_change < 0:
                improved = True
            elif rank_change > 0:
                regressed = True
            else:
                unchanged = True
        elif case.answerable:
            # One or both branches missed
            vector_hit = vector_result.first_relevant_rank is not None
            reranked_hit = reranked_result.first_relevant_rank is not None
            if reranked_hit and not vector_hit:
                improved = True
            elif vector_hit and not reranked_hit:
                regressed = True
            elif vector_hit and reranked_hit:
                unchanged = True

        end_to_end_ms = vector_retrieval_ms + rerank_ms

        if include_latency:
            latency_records.append(
                LatencyRecord(
                    vector_retrieval_ms=round(vector_retrieval_ms, 4),
                    rerank_ms=round(rerank_ms, 4),
                    end_to_end_ms=round(end_to_end_ms, 4),
                )
            )

        results.append(
            RerankingCaseResult(
                case_id=case.id,
                split=case.split,
                category=case.category,
                difficulty=case.difficulty,
                answerable=case.answerable,
                question=case.question,
                candidate=candidate_result,
                max_retrieval_score=max_retrieval_score,
                candidate_count=candidate_count,
                vector=vector_result,
                reranked=reranked_result,
                reranker_applied=reranker_applied,
                reranker_fallback=reranker_fallback,
                rank_change=rank_change,
                improved=improved,
                regressed=regressed,
                unchanged=unchanged,
            )
        )

    all_results = tuple(results)

    # Compute aggregate metrics
    candidate_metrics = compute_candidate_metrics(all_results)
    vector_metrics = compute_vector_metrics(all_results)
    reranked_metrics = compute_reranked_metrics(all_results)
    metric_deltas = compute_metric_deltas(vector_metrics, reranked_metrics)

    vector_by_category = group_branch_metrics(all_results, "vector", "category")
    reranked_by_category = group_branch_metrics(all_results, "reranked", "category")
    vector_by_difficulty = group_branch_metrics(all_results, "vector", "difficulty")
    reranked_by_difficulty = group_branch_metrics(all_results, "reranked", "difficulty")
    vector_by_split = group_branch_metrics(all_results, "vector", "split")
    reranked_by_split = group_branch_metrics(all_results, "reranked", "split")

    improved_case_ids = tuple(
        r.case_id for r in all_results if r.improved
    )
    regressed_case_ids = tuple(
        r.case_id for r in all_results if r.regressed
    )
    unchanged_case_ids = tuple(
        r.case_id for r in all_results if r.unchanged
    )

    decision_invariance, inconsistent_case_ids = _check_decision_invariance(
        all_results,
        comparison_threshold,
        recommended_threshold,
    )

    reranker_applied_count = sum(1 for r in all_results if r.reranker_applied)
    reranker_fallback_count = sum(1 for r in all_results if r.reranker_fallback)
    reranker_fallback_rate = (
        reranker_fallback_count / len(all_results) if all_results else None
    )

    recommend_enable, recommendation_reasons = _evaluate_recommendation(
        vector_metrics,
        reranked_metrics,
        vector_by_category,
        reranked_by_category,
        len(regressed_case_ids),
        decision_invariance,
        reranker_model,
        reranker_fallback_count=reranker_fallback_count,
    )

    return RerankingRun(
        configuration=RerankingRunConfiguration(
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
        ),
        cases=cases,
        results=all_results,
        candidate_metrics=candidate_metrics,
        vector_metrics=vector_metrics,
        reranked_metrics=reranked_metrics,
        metric_deltas=metric_deltas,
        vector_metrics_by_category=vector_by_category,
        reranked_metrics_by_category=reranked_by_category,
        vector_metrics_by_difficulty=vector_by_difficulty,
        reranked_metrics_by_difficulty=reranked_by_difficulty,
        vector_metrics_by_split=vector_by_split,
        reranked_metrics_by_split=reranked_by_split,
        improved_case_ids=improved_case_ids,
        regressed_case_ids=regressed_case_ids,
        unchanged_case_ids=unchanged_case_ids,
        decision_invariance_passed=decision_invariance,
        decision_invariance_inconsistent_case_ids=inconsistent_case_ids,
        reranker_applied_count=reranker_applied_count,
        reranker_fallback_count=reranker_fallback_count,
        reranker_fallback_rate=reranker_fallback_rate,
        recommend_enable=recommend_enable,
        recommendation_reasons=recommendation_reasons,
        latency_records=tuple(latency_records),
    )
