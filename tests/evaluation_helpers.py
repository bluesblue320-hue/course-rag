"""Shared offline fixtures for the evaluation test suite.

Everything here is deterministic and network-free: the fake embedding maps
text to a vector with character bigram hashing, so tests never download or
load a real model.
"""

import json
import zlib
from pathlib import Path

import numpy as np

from src.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvidenceExpectation,
    RetrievedSource,
)


class FakeEmbeddingService:
    """Encode text with hashed character bigrams instead of a real model."""

    def __init__(self, dimensions: int = 96) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions 必须大于 0")
        self.dimensions = dimensions
        self.document_batches = 0
        self.query_calls = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=float)
        cleaned = text.strip().casefold()
        for index in range(max(len(cleaned) - 1, 0)):
            bigram = cleaned[index : index + 2]
            slot = zlib.crc32(bigram.encode("utf-8")) % self.dimensions
            vector[slot] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            # Keep the vector non-zero so the production retriever accepts it.
            vector[0] = 1.0
            norm = 1.0
        return vector / norm

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_batches += 1
        if not texts:
            raise ValueError("文档文本不能为空")
        return np.vstack([self._vector(text) for text in texts])

    def encode_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        if not query.strip():
            raise ValueError("查询文本不能为空")
        return self._vector(query)


def make_source(
    rank: int,
    score: float,
    text: str = "任意正文",
    document_id: str = "doc-a",
    chunk_index: int = 0,
    filename: str = "doc-a.md",
    page_number: int | None = None,
) -> RetrievedSource:
    """Build one retrieved source with sensible defaults."""
    return RetrievedSource(
        rank=rank,
        score=score,
        text=text,
        chunk_index=chunk_index,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
    )


def make_expectation(
    document_id: str = "doc-a",
    page_number: int | None = None,
    required_terms: tuple[str, ...] = ("关键短语",),
) -> EvidenceExpectation:
    """Build one evidence expectation with sensible defaults."""
    return EvidenceExpectation(
        document_id=document_id,
        page_number=page_number,
        required_terms=required_terms,
    )


def make_case(
    case_id: str = "c-001",
    split: str = "calibration",
    question: str = "一个问题？",
    answerable: bool = True,
    category: str = "direct",
    difficulty: str = "easy",
    expected_evidence: tuple[EvidenceExpectation, ...] | None = None,
    notes: str = "",
) -> EvaluationCase:
    """Build one evaluation case with sensible defaults."""
    if expected_evidence is None:
        expected_evidence = (make_expectation(),) if answerable else ()
    return EvaluationCase(
        id=case_id,
        split=split,  # type: ignore[arg-type]
        question=question,
        answerable=answerable,
        category=category,
        difficulty=difficulty,
        expected_evidence=expected_evidence,
        notes=notes,
    )


def make_result(
    case_id: str = "c-001",
    split: str = "calibration",
    answerable: bool = True,
    max_relevance_score: float | None = 0.6,
    first_relevant_rank: int | None = 1,
    matched: tuple[int, int, int] = (1, 1, 1),
    total_evidence_count: int = 1,
    category: str = "direct",
    difficulty: str = "easy",
    question: str = "一个问题？",
    sources: tuple[RetrievedSource, ...] | None = None,
) -> EvaluationResult:
    """Build one evaluation result without running retrieval."""
    if sources is None:
        sources = (
            ()
            if max_relevance_score is None
            else (make_source(1, max_relevance_score),)
        )
    if answerable:
        hit_1: bool | None = matched[0] > 0
        hit_3: bool | None = matched[1] > 0
        hit_5: bool | None = matched[2] > 0
        reciprocal: float | None = (
            0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        )
    else:
        hit_1 = hit_3 = hit_5 = None
        reciprocal = None
        first_relevant_rank = None
        matched = (0, 0, 0)
        total_evidence_count = 0

    return EvaluationResult(
        case_id=case_id,
        split=split,
        question=question,
        answerable=answerable,
        category=category,
        difficulty=difficulty,
        max_relevance_score=max_relevance_score,
        first_relevant_rank=first_relevant_rank,
        matched_evidence_count_at_1=matched[0],
        matched_evidence_count_at_3=matched[1],
        matched_evidence_count_at_5=matched[2],
        total_evidence_count=total_evidence_count,
        hit_at_1=hit_1,
        hit_at_3=hit_3,
        hit_at_5=hit_5,
        reciprocal_rank=reciprocal,
        sources=sources,
    )


ALPHA_TEXT = (
    "阿尔法文档讲解订单结算流程。订单结算流程由结算服务负责，"
    "结算服务读取订单明细并生成结算单。结算单包含金额、币种和状态三个字段。"
)
BETA_TEXT = (
    "贝塔文档讲解仓库补货策略。补货策略由补货服务计算，"
    "补货服务根据库存水位和销量预测生成补货建议。补货建议包含数量与到货日期。"
)


def write_mini_corpus(root: Path) -> Path:
    """Write a two-document corpus plus manifest and return the manifest path."""
    corpus_dir = root / "eval" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "alpha.md").write_text(ALPHA_TEXT, encoding="utf-8")
    (corpus_dir / "beta.md").write_text(BETA_TEXT, encoding="utf-8")

    manifest_path = root / "eval" / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "mini-alpha",
                        "path": "eval/corpus/alpha.md",
                        "filename": "alpha.md",
                        "content_type": "text/markdown",
                    },
                    {
                        "document_id": "mini-beta",
                        "path": "eval/corpus/beta.md",
                        "filename": "beta.md",
                        "content_type": "text/markdown",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


MINI_DATASET_ROWS: tuple[dict[str, object], ...] = (
    {
        "id": "mini-001",
        "split": "calibration",
        "question": "订单结算流程由谁负责？",
        "answerable": True,
        "category": "direct",
        "difficulty": "easy",
        "expected_evidence": [
            {
                "document_id": "mini-alpha",
                "page_number": None,
                "required_terms": ["订单结算流程由结算服务负责"],
            }
        ],
        "notes": "",
    },
    {
        "id": "mini-002",
        "split": "calibration",
        "question": "补货建议里包含哪些信息？",
        "answerable": True,
        "category": "paraphrase",
        "difficulty": "medium",
        "expected_evidence": [
            {
                "document_id": "mini-beta",
                "page_number": None,
                "required_terms": ["补货建议包含数量与到货日期"],
            }
        ],
        "notes": "",
    },
    {
        "id": "mini-003",
        "split": "calibration",
        "question": "今天晚饭吃什么比较好？",
        "answerable": False,
        "category": "out_of_scope_far",
        "difficulty": "easy",
        "expected_evidence": [],
        "notes": "",
    },
    {
        "id": "mini-004",
        "split": "test",
        "question": "结算单包含哪几个字段？",
        "answerable": True,
        "category": "direct",
        "difficulty": "easy",
        "expected_evidence": [
            {
                "document_id": "mini-alpha",
                "page_number": None,
                "required_terms": ["结算单包含金额、币种和状态三个字段"],
            }
        ],
        "notes": "",
    },
    {
        "id": "mini-005",
        "split": "test",
        "question": "补货服务的灰度发布方案是什么？",
        "answerable": False,
        "category": "out_of_scope_near",
        "difficulty": "hard",
        "expected_evidence": [],
        "notes": "",
    },
)


def write_mini_dataset(
    root: Path,
    rows: tuple[dict[str, object], ...] = MINI_DATASET_ROWS,
) -> Path:
    """Write a small JSONL dataset and return its path."""
    dataset_path = root / "eval" / "dataset.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return dataset_path
