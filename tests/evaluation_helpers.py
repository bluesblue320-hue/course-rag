"""Shared offline helpers for evaluation tests."""

import json
from pathlib import Path

import numpy as np

from src.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvidenceExpectation,
)


class FakeEmbeddingService:
    """Keyword-based fake embedding with call accounting."""

    model_name = "fake/eval-embedding"

    def __init__(self) -> None:
        self.document_encode_count = 0
        self.query_encode_count = 0
        self.encoded_queries: list[str] = []

    def _vector_for(self, text: str) -> list[float]:
        if "alpha" in text:
            return [1.0, 0.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0, 0.0]
        if "gamma" in text:
            return [0.0, 0.0, 1.0]
        return [0.5, 0.5, 0.5]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_encode_count += 1
        return np.asarray([self._vector_for(text) for text in texts])

    def encode_query(self, query: str) -> np.ndarray:
        self.query_encode_count += 1
        self.encoded_queries.append(query)
        return np.asarray(self._vector_for(query))


def write_corpus(
    tmp_path: Path,
    documents: dict[str, list[str]],
    suffix: str = ".md",
) -> Path:
    """Write one manifest and one file per document into a temp directory.

    Each document value is a list of paragraphs; the whole document text is
    the paragraphs joined by blank lines. Short paragraphs keep the corpus
    under the default chunk window for deterministic single-chunk documents.
    """
    manifest_records = []
    for document_id, paragraphs in documents.items():
        filename = f"{document_id}{suffix}"
        content = "\n\n".join(paragraphs)
        (tmp_path / filename).write_text(content, encoding="utf-8")
        manifest_records.append(
            {
                "document_id": document_id,
                "path": filename,
                "filename": filename,
                "content_type": "text/markdown",
            }
        )
    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps({"documents": manifest_records}, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def write_dataset(
    tmp_path: Path,
    cases: list[dict[str, object]],
) -> Path:
    """Write one JSONL dataset file with one case per line."""
    dataset_path = tmp_path / "dataset.jsonl"
    lines = [json.dumps(case, ensure_ascii=False) for case in cases]
    dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset_path


def make_case(
    case_id: str,
    *,
    split: str = "calibration",
    question: str = "alpha 服务是什么？",
    answerable: bool = True,
    category: str = "direct",
    difficulty: str = "easy",
    evidence: list[tuple[str, tuple[str, ...]]] | None = None,
    notes: str = "",
) -> dict[str, object]:
    """Build one raw dataset case dictionary with valid defaults."""
    if evidence is None and answerable:
        evidence = [("eval-doc-a", ("alpha",))]
    expected = [
        {
            "document_id": document_id,
            "page_number": None,
            "required_terms": list(terms),
        }
        for document_id, terms in (evidence or [])
    ]
    return {
        "id": case_id,
        "split": split,
        "question": question,
        "answerable": answerable,
        "category": category,
        "difficulty": difficulty,
        "expected_evidence": expected,
        "notes": notes,
    }


def make_evaluation_case(
    case_id: str,
    *,
    split: str = "calibration",
    question: str = "问题",
    answerable: bool = True,
    category: str = "direct",
    difficulty: str = "easy",
    expected_evidence: tuple[EvidenceExpectation, ...] = (
        EvidenceExpectation(document_id="eval-doc-a", page_number=None, required_terms=("alpha",)),
    ),
) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        split=split,  # type: ignore[arg-type]
        question=question,
        answerable=answerable,
        category=category,
        difficulty=difficulty,
        expected_evidence=expected_evidence,
        notes="",
    )


def make_result(
    case_id: str,
    *,
    split: str = "calibration",
    question: str = "问题",
    answerable: bool = True,
    category: str = "direct",
    difficulty: str = "easy",
    max_score: float | None = 0.6,
    first_relevant_rank: int | None = 1,
    matched_at_1: int = 1,
    matched_at_3: int = 1,
    matched_at_5: int = 1,
    total_evidence: int = 1,
    reciprocal_rank: float | None = 1.0,
    hit_at_1: bool | None = None,
    hit_at_3: bool | None = None,
    hit_at_5: bool | None = None,
    sources: tuple[dict[str, object], ...] = (),
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        split=split,
        question=question,
        answerable=answerable,
        category=category,
        difficulty=difficulty,
        max_relevance_score=max_score,
        first_relevant_rank=first_relevant_rank,
        matched_evidence_count_at_1=matched_at_1,
        matched_evidence_count_at_3=matched_at_3,
        matched_evidence_count_at_5=matched_at_5,
        total_evidence_count=total_evidence,
        hit_at_1=(
            matched_at_1 > 0
            if hit_at_1 is None
            else (hit_at_1 if answerable else None)
        ),
        hit_at_3=(
            matched_at_3 > 0
            if hit_at_3 is None
            else (hit_at_3 if answerable else None)
        ),
        hit_at_5=(
            matched_at_5 > 0
            if hit_at_5 is None
            else (hit_at_5 if answerable else None)
        ),
        reciprocal_rank=reciprocal_rank,
        sources=sources,
    )
