"""Offline tests for the evaluation runner with fake embeddings."""

import json
from pathlib import Path

import pytest

from src.document_chunker import chunk_document as production_chunk_document
from src.document_loaders import MarkdownDocumentLoader
from src.evaluation.models import EvaluationDataError
from src.evaluation.runner import EvaluationRunner

from evaluation_helpers import (
    FakeEmbeddingService,
    make_case,
    write_corpus,
    write_dataset,
)


def _default_corpus(tmp_path: Path) -> Path:
    return write_corpus(
        tmp_path,
        {
            "eval-doc-a": ["alpha 服务负责核心业务逻辑"],
            "eval-doc-b": ["beta 数据库负责数据访问"],
        },
    )


def _default_dataset(tmp_path: Path) -> Path:
    return write_dataset(
        tmp_path,
        [
            make_case("q001", question="alpha 服务是什么？"),
            make_case("q002", question="beta 数据库是什么？", evidence=[("eval-doc-b", ("beta",))]),
            make_case(
                "q003",
                question="alpha 与 beta 的关系？",
                category="multi_evidence",
                evidence=[("eval-doc-a", ("alpha",)), ("eval-doc-b", ("beta",))],
            ),
            make_case(
                "q004",
                question="今天的天气怎么样？",
                answerable=False,
                category="out_of_scope_far",
                evidence=[],
            ),
            make_case(
                "q005",
                split="test",
                question="beta 的存储位置？",
                evidence=[("eval-doc-b", ("beta",))],
            ),
            make_case(
                "q006",
                split="test",
                question="股票明天涨跌？",
                answerable=False,
                category="out_of_scope_far",
                evidence=[],
            ),
        ],
    )


def test_runner_uses_production_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)
    calls: list[Path] = []
    original_load = MarkdownDocumentLoader.load

    def recording_load(self: object, file_path: Path) -> object:
        calls.append(file_path)
        return original_load(self, file_path)

    monkeypatch.setattr(MarkdownDocumentLoader, "load", recording_load)

    runner = EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path)
    runner.run(manifest, dataset, top_k=5)

    assert len(calls) == 2


def test_runner_uses_production_chunk_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)
    calls: list[object] = []
    original = production_chunk_document

    def recording(document: object, document_id: str, filename: str) -> object:
        calls.append(document_id)
        return original(document, document_id, filename)

    monkeypatch.setattr(
        "src.evaluation.corpus.chunk_document",
        recording,
    )

    runner = EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path)
    runner.run(manifest, dataset, top_k=5)

    assert sorted(calls) == ["eval-doc-a", "eval-doc-b"]


def test_runner_uses_knowledge_index_and_embeds_documents_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.knowledge_index import KnowledgeIndex

    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)
    embedding = FakeEmbeddingService()
    search_calls: list[tuple[object, int]] = []

    original_search = KnowledgeIndex.search

    def recording_search(
        self: object,
        query_embedding: object,
        top_k: int,
    ) -> list[dict[str, object]]:
        search_calls.append((query_embedding, top_k))
        return original_search(self, query_embedding, top_k)

    monkeypatch.setattr(KnowledgeIndex, "search", recording_search)

    runner = EvaluationRunner(embedding, base_dir=tmp_path)
    runner.run(manifest, dataset, top_k=5)

    assert embedding.document_encode_count == 1
    assert embedding.query_encode_count == 6
    assert len(search_calls) == 6
    assert all(top_k == 5 for _, top_k in search_calls)


def test_threshold_scan_does_not_re_retrieve_or_re_embed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.knowledge_index import KnowledgeIndex

    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)
    embedding = FakeEmbeddingService()
    search_calls: list[object] = []
    original_search = KnowledgeIndex.search

    def recording_search(self: object, query_embedding: object, top_k: int) -> list[dict[str, object]]:
        search_calls.append(query_embedding)
        return original_search(self, query_embedding, top_k)

    monkeypatch.setattr(KnowledgeIndex, "search", recording_search)

    runner = EvaluationRunner(embedding, base_dir=tmp_path)
    runner.run(
        manifest,
        dataset,
        top_k=5,
        threshold_start=0.20,
        threshold_end=0.60,
        threshold_step=0.01,
    )

    assert len(search_calls) == 6
    assert embedding.query_encode_count == 6
    assert embedding.document_encode_count == 1


def test_runner_never_initializes_generation(
    tmp_path: Path,
) -> None:
    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)

    runner = EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path)
    report_data = runner.run(manifest, dataset, top_k=5)

    assert "generation" not in report_data
    from src.evaluation import runner as runner_module

    assert not hasattr(runner_module, "GenerationService")
    assert not hasattr(runner_module, "PromptBuilder")


def test_runner_produces_stable_ordered_results(tmp_path: Path) -> None:
    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)

    first = EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path).run(manifest, dataset, top_k=5)
    second = EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path).run(manifest, dataset, top_k=5)

    assert first == second


def test_calibration_and_test_are_separated(tmp_path: Path) -> None:
    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)

    report_data = EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path).run(
        manifest, dataset, top_k=5
    )

    assert report_data["split_counts"] == {"calibration": 4, "test": 2}
    assert set(report_data["failure_case_ids"].keys()) == {
        "retrieval_failures",
        "false_refusals",
        "false_answers",
    }


def test_report_uses_fixed_recommended_threshold_for_test(tmp_path: Path) -> None:
    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)

    report_data = EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path).run(
        manifest,
        dataset,
        top_k=5,
        current_threshold=0.35,
    )

    test_block = report_data["test_metrics_recommended"]
    assert test_block["decision"]["tp"] + test_block["decision"]["fp"] + (
        test_block["decision"]["tn"] + test_block["decision"]["fn"]
    ) == report_data["split_counts"]["test"]
    scanned_thresholds = [
        row["threshold"]
        for row in report_data["calibration_threshold_table"]
    ]
    assert report_data["recommended_threshold"] in scanned_thresholds


def test_runner_writes_reports(tmp_path: Path) -> None:
    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)
    output_dir = tmp_path / "out"

    EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path).run(
        manifest,
        dataset,
        top_k=5,
        output_dir=output_dir,
    )

    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "cases.csv").is_file()
    assert (output_dir / "report.md").is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "1.0"


def test_runner_rejects_invalid_top_k(tmp_path: Path) -> None:
    manifest = _default_corpus(tmp_path)
    dataset = _default_dataset(tmp_path)

    with pytest.raises(EvaluationDataError, match="top_k"):
        EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path).run(manifest, dataset, top_k=0)


def test_runner_failure_cases_are_generated(tmp_path: Path) -> None:
    manifest = write_corpus(
        tmp_path,
        {
            "eval-doc-a": ["alpha 服务负责核心业务逻辑。" * 150],
            "eval-doc-b": ["beta 数据库负责数据访问"],
        },
    )
    dataset = write_dataset(
        tmp_path,
        [
            make_case("q001", question="完全不相关的提问？", evidence=[("eval-doc-b", ("beta",))]),
            make_case(
                "q002",
                question="alpha 内容的边界在哪里？",
                answerable=False,
                category="out_of_scope_near",
                evidence=[],
            ),
        ],
    )

    report_data = EvaluationRunner(FakeEmbeddingService(), base_dir=tmp_path).run(
        manifest, dataset, top_k=5
    )

    assert "q001" in report_data["failure_case_ids"]["retrieval_failures"]
    assert "q002" in report_data["failure_case_ids"]["false_answers"]
