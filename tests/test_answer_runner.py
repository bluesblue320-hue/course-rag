"""Tests for the answer evaluation runner (offline and live modes)."""

import json
from pathlib import Path

import pytest

from src.evaluation.answer_annotations import load_answer_annotations
from src.evaluation.answer_responses import load_answer_responses
from src.evaluation.answer_runner import (
    run_live_answer_evaluation,
    run_offline_answer_evaluation,
    write_responses_jsonl,
)
from src.evaluation.dataset import load_dataset
from src.exceptions import AnswerEvaluationError
from src.rag_service import INSUFFICIENT_CONTEXT_ANSWER
from tests.evaluation_helpers import FakeEmbeddingService

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "answer_evaluation"
DATASET_PATH = FIXTURE_ROOT / "dataset.jsonl"
ANNOTATIONS_PATH = FIXTURE_ROOT / "annotations.jsonl"
RESPONSES_PATH = FIXTURE_ROOT / "responses.jsonl"


class FakeGenerationService:
    """Deterministic fake returning citations based on the prompt question."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False
        self.close_count = 0
        self.fail_next = False

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.fail_next:
            raise RuntimeError("provider failed")
        if "Router" in prompt:
            return "Router 层不应该承载业务逻辑。[来源1]"
        if "RAG" in prompt:
            return "RAG 是检索增强生成。[来源1]"
        return "这是一个确定性回答。[来源1]"

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


def _load_fixture(tmp_path: Path):
    cases = load_dataset(DATASET_PATH)
    annotations = load_answer_annotations(ANNOTATIONS_PATH, cases)
    responses = load_answer_responses(RESPONSES_PATH, cases)
    return cases, annotations, responses


class TestOfflineRunner:
    def test_offline_does_not_require_models_and_builds_run(
        self, tmp_path: Path
    ) -> None:
        cases, annotations, responses = _load_fixture(tmp_path)
        run = run_offline_answer_evaluation(
            cases=cases,
            annotations=annotations,
            responses=responses,
            dataset_path="tests/fixtures/answer_evaluation/dataset.jsonl",
            annotations_path="tests/fixtures/answer_evaluation/annotations.jsonl",
            responses_path="tests/fixtures/answer_evaluation/responses.jsonl",
            run_name="local-fixture",
            model_label="deterministic-fixture",
        )
        assert run.configuration.mode == "offline"
        assert run.configuration.embedding_model is None
        assert run.configuration.top_k is None
        assert run.configuration.responses_path == (
            "tests/fixtures/answer_evaluation/responses.jsonl"
        )
        assert run.aggregate_metrics.case_count == 8
        assert len(run.case_metrics) == 8

    def test_mismatched_counts_are_rejected(self, tmp_path: Path) -> None:
        cases, annotations, _ = _load_fixture(tmp_path)
        with pytest.raises(AnswerEvaluationError):
            run_offline_answer_evaluation(
                cases=cases,
                annotations=annotations,
                responses=(),
                dataset_path="d",
                annotations_path="a",
                responses_path="r",
                run_name="x",
            )

    def test_responses_jsonl_roundtrip(self, tmp_path: Path) -> None:
        cases, annotations, responses = _load_fixture(tmp_path)
        run = run_offline_answer_evaluation(
            cases=cases,
            annotations=annotations,
            responses=responses,
            dataset_path="d",
            annotations_path="a",
            responses_path="r",
            run_name="x",
        )
        path = write_responses_jsonl(run.responses, tmp_path / "out")
        assert path.name == "responses.jsonl"
        roundtrip = load_answer_responses(path, cases)
        assert [response.case_id for response in roundtrip] == [
            case.id for case in cases
        ]


class TestLiveRunner:
    def _mini_corpus(self, tmp_path: Path):
        from src.evaluation.corpus import load_corpus, load_manifest

        corpus_dir = tmp_path / "eval" / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        (corpus_dir / "backend.md").write_text(
            "Router 层不应该承载业务逻辑。\nService 层负责业务逻辑。\n"
            "Repository 层负责数据访问。",
            encoding="utf-8",
        )
        (corpus_dir / "rag.md").write_text(
            "RAG 是检索增强生成。\n更换 Embedding 模型必须重建全部文档向量。",
            encoding="utf-8",
        )
        (corpus_dir / "database.md").write_text(
            "主键唯一标识表里的一行。\n"
            "连接数据库需要地址、端口、库名和账号密码。",
            encoding="utf-8",
        )
        manifest_path = tmp_path / "eval" / "corpus_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "documents": [
                        {
                            "document_id": "fixture-backend",
                            "path": "eval/corpus/backend.md",
                            "filename": "backend.md",
                            "content_type": "text/markdown",
                        },
                        {
                            "document_id": "fixture-rag",
                            "path": "eval/corpus/rag.md",
                            "filename": "rag.md",
                            "content_type": "text/markdown",
                        },
                        {
                            "document_id": "fixture-database",
                            "path": "eval/corpus/database.md",
                            "filename": "database.md",
                            "content_type": "text/markdown",
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        documents = load_manifest(manifest_path, base_dir=tmp_path)
        return load_corpus(documents)

    def test_live_mode_counts_embedding_and_generation_calls(
        self, tmp_path: Path
    ) -> None:
        cases, annotations, _ = _load_fixture(tmp_path)
        corpus = self._mini_corpus(tmp_path)
        embedding = FakeEmbeddingService()
        generation = FakeGenerationService()

        run = run_live_answer_evaluation(
            corpus=corpus,
            cases=cases,
            annotations=annotations,
            embedding_service=embedding,
            generation_service=generation,
            embedding_model="fake-bigram",
            top_k=3,
            threshold=0.35,
            dataset_path="d",
            annotations_path="a",
            run_name="live-fixture",
            model_label="fake",
            prompt_version="prompt-builder-v1",
        )
        # Documents encoded once, every question encoded once.
        assert embedding.document_batches == 1
        assert embedding.query_calls == len(cases)
        # Only answerable cases above threshold reach generate().
        generated = len(generation.calls)
        assert generated >= 1
        assert run.configuration.mode == "live"
        assert generation.closed
        assert generation.close_count == 1

    def test_live_mode_closes_generation_on_failure(self, tmp_path: Path) -> None:
        cases, annotations, _ = _load_fixture(tmp_path)
        corpus = self._mini_corpus(tmp_path)
        embedding = FakeEmbeddingService()
        generation = FakeGenerationService()
        generation.fail_next = True

        with pytest.raises(RuntimeError):
            run_live_answer_evaluation(
                corpus=corpus,
                cases=cases,
                annotations=annotations,
                embedding_service=embedding,
                generation_service=generation,
                embedding_model="fake-bigram",
                top_k=3,
                threshold=0.35,
                dataset_path="d",
                annotations_path="a",
                run_name="live-fixture",
                model_label="fake",
                prompt_version="prompt-builder-v1",
            )
        assert generation.closed

    def test_insufficient_context_does_not_call_generate(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "only-refusal",
                    "split": "test",
                    "question": "宫保鸡丁怎么做好吃？",
                    "answerable": False,
                    "category": "out_of_scope_far",
                    "difficulty": "easy",
                    "expected_evidence": [],
                    "notes": "",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        annotations_path = tmp_path / "annotations.jsonl"
        annotations_path.write_text(
            json.dumps(
                {
                    "case_id": "only-refusal",
                    "reference_answer": INSUFFICIENT_CONTEXT_ANSWER,
                    "required_facts": [],
                    "forbidden_phrases": [],
                    "notes": "",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        cases = load_dataset(dataset_path)
        annotations = load_answer_annotations(annotations_path, cases)
        corpus = self._mini_corpus(tmp_path)
        embedding = FakeEmbeddingService()
        generation = FakeGenerationService()

        run = run_live_answer_evaluation(
            corpus=corpus,
            cases=cases,
            annotations=annotations,
            embedding_service=embedding,
            generation_service=generation,
            embedding_model="fake-bigram",
            top_k=3,
            threshold=0.99,
            dataset_path="d",
            annotations_path="a",
            run_name="live-fixture",
            model_label="fake",
            prompt_version="prompt-builder-v1",
        )
        assert run.responses[0].answer_status == "insufficient_context"
        assert run.responses[0].answer == INSUFFICIENT_CONTEXT_ANSWER
        assert generation.calls == []

    def test_invalid_threshold_is_rejected(self, tmp_path: Path) -> None:
        cases, annotations, _ = _load_fixture(tmp_path)
        corpus = self._mini_corpus(tmp_path)
        with pytest.raises(AnswerEvaluationError):
            run_live_answer_evaluation(
                corpus=corpus,
                cases=cases,
                annotations=annotations,
                embedding_service=FakeEmbeddingService(),
                generation_service=FakeGenerationService(),
                embedding_model="fake-bigram",
                top_k=3,
                threshold=1.5,
                dataset_path="d",
                annotations_path="a",
                run_name="x",
                model_label="fake",
                prompt_version="p",
            )
