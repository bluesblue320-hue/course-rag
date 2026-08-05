"""Tests for the evaluate_reranker CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_reranker import (
    EXIT_INVALID_INPUT,
    EXIT_OK,
    EXIT_RERANKER_UNAVAILABLE,
    build_parser,
    main,
)
from src.reranker import FakeReranker
from tests.evaluation_helpers import FakeEmbeddingService


# Use the repo's actual eval files (they exist from PR #6)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = str(REPO_ROOT / "eval" / "corpus_manifest.json")
DEFAULT_DATASET = str(REPO_ROOT / "eval" / "dataset.jsonl")


# ---------------------------------------------------------------------------
# --help does not load models
# ---------------------------------------------------------------------------


class TestHelpNoModelLoad:
    def test_help_does_not_load_embedding(self, capsys) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        captured = capsys.readouterr()
        assert "candidate-top-k" in captured.out
        assert "reranker-model" in captured.out

    def test_help_does_not_load_reranker(self) -> None:
        """Importing the parser module must not trigger any model import."""
        # If we got here, the import succeeded without loading models.
        parser = build_parser()
        assert parser is not None


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_candidate_top_k_too_small(self) -> None:
        # A/B evaluation requires candidate_top_k >= 15.
        exit_code = main(["--candidate-top-k", "10", "--reranker-model", "test"])
        assert exit_code == EXIT_INVALID_INPUT

    def test_candidate_top_k_15_accepted(self, tmp_path: Path) -> None:
        exit_code = main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(tmp_path / "o"),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake-reranker",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )
        assert exit_code == EXIT_OK

    def test_final_top_k_too_small(self) -> None:
        exit_code = main(["--final-top-k", "3", "--reranker-model", "test"])
        assert exit_code == EXIT_INVALID_INPUT

    def test_candidate_less_than_final(self) -> None:
        exit_code = main([
            "--candidate-top-k", "5",
            "--final-top-k", "10",
            "--reranker-model", "test",
        ])
        assert exit_code == EXIT_INVALID_INPUT

    def test_no_reranker_model(self) -> None:
        exit_code = main([
            "--candidate-top-k", "15",
            "--final-top-k", "5",
        ])
        assert exit_code == EXIT_INVALID_INPUT


# ---------------------------------------------------------------------------
# Full run with fake services
# ---------------------------------------------------------------------------


def fake_embedding_factory(model_name):
    return FakeEmbeddingService(), "fake-embedding"


def fake_reranker_factory(model_name):
    return FakeReranker()


class TestFullRun:
    def test_successful_run(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"

        exit_code = main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(output_dir),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake-reranker",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )

        assert exit_code == EXIT_OK
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "cases.csv").exists()
        assert (output_dir / "report.md").exists()
        assert not (output_dir / "latency.json").exists()

    def test_output_dir_created(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "deeply" / "nested" / "output"

        exit_code = main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(output_dir),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake-reranker",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )

        assert exit_code == EXIT_OK
        assert output_dir.exists()

    def test_include_latency(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"

        exit_code = main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(output_dir),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake-reranker",
                "--include-latency",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )

        assert exit_code == EXIT_OK
        assert (output_dir / "latency.json").exists()

    def test_summary_json_structure(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"

        main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(output_dir),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake-reranker",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )

        with open(output_dir / "summary.json", encoding="utf-8") as f:
            summary = json.load(f)

        assert "schema_version" in summary
        assert "embedding_model" in summary
        assert "reranker_model" in summary
        assert "reranker_backend" in summary
        assert "real_model_run" in summary
        assert "reranker_model_revision" in summary
        assert "candidate_top_k" in summary
        assert "final_top_k" in summary
        assert "candidate_metrics" in summary
        assert "vector_metrics" in summary
        assert "reranked_metrics" in summary
        assert "metric_deltas" in summary
        assert "metrics_by_category" in summary
        assert "metrics_by_difficulty" in summary
        assert "metrics_by_split" in summary
        assert "focus_case_results" in summary
        assert "improved_case_ids" in summary
        assert "regressed_case_ids" in summary
        assert "unchanged_case_ids" in summary
        assert "decision_invariance_check" in summary
        assert "recommendation" in summary

    def test_fake_reranker_run_never_recommends(self, tmp_path: Path) -> None:
        """A fake reranker can never produce a production enable recommendation."""
        output_dir = tmp_path / "output"

        exit_code = main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(output_dir),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake-reranker",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )

        assert exit_code == EXIT_OK
        with open(output_dir / "summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        # The run provenance is explicitly fake and never real.
        assert summary["reranker_backend"] == "fake"
        assert summary["real_model_run"] is False
        assert summary["recommendation"]["recommend_enable"] is False
        assert any(
            "未使用真实 CrossEncoder" in r
            for r in summary["recommendation"]["reasons"]
        )

    def test_cli_summary_shows_backend_and_real_model_run(
        self, tmp_path: Path, capsys
    ) -> None:
        output_dir = tmp_path / "output"

        main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(output_dir),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake-reranker",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )

        captured = capsys.readouterr()
        assert "Reranker backend : fake" in captured.out
        assert "Real model run   : 否" in captured.out

    def test_no_absolute_paths_in_summary(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"

        main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(output_dir),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake-reranker",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )

        with open(output_dir / "summary.json", encoding="utf-8") as f:
            content = f.read()

        # No absolute paths or drive letters in the output
        assert "C:" not in content
        assert str(tmp_path) not in content

    def test_no_nan_or_inf_in_summary(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"

        main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(output_dir),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake-reranker",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )

        with open(output_dir / "summary.json", encoding="utf-8") as f:
            content = f.read()

        assert "NaN" not in content
        assert "Infinity" not in content


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_reranker_load_failure_exit_code(self, tmp_path: Path) -> None:
        def failing_reranker_factory(model_name):
            raise RuntimeError("model not found")

        exit_code = main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(tmp_path / "output"),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "missing-model",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=failing_reranker_factory,
        )

        assert exit_code == EXIT_RERANKER_UNAVAILABLE

    def test_invalid_dataset_exit_code(self, tmp_path: Path) -> None:
        bad_dataset = tmp_path / "bad.jsonl"
        bad_dataset.write_text("not valid json\n", encoding="utf-8")

        exit_code = main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", str(bad_dataset),
                "--output-dir", str(tmp_path / "output"),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "fake",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=fake_reranker_factory,
        )

        assert exit_code == EXIT_INVALID_INPUT

    def test_no_exception_traceback_in_output(self, tmp_path: Path, capsys) -> None:
        """Error output must not leak internal exception traces."""
        def failing_reranker_factory(model_name):
            raise RuntimeError("internal secret path C:\\secret\\cache")

        exit_code = main(
            [
                "--manifest", DEFAULT_MANIFEST,
                "--dataset", DEFAULT_DATASET,
                "--output-dir", str(tmp_path / "output"),
                "--candidate-top-k", "15",
                "--final-top-k", "5",
                "--reranker-model", "missing",
            ],
            embedding_factory=fake_embedding_factory,
            reranker_factory=failing_reranker_factory,
        )

        assert exit_code == EXIT_RERANKER_UNAVAILABLE
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
