"""Unit tests for the reranker abstraction."""

from __future__ import annotations

import math
import os

import pytest

from src.reranker import (
    CrossEncoderReranker,
    FakeReranker,
    RERANKER_CANDIDATE_TOP_K_ERROR,
    RERANKER_ENABLED_ERROR,
    RERANKER_MODEL_ERROR,
    LOCAL_MODEL_DISPLAY,
    RerankInput,
    RerankScore,
    RerankerConfig,
    _to_finite_float,
    _validate_rerank_scores,
    requested_reranker_enabled,
    resolve_reranker_config,
    safe_rerank,
    safe_reranker_model_display_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(*pairs: tuple[str, str]) -> list[RerankInput]:
    return [RerankInput(candidate_id=cid, text=text) for cid, text in pairs]


# ---------------------------------------------------------------------------
# RerankInput / RerankScore
# ---------------------------------------------------------------------------


class TestRerankInput:
    def test_is_frozen(self) -> None:
        ri = RerankInput(candidate_id="c1", text="hello")
        with pytest.raises(AttributeError):
            ri.candidate_id = "c2"  # type: ignore[misc]

    def test_fields(self) -> None:
        ri = RerankInput(candidate_id="c1", text="hello")
        assert ri.candidate_id == "c1"
        assert ri.text == "hello"


class TestRerankScore:
    def test_is_frozen(self) -> None:
        rs = RerankScore(candidate_id="c1", score=0.5)
        with pytest.raises(AttributeError):
            rs.score = 0.9  # type: ignore[misc]

    def test_fields(self) -> None:
        rs = RerankScore(candidate_id="c1", score=0.5)
        assert rs.candidate_id == "c1"
        assert rs.score == 0.5


# ---------------------------------------------------------------------------
# _to_finite_float
# ---------------------------------------------------------------------------


class TestToFiniteFloat:
    def test_int(self) -> None:
        assert _to_finite_float(42) == 42.0

    def test_float(self) -> None:
        assert _to_finite_float(3.14) == pytest.approx(3.14)

    def test_numpy_float(self) -> None:
        import numpy as np
        assert _to_finite_float(np.float64(0.5)) == 0.5

    def test_bool_rejected(self) -> None:
        with pytest.raises(ValueError, match="布尔值"):
            _to_finite_float(True)

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="有限数"):
            _to_finite_float(float("nan"))

    def test_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="有限数"):
            _to_finite_float(float("inf"))

    def test_neg_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="有限数"):
            _to_finite_float(float("-inf"))

    def test_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="不是数字"):
            _to_finite_float("not-a-number")

    def test_none_rejected(self) -> None:
        with pytest.raises(ValueError, match="不是数字"):
            _to_finite_float(None)


# ---------------------------------------------------------------------------
# _validate_rerank_scores
# ---------------------------------------------------------------------------


class TestValidateRerankScores:
    def test_empty(self) -> None:
        result = _validate_rerank_scores([], [])
        assert result == ()

    def test_single(self) -> None:
        candidates = make_inputs(("c1", "text1"))
        scores = [RerankScore(candidate_id="c1", score=0.5)]
        result = _validate_rerank_scores(candidates, scores)
        assert len(result) == 1
        assert result[0].candidate_id == "c1"
        assert result[0].score == 0.5

    def test_multiple(self) -> None:
        candidates = make_inputs(("c1", "t1"), ("c2", "t2"), ("c3", "t3"))
        scores = [
            RerankScore(candidate_id="c1", score=0.9),
            RerankScore(candidate_id="c2", score=0.5),
            RerankScore(candidate_id="c3", score=0.1),
        ]
        result = _validate_rerank_scores(candidates, scores)
        assert len(result) == 3

    def test_count_mismatch(self) -> None:
        candidates = make_inputs(("c1", "t1"), ("c2", "t2"))
        scores = [RerankScore(candidate_id="c1", score=0.5)]
        with pytest.raises(ValueError, match="返回数量不正确"):
            _validate_rerank_scores(candidates, scores)

    def test_duplicate_id(self) -> None:
        candidates = make_inputs(("c1", "t1"), ("c2", "t2"))
        scores = [
            RerankScore(candidate_id="c1", score=0.5),
            RerankScore(candidate_id="c1", score=0.9),
        ]
        with pytest.raises(ValueError, match="重复"):
            _validate_rerank_scores(candidates, scores)

    def test_unknown_id(self) -> None:
        candidates = make_inputs(("c1", "t1"))
        scores = [RerankScore(candidate_id="c99", score=0.5)]
        with pytest.raises(ValueError, match="未知"):
            _validate_rerank_scores(candidates, scores)

    def test_nan_score(self) -> None:
        candidates = make_inputs(("c1", "t1"))
        scores = [RerankScore(candidate_id="c1", score=float("nan"))]
        with pytest.raises(ValueError, match="有限数"):
            _validate_rerank_scores(candidates, scores)

    def test_inf_score(self) -> None:
        candidates = make_inputs(("c1", "t1"))
        scores = [RerankScore(candidate_id="c1", score=float("inf"))]
        with pytest.raises(ValueError, match="有限数"):
            _validate_rerank_scores(candidates, scores)


# ---------------------------------------------------------------------------
# FakeReranker
# ---------------------------------------------------------------------------


class TestFakeReranker:
    def test_empty_candidates(self) -> None:
        r = FakeReranker()
        result = r.rerank("query", [])
        assert result == ()

    def test_single_candidate(self) -> None:
        r = FakeReranker()
        candidates = make_inputs(("c1", "text1"))
        result = r.rerank("query", candidates)
        assert len(result) == 1
        assert result[0].candidate_id == "c1"

    def test_multiple_candidates(self) -> None:
        r = FakeReranker()
        candidates = make_inputs(("c1", "t1"), ("c2", "t2"), ("c3", "t3"))
        result = r.rerank("query", candidates)
        assert len(result) == 3
        ids = {s.candidate_id for s in result}
        assert ids == {"c1", "c2", "c3"}

    def test_returns_all_candidates_exactly_once(self) -> None:
        r = FakeReranker()
        candidates = make_inputs(("a", "x"), ("b", "y"), ("c", "z"))
        result = r.rerank("query", candidates)
        assert len(result) == len(candidates)
        result_ids = [s.candidate_id for s in result]
        assert len(set(result_ids)) == len(result_ids)

    def test_score_map_overrides_default(self) -> None:
        r = FakeReranker(score_map={"c2": 10.0, "c1": 5.0})
        candidates = make_inputs(("c1", "t1"), ("c2", "t2"))
        result = r.rerank("query", candidates)
        score_map = {s.candidate_id: s.score for s in result}
        assert score_map["c1"] == 5.0
        assert score_map["c2"] == 10.0

    def test_default_preserves_order(self) -> None:
        r = FakeReranker()
        candidates = make_inputs(("c1", "t1"), ("c2", "t2"), ("c3", "t3"))
        result = r.rerank("query", candidates)
        # Default: earlier = higher score
        assert result[0].score > result[1].score > result[2].score

    def test_fail_mode(self) -> None:
        r = FakeReranker(fail=True)
        candidates = make_inputs(("c1", "t1"))
        with pytest.raises(RuntimeError, match="模拟失败"):
            r.rerank("query", candidates)

    def test_call_count(self) -> None:
        r = FakeReranker()
        candidates = make_inputs(("c1", "t1"))
        assert r.call_count == 0
        r.rerank("q", candidates)
        assert r.call_count == 1
        r.rerank("q", candidates)
        assert r.call_count == 2

    def test_scores_not_in_0_1(self) -> None:
        """Reranker scores are not assumed to be probabilities."""
        r = FakeReranker(score_map={"c1": 15.7, "c2": -3.2})
        candidates = make_inputs(("c1", "t1"), ("c2", "t2"))
        result = r.rerank("query", candidates)
        scores = {s.candidate_id: s.score for s in result}
        assert scores["c1"] == 15.7
        assert scores["c2"] == -3.2


# ---------------------------------------------------------------------------
# safe_rerank
# ---------------------------------------------------------------------------


class TestSafeRerank:
    def test_success(self) -> None:
        r = FakeReranker()
        candidates = make_inputs(("c1", "t1"), ("c2", "t2"))
        scores, fallback = safe_rerank(r, "query", candidates)
        assert fallback is False
        assert len(scores) == 2

    def test_failure_fallback(self) -> None:
        r = FakeReranker(fail=True)
        candidates = make_inputs(("c1", "t1"))
        scores, fallback = safe_rerank(r, "query", candidates)
        assert fallback is True
        assert scores == ()

    def test_empty_candidates(self) -> None:
        r = FakeReranker()
        scores, fallback = safe_rerank(r, "query", [])
        assert fallback is False
        assert scores == ()


# ---------------------------------------------------------------------------
# resolve_reranker_config
# ---------------------------------------------------------------------------


class TestResolveRerankerConfig:
    def test_default_disabled(self) -> None:
        config = resolve_reranker_config(final_top_k=5)
        assert config.enabled is False
        assert config.model_name == ""
        assert config.candidate_top_k == 15

    def test_enabled_with_model(self) -> None:
        config = resolve_reranker_config(
            explicit_enabled="true",
            explicit_model="cross-encoder/test",
            explicit_candidate_top_k=20,
            final_top_k=5,
        )
        assert config.enabled is True
        assert config.model_name == "cross-encoder/test"
        assert config.candidate_top_k == 20

    def test_enabled_without_model_raises(self) -> None:
        with pytest.raises(ValueError, match="不能为空"):
            resolve_reranker_config(
                explicit_enabled="true",
                explicit_model="",
                final_top_k=5,
            )

    def test_invalid_enabled_raises(self) -> None:
        with pytest.raises(ValueError, match="布尔值"):
            resolve_reranker_config(
                explicit_enabled="maybe",
                final_top_k=5,
            )

    def test_candidate_top_k_too_small(self) -> None:
        with pytest.raises(ValueError, match="大于等于"):
            resolve_reranker_config(
                explicit_enabled="false",
                explicit_candidate_top_k=3,
                final_top_k=5,
            )

    def test_candidate_top_k_too_large(self) -> None:
        with pytest.raises(ValueError, match="不能超过"):
            resolve_reranker_config(
                explicit_enabled="false",
                explicit_candidate_top_k=200,
                final_top_k=5,
            )

    def test_candidate_top_k_less_than_final(self) -> None:
        with pytest.raises(ValueError, match="final_top_k"):
            resolve_reranker_config(
                explicit_enabled="false",
                explicit_candidate_top_k=5,
                final_top_k=10,
            )

    def test_env_var_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_RERANKER_ENABLED", "true")
        monkeypatch.setenv("RAG_RERANKER_MODEL", "test-model")
        monkeypatch.setenv("RAG_RERANKER_CANDIDATE_TOP_K", "20")
        config = resolve_reranker_config(final_top_k=5)
        assert config.enabled is True
        assert config.model_name == "test-model"
        assert config.candidate_top_k == 20

    def test_env_var_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_RERANKER_ENABLED", "false")
        monkeypatch.delenv("RAG_RERANKER_MODEL", raising=False)
        monkeypatch.delenv("RAG_RERANKER_CANDIDATE_TOP_K", raising=False)
        config = resolve_reranker_config(final_top_k=5)
        assert config.enabled is False


# ---------------------------------------------------------------------------
# CrossEncoderReranker (lazy import / no model at import)
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerImport:
    def test_module_import_does_not_load_model(self) -> None:
        """Importing src.reranker must not instantiate any model."""
        import importlib
        import src.reranker
        importlib.reload(src.reranker)
        # If we get here without error, the import is safe.
        assert hasattr(src.reranker, "CrossEncoderReranker")

    def test_construction_requires_model_name(self) -> None:
        with pytest.raises(ValueError, match="不能为空"):
            CrossEncoderReranker("")

    def test_construction_raises_on_offline_stub(self) -> None:
        """The conftest stub raises AssertionError; the reranker must propagate it."""
        with pytest.raises((AssertionError, RuntimeError)):
            CrossEncoderReranker("fake-model")

    def test_construction_passes_local_files_only(self, monkeypatch) -> None:
        """CrossEncoder must be loaded with local_files_only=True (no network)."""
        import sentence_transformers

        captured: dict[str, object] = {}

        class _RecordingCrossEncoder:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                captured["model_name"] = model_name
                captured.update(kwargs)

        monkeypatch.setattr(
            sentence_transformers, "CrossEncoder", _RecordingCrossEncoder
        )

        CrossEncoderReranker("local-zh-model")

        assert captured.get("model_name") == "local-zh-model"
        assert captured.get("local_files_only") is True

    def test_construction_does_not_use_network_on_missing_model(
        self, monkeypatch
    ) -> None:
        """A missing local model must fail without reaching the network."""
        import sentence_transformers

        class _OfflineOnlyCrossEncoder:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                # Simulate HF offline failure; must NOT attempt a download.
                if kwargs.get("local_files_only") is not True:
                    raise AssertionError("network download would have been attempted")
                raise FileNotFoundError(f"model not cached locally: {model_name}")

        monkeypatch.setattr(
            sentence_transformers, "CrossEncoder", _OfflineOnlyCrossEncoder
        )

        with pytest.raises(FileNotFoundError):
            CrossEncoderReranker("not-cached-model")


# ---------------------------------------------------------------------------
# Health-safe model display name (Fix 3, round 2)
# ---------------------------------------------------------------------------


class TestSafeRerankerModelDisplayName:
    def test_empty_returns_none(self) -> None:
        assert safe_reranker_model_display_name(None) is None
        assert safe_reranker_model_display_name("") is None
        assert safe_reranker_model_display_name("   ") is None

    def test_huggingface_id_kept(self) -> None:
        assert (
            safe_reranker_model_display_name("BAAI/bge-reranker-v2-m3")
            == "BAAI/bge-reranker-v2-m3"
        )

    def test_plain_name_kept(self) -> None:
        assert safe_reranker_model_display_name("local-zh-model") == "local-zh-model"

    def test_posix_absolute_path_redacted(self) -> None:
        raw = "/home/user/.cache/models/my-reranker"
        assert safe_reranker_model_display_name(raw) == LOCAL_MODEL_DISPLAY

    def test_windows_absolute_path_redacted(self) -> None:
        raw = "C:\\Users\\name\\.cache\\models\\my-reranker"
        assert safe_reranker_model_display_name(raw) == LOCAL_MODEL_DISPLAY

    def test_windows_forward_slash_path_redacted(self) -> None:
        raw = "C:/Users/name/.cache/models/my-reranker"
        assert safe_reranker_model_display_name(raw) == LOCAL_MODEL_DISPLAY

    def test_unc_path_redacted(self) -> None:
        raw = "\\\\server\\share\\models\\my-reranker"
        assert safe_reranker_model_display_name(raw) == LOCAL_MODEL_DISPLAY

    def test_file_uri_redacted(self) -> None:
        raw = "file:///home/user/.cache/models/my-reranker"
        assert safe_reranker_model_display_name(raw) == LOCAL_MODEL_DISPLAY

    def test_tilde_path_redacted(self) -> None:
        raw = "~/.cache/models/my-reranker"
        assert safe_reranker_model_display_name(raw) == LOCAL_MODEL_DISPLAY

    def test_no_username_or_drive_leak(self) -> None:
        for raw in (
            "C:\\Users\\secret-user\\.cache\\models\\my-reranker",
            "/home/secret-user/.cache/models/my-reranker",
            "file:///home/secret-user/.cache/models/my-reranker",
            "~/.cache/models/my-reranker",
        ):
            out = safe_reranker_model_display_name(raw)
            assert out == LOCAL_MODEL_DISPLAY
            assert "secret-user" not in out
            assert "C:" not in out
            assert ".cache" not in out


# ---------------------------------------------------------------------------
# Enable-flag parsing (Fix 3, round 2)
# ---------------------------------------------------------------------------


class TestRequestedRerankerEnabled:
    def test_true(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_RERANKER_ENABLED", "true")
        assert requested_reranker_enabled() is True

    def test_false(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_RERANKER_ENABLED", "false")
        assert requested_reranker_enabled() is False

    def test_default_false_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("RAG_RERANKER_ENABLED", raising=False)
        assert requested_reranker_enabled() is False

    def test_unparseable_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_RERANKER_ENABLED", "maybe")
        assert requested_reranker_enabled() is None
