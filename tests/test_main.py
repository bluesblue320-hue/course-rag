from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import src.main as main_module


class FakeEmbeddingService:
    query_count = 0

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))

    def encode_query(self, query: str) -> np.ndarray:
        type(self).query_count += 1
        return np.array([1.0, 0.0])


def _set_inputs(
    monkeypatch: pytest.MonkeyPatch,
    values: list[str],
) -> None:
    inputs: Iterator[str] = iter(values)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))


def test_main_loads_chunks_searches_and_formats_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("service 层负责业务逻辑。", encoding="utf-8")
    monkeypatch.setattr(main_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(main_module, "EmbeddingService", FakeEmbeddingService)
    _set_inputs(monkeypatch, ["业务逻辑在哪里？", "exit"])

    main_module.main()

    output = capsys.readouterr().out
    assert "成功加载 1 个 Chunk" in output
    assert "Top 1" in output
    assert "相似度：1.0000" in output
    assert "Chunk编号：0" in output
    assert "原文：service 层负责业务逻辑。" in output
    assert "程序已退出。" in output


@pytest.mark.parametrize("exit_value", ["", "exit", "QUIT"])
def test_main_exits_without_encoding_a_query(
    exit_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("repository 层访问数据库。", encoding="utf-8")
    monkeypatch.setattr(main_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(main_module, "EmbeddingService", FakeEmbeddingService)
    FakeEmbeddingService.query_count = 0
    _set_inputs(monkeypatch, [exit_value])

    main_module.main()

    assert FakeEmbeddingService.query_count == 0
