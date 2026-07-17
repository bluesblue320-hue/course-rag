from pathlib import Path

import pytest

from src.loader import load_text


def test_load_text_reads_utf8_and_strips_outer_whitespace(tmp_path: Path) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("  业务逻辑属于 service 层。\n", encoding="utf-8")

    result = load_text(str(knowledge_file))

    assert result == "业务逻辑属于 service 层。"


def test_load_text_raises_when_file_does_not_exist(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="文本文件不存在"):
        load_text(str(missing_file))


def test_load_text_raises_when_file_is_empty(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match="文本文件为空"):
        load_text(str(empty_file))
