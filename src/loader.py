"""Load plain-text knowledge files."""

from pathlib import Path


def load_text(file_path: str) -> str:
    """Read a non-empty UTF-8 text file and trim outer whitespace."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"文本文件不存在：{path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"文本文件为空：{path}")

    return text
