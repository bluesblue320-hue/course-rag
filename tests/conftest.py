"""Keep the test suite from importing the real embedding model runtime."""

import sys
from types import ModuleType


class OfflineSentenceTransformer:
    """Fail fast when a test forgets to inject its fake embedding model."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("测试必须注入 Fake SentenceTransformer")


class OfflineCrossEncoder:
    """Fail fast when a test forgets to inject its fake CrossEncoder."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("测试必须注入 Fake CrossEncoder")


sentence_transformers = ModuleType("sentence_transformers")
setattr(
    sentence_transformers,
    "SentenceTransformer",
    OfflineSentenceTransformer,
)
setattr(
    sentence_transformers,
    "CrossEncoder",
    OfflineCrossEncoder,
)
sys.modules["sentence_transformers"] = sentence_transformers
