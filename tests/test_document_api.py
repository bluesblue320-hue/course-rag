"""Offline API tests for document upload, listing, and deletion."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.exceptions import (
    DocumentMetadataError,
    GenerationConfigurationError,
    UploadTooLargeError,
)

from pdf_helpers import make_text_pdf


class FakeEmbeddingService:
    fail_documents = False

    def __init__(self) -> None:
        self.model_name = "fake/test-model"

    def _vector_for(self, text: str) -> list[float]:
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if type(self).fail_documents:
            raise RuntimeError("embedding exploded")
        return np.asarray([self._vector_for(text) for text in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self._vector_for(query))


class FakeGenerationService:
    def __init__(self) -> None:
        self.model_name = "fake/llm-model"

    def generate(self, _prompt: str) -> str:
        return "这是基于上传资料的回答。[来源1]"

    def close(self) -> None:
        return None


class MisconfiguredGenerationService:
    def __init__(self) -> None:
        raise GenerationConfigurationError("缺少 LLM API Key 配置")

    def generate(self, _prompt: str) -> str:
        return "不应生成答案"

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def clear_document_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_MIN_RELEVANCE_SCORE", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_BYTES", raising=False)


@pytest.fixture(autouse=True)
def reset_fake_embedding_fail_flag() -> Iterator[None]:
    FakeEmbeddingService.fail_documents = False
    yield
    FakeEmbeddingService.fail_documents = False


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        FakeGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def _upload(
    client: TestClient,
    filename: str,
    content: bytes,
    content_type: str = "text/plain",
):
    return client.post(
        "/documents",
        files={"file": (filename, content, content_type)},
    )


def test_upload_txt_returns_created_document(client: TestClient) -> None:
    response = _upload(
        client,
        "notes.txt",
        "alpha 上传的课程内容".encode("utf-8"),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["content_type"] == "text/plain"
    assert body["size_bytes"] == len("alpha 上传的课程内容".encode("utf-8"))
    assert body["text_length"] > 0
    assert body["chunk_count"] >= 1
    assert body["is_builtin"] is False
    assert body["index_status"] == "ready"
    assert body["document_id"]


def test_get_documents_lists_builtin_first_and_totals(
    client: TestClient,
) -> None:
    _upload(client, "notes.txt", "alpha 内容".encode("utf-8"))

    response = client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == 2
    assert body["chunk_count"] >= 5
    assert body["documents"][0]["filename"] == "knowledge.txt"
    assert body["documents"][0]["is_builtin"] is True
    assert body["documents"][1]["filename"] == "notes.txt"
    assert body["documents"][1]["is_builtin"] is False


def test_delete_uploaded_document_removes_content(
    client: TestClient,
) -> None:
    created = _upload(client, "notes.txt", "alpha 内容".encode("utf-8"))
    document_id = created.json()["document_id"]

    delete_response = client.delete(f"/documents/{document_id}")

    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "document_id": document_id,
        "deleted": True,
        "document_count": 1,
        "chunk_count": 4,
    }

    search_response = client.post(
        "/search",
        json={"query": "alpha"},
    )
    assert all(
        "alpha" not in result["text"]
        for result in search_response.json()["results"]
    )


def test_uploaded_content_is_searchable(client: TestClient) -> None:
    _upload(client, "notes.txt", "alpha 上传的课程内容".encode("utf-8"))

    response = client.post("/search", json={"query": "alpha"})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["filename"] == "notes.txt"
    assert results[0]["page_number"] is None
    assert results[0]["document_id"]
    assert "alpha" in results[0]["text"]


def test_pdf_upload_is_searchable_with_page_number(client: TestClient) -> None:
    _upload(
        client,
        "course.pdf",
        make_text_pdf(["alpha page one", "alpha page two"]),
        "application/pdf",
    )

    response = client.post("/search", json={"query": "alpha", "top_k": 2})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [result["filename"] for result in results] == [
        "course.pdf",
        "course.pdf",
    ]
    assert [result["page_number"] for result in results] == [1, 2]


def test_uploaded_content_is_usable_in_ask(client: TestClient) -> None:
    _upload(client, "notes.txt", "alpha 上传的课程内容".encode("utf-8"))

    response = client.post("/ask", json={"question": "alpha 讲了什么？"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer_status"] == "answered"
    assert body["sources"][0]["filename"] == "notes.txt"


def test_upload_too_large_returns_413(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10")
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        FakeGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as test_client:
        response = _upload(test_client, "big.txt", b"x" * 20)

    assert response.status_code == 413
    assert response.json() == {
        "code": "UPLOAD_TOO_LARGE",
        "message": "文件超过上传大小限制",
    }


def test_unsupported_extension_returns_415(client: TestClient) -> None:
    response = _upload(
        client,
        "notes.docx",
        b"word content",
        "application/vnd.openxmlformats-officedocument",
    )

    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_DOCUMENT_TYPE"


def test_unsupported_content_type_returns_415(client: TestClient) -> None:
    response = _upload(client, "notes.txt", b"content", "image/png")

    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_DOCUMENT_TYPE"


def test_empty_file_returns_422(client: TestClient) -> None:
    response = _upload(client, "empty.txt", b"")

    assert response.status_code == 422
    assert response.json() == {
        "code": "EMPTY_DOCUMENT",
        "message": "文档内容为空",
    }


def test_broken_pdf_returns_422_without_path_leak(client: TestClient) -> None:
    response = _upload(
        client,
        "broken.pdf",
        b"not a pdf at all",
        "application/pdf",
    )

    assert response.status_code == 422
    assert response.json()["code"] == "DOCUMENT_PARSE_FAILED"
    assert "tmp" not in response.text
    assert "uploads" not in response.text


def test_delete_unknown_document_returns_404(client: TestClient) -> None:
    response = client.delete("/documents/not-a-real-id")

    assert response.status_code == 404
    assert response.json()["code"] == "DOCUMENT_NOT_FOUND"


def test_delete_builtin_document_returns_409(client: TestClient) -> None:
    response = client.delete("/documents/builtin-knowledge")

    assert response.status_code == 409
    assert response.json()["code"] == "BUILTIN_DOCUMENT_CANNOT_BE_DELETED"


def test_upload_works_without_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        MisconfiguredGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as test_client:
        upload_response = _upload(
            test_client,
            "notes.txt",
            "alpha 内容".encode("utf-8"),
        )
        search_response = test_client.post(
            "/search",
            json={"query": "alpha"},
        )

    assert upload_response.status_code == 201
    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["filename"] == "notes.txt"


def test_upload_works_with_invalid_rag_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_MIN_RELEVANCE_SCORE", "not-a-number")
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        FakeGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as test_client:
        upload_response = _upload(
            test_client,
            "notes.txt",
            "alpha 内容".encode("utf-8"),
        )
        search_response = test_client.post(
            "/search",
            json={"query": "alpha"},
        )

    assert upload_response.status_code == 201
    assert search_response.status_code == 200


def test_upload_configuration_error_disables_upload_but_keeps_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "not-an-integer")
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        FakeGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as test_client:
        upload_response = _upload(
            test_client,
            "notes.txt",
            "alpha 内容".encode("utf-8"),
        )
        list_response = test_client.get("/documents")
        search_response = test_client.post(
            "/search",
            json={"query": "alpha"},
        )

    assert upload_response.status_code == 503
    assert upload_response.json() == {
        "code": "UPLOAD_NOT_CONFIGURED",
        "message": "文档上传配置无效",
    }
    assert list_response.status_code == 200
    assert search_response.status_code == 200


def test_upload_survives_restart_via_persisted_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api_module,
        "METADATA_PATH",
        tmp_path / "documents.json",
    )
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        FakeGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with TestClient(api_module.app) as first_client:
        upload_response = _upload(
            first_client,
            "notes.txt",
            "alpha 内容".encode("utf-8"),
        )
        assert upload_response.status_code == 201

    with TestClient(api_module.app) as second_client:
        list_response = second_client.get("/documents")
        search_response = second_client.post(
            "/search",
            json={"query": "alpha"},
        )

    assert list_response.status_code == 200
    body = list_response.json()
    assert body["document_count"] == 2
    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["filename"] == "notes.txt"


class _RecordingFile:
    """Fake file that records every read() size and forbids unbounded reads."""

    def __init__(self, data: bytes, calls: list[int | None]) -> None:
        self._data = data
        self._calls = calls
        self.position = 0

    def read(self, size: int | None = -1) -> bytes:
        self._calls.append(size)
        if size is None or size < 0:
            raise AssertionError("不允许无界读取上传文件")
        chunk = self._data[self.position : self.position + size]
        self.position += len(chunk)
        return chunk


class _RecordingUpload:
    def __init__(self, data: bytes, calls: list[int | None]) -> None:
        self.file = _RecordingFile(data, calls)


def test_read_upload_with_limit_reads_exactly_max_plus_one() -> None:
    calls: list[int | None] = []
    upload = _RecordingUpload(b"x" * 100, calls)

    with pytest.raises(UploadTooLargeError):
        api_module.read_upload_with_limit(upload, 10)

    assert calls == [11]
    assert upload.file.position == 11


def test_read_upload_with_limit_never_uses_unbounded_read() -> None:
    calls: list[int | None] = []
    upload = _RecordingUpload(b"x" * 5, calls)

    api_module.read_upload_with_limit(upload, 10)

    assert calls == [11]
    assert upload.file.position == 5


def test_read_upload_with_limit_accepts_file_below_limit() -> None:
    data = api_module.read_upload_with_limit(_RecordingUpload(b"x" * 5, []), 10)

    assert data == b"x" * 5


def test_read_upload_with_limit_accepts_file_equal_to_limit() -> None:
    data = api_module.read_upload_with_limit(_RecordingUpload(b"x" * 10, []), 10)

    assert len(data) == 10


def test_read_upload_with_limit_rejects_one_byte_over() -> None:
    with pytest.raises(UploadTooLargeError):
        api_module.read_upload_with_limit(_RecordingUpload(b"x" * 11, []), 10)


def test_read_upload_with_limit_does_not_copy_a_huge_file_into_memory() -> None:
    calls: list[int | None] = []
    upload = _RecordingUpload(b"x" * 1_000_000, calls)

    with pytest.raises(UploadTooLargeError):
        api_module.read_upload_with_limit(upload, 10)

    assert upload.file.position == 11
    assert calls == [11]


def _stored_name(seed: str) -> str:
    hex_slug = "".join(
        character for character in seed if character in "0123456789abcdef"
    )
    return f"{hex_slug}{'a' * (32 - len(hex_slug))}.txt"


def _record_dict(
    document_id: str,
    stored_filename: str | None,
    *,
    is_builtin: bool = False,
    content_type: str = "text/plain",
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "original_filename": f"{document_id}.txt",
        "stored_filename": stored_filename,
        "content_type": content_type,
        "size_bytes": 10,
        "text_length": 10,
        "chunk_count": 1,
        "created_at": "2026-08-03T08:00:00Z",
        "is_builtin": is_builtin,
    }


def _prepare_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_records: list[dict[str, object]],
    upload_files: dict[str, bytes] | None = None,
) -> Path:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (upload_files or {}).items():
        (upload_dir / filename).write_bytes(content)
    metadata_path = tmp_path / "documents.json"
    metadata_path.write_text(
        json.dumps({"documents": metadata_records}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api_module, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        FakeGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )
    return metadata_path


def test_startup_reconciles_missing_persisted_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_name = _stored_name("missing")
    metadata_path = _prepare_startup(
        tmp_path,
        monkeypatch,
        [_record_dict("doc-missing", missing_name)],
    )

    with TestClient(api_module.app) as test_client:
        list_response = test_client.get("/documents")
        search_response = test_client.post(
            "/search",
            json={"query": "A"},
        )

    assert list_response.status_code == 200
    body = list_response.json()
    assert body["document_count"] == 1
    assert body["documents"][0]["is_builtin"] is True
    assert search_response.status_code == 200
    assert all(
        result["document_id"] != "doc-missing"
        for result in search_response.json()["results"]
    )
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert all(
        record["document_id"] != "doc-missing"
        for record in persisted["documents"]
    )


def test_startup_reconciles_corrupt_persisted_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corrupt_name = _stored_name("corrupt")[:-4] + ".pdf"
    metadata_path = _prepare_startup(
        tmp_path,
        monkeypatch,
        [
            _record_dict(
                "doc-corrupt",
                corrupt_name,
                content_type="application/pdf",
            )
        ],
        upload_files={corrupt_name: b"not a real pdf"},
    )

    with TestClient(api_module.app) as test_client:
        list_response = test_client.get("/documents")

    assert list_response.status_code == 200
    body = list_response.json()
    assert body["document_count"] == 1
    assert all(
        document["document_id"] != "doc-corrupt"
        for document in body["documents"]
    )
    assert not (tmp_path / "uploads" / corrupt_name).exists()
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert persisted["documents"] == []


def test_startup_reconciles_empty_persisted_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_name = _stored_name("empty")
    _prepare_startup(
        tmp_path,
        monkeypatch,
        [_record_dict("doc-empty", empty_name)],
        upload_files={empty_name: b""},
    )

    with TestClient(api_module.app) as test_client:
        list_response = test_client.get("/documents")

    assert list_response.status_code == 200
    assert all(
        document["document_id"] != "doc-empty"
        for document in list_response.json()["documents"]
    )


def test_startup_keeps_valid_persisted_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_name = _stored_name("valid")
    _prepare_startup(
        tmp_path,
        monkeypatch,
        [_record_dict("doc-valid", valid_name)],
        upload_files={valid_name: "alpha 持久化内容".encode("utf-8")},
    )

    with TestClient(api_module.app) as test_client:
        list_response = test_client.get("/documents")
        search_response = test_client.post(
            "/search",
            json={"query": "alpha"},
        )

    assert list_response.status_code == 200
    assert list_response.json()["document_count"] == 2
    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["filename"] == "doc-valid.txt"


def test_startup_single_invalid_document_does_not_break_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_name = _stored_name("missing")
    valid_name = _stored_name("valid")
    _prepare_startup(
        tmp_path,
        monkeypatch,
        [
            _record_dict("doc-missing", missing_name),
            _record_dict("doc-valid", valid_name),
        ],
        upload_files={valid_name: "alpha 持久化内容".encode("utf-8")},
    )

    with TestClient(api_module.app) as test_client:
        list_response = test_client.get("/documents")
        search_response = test_client.post(
            "/search",
            json={"query": "alpha"},
        )
        builtin_search = test_client.post("/search", json={"query": "课程"})

    assert list_response.status_code == 200
    assert list_response.json()["document_count"] == 2
    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["filename"] == "doc-valid.txt"
    assert builtin_search.status_code == 200


def test_startup_corrupted_json_still_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_file = tmp_path / "knowledge.txt"
    knowledge_file.write_text("A" * 900, encoding="utf-8")
    monkeypatch.setattr(api_module, "KNOWLEDGE_PATH", knowledge_file)
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path / "uploads")
    metadata_path = tmp_path / "documents.json"
    metadata_path.write_text("{ not valid json", encoding="utf-8")
    monkeypatch.setattr(api_module, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(
        api_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        api_module,
        "GenerationService",
        FakeGenerationService,
    )
    monkeypatch.setattr(
        api_module,
        "load_dotenv",
        lambda *, dotenv_path, override: None,
    )

    with pytest.raises(DocumentMetadataError):
        with TestClient(api_module.app):
            pass


def test_startup_rejects_traversal_metadata_and_never_touches_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = tmp_path / "escape.txt"
    sentinel.write_text("sentinel-content", encoding="utf-8")
    _prepare_startup(
        tmp_path,
        monkeypatch,
        [_record_dict("doc-evil", "../../escape.txt")],
    )

    with pytest.raises(DocumentMetadataError):
        with TestClient(api_module.app):
            pass

    assert sentinel.read_text(encoding="utf-8") == "sentinel-content"


STABLE_500 = {
    "code": "DOCUMENT_INGESTION_FAILED",
    "message": "文档处理失败，请稍后重试",
}


def test_upload_embedding_failure_returns_stable_500_and_rolls_back(
    client: TestClient,
    tmp_path: Path,
) -> None:
    first_upload = _upload(client, "notes.txt", "alpha 内容".encode("utf-8"))
    assert first_upload.status_code == 201
    first_stored_name = list((tmp_path / "uploads").iterdir())[0].name

    FakeEmbeddingService.fail_documents = True
    try:
        response = _upload(client, "other.txt", "beta 内容".encode("utf-8"))
    finally:
        FakeEmbeddingService.fail_documents = False

    assert response.status_code == 500
    assert response.json() == STABLE_500
    assert "exploded" not in response.text
    assert "uploads" not in response.text
    remaining_files = list((tmp_path / "uploads").iterdir())
    assert len(remaining_files) == 1
    assert remaining_files[0].name == first_stored_name
    listing = client.get("/documents").json()
    assert listing["document_count"] == 2
    search = client.post("/search", json={"query": "alpha"}).json()
    assert search["results"][0]["filename"] == "notes.txt"


def test_upload_index_failure_returns_stable_500_and_rolls_back(
    client: TestClient,
) -> None:
    index = client.app.state.knowledge_index
    original_replace = index.replace

    def failing_replace(chunks: object, embeddings: object) -> None:
        raise RuntimeError("index exploded")

    index.replace = failing_replace  # type: ignore[method-assign]
    try:
        response = _upload(client, "notes.txt", "alpha 内容".encode("utf-8"))
    finally:
        index.replace = original_replace

    assert response.status_code == 500
    assert response.json() == STABLE_500
    assert "index exploded" not in response.text
    listing = client.get("/documents").json()
    assert listing["document_count"] == 1
    search = client.post("/search", json={"query": "课程"}).json()
    assert search["results"]


def test_delete_rebuild_failure_returns_stable_500_and_keeps_state(
    client: TestClient,
    tmp_path: Path,
) -> None:
    created = _upload(client, "notes.txt", "alpha 内容".encode("utf-8"))
    document_id = created.json()["document_id"]
    stored_file = list((tmp_path / "uploads").iterdir())[0]
    assert stored_file.is_file()

    FakeEmbeddingService.fail_documents = True
    try:
        response = client.delete(f"/documents/{document_id}")
    finally:
        FakeEmbeddingService.fail_documents = False

    assert response.status_code == 500
    assert response.json() == STABLE_500
    assert "exploded" not in response.text
    listing = client.get("/documents").json()
    assert listing["document_count"] == 2
    assert any(
        document["document_id"] == document_id
        for document in listing["documents"]
    )
    assert stored_file.exists()
    search = client.post("/search", json={"query": "alpha"}).json()
    assert search["results"][0]["filename"] == "notes.txt"


def test_broken_pdf_upload_stays_422_after_failure_mapping(
    client: TestClient,
) -> None:
    response = _upload(
        client,
        "broken.pdf",
        b"not a pdf at all",
        "application/pdf",
    )

    assert response.status_code == 422
    assert response.json()["code"] == "DOCUMENT_PARSE_FAILED"
