"""Offline API tests for document upload, listing, and deletion."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.exceptions import GenerationConfigurationError

from pdf_helpers import make_text_pdf


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.model_name = "fake/test-model"

    def _vector_for(self, text: str) -> list[float]:
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
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
