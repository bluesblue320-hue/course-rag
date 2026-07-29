from typing import Any

import pytest

from src.exceptions import GenerationConfigurationError, GenerationError
from src.generation import DEFAULT_BASE_URL, GenerationService


class FakeResponse:
    def __init__(
        self,
        content: object = "生成的答案",
        error: Exception | None = None,
    ) -> None:
        self._content = content
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": self._content,
                    }
                }
            ]
        }


class FakeClient:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response or FakeResponse()
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self._error is not None:
            raise self._error
        return self._response


@pytest.fixture(autouse=True)
def clear_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable_name in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(variable_name, raising=False)


def make_service(
    client: FakeClient,
    **overrides: object,
) -> GenerationService:
    options: dict[str, object] = {
        "api_key": "test-secret-key",
        "model_name": "test-model",
        "client": client,
    }
    options.update(overrides)
    return GenerationService(**options)


def test_generate_returns_answer() -> None:
    service = make_service(FakeClient(FakeResponse("课程答案")))

    assert service.generate("课程问题") == "课程答案"


def test_generate_passes_cleaned_prompt_to_client() -> None:
    client = FakeClient()
    service = make_service(client)

    service.generate("  课程问题  ")

    assert client.calls[0]["json"]["messages"] == [
        {"role": "user", "content": "课程问题"}
    ]


def test_generate_passes_model_name_to_client() -> None:
    client = FakeClient()
    service = make_service(client, model_name="course-model")

    service.generate("课程问题")

    assert client.calls[0]["json"]["model"] == "course-model"
    assert service.model_name == "course-model"


def test_generate_strips_answer() -> None:
    service = make_service(FakeClient(FakeResponse("  精简答案\n")))

    assert service.generate("课程问题") == "精简答案"


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_generate_rejects_empty_prompt(prompt: str) -> None:
    service = make_service(FakeClient())

    with pytest.raises(ValueError, match="prompt 不能为空"):
        service.generate(prompt)


def test_missing_api_key_raises_configuration_error() -> None:
    with pytest.raises(GenerationConfigurationError, match="API Key"):
        GenerationService(model_name="test-model", client=FakeClient())


def test_missing_model_name_raises_configuration_error() -> None:
    with pytest.raises(GenerationConfigurationError, match="模型名称"):
        GenerationService(api_key="test-key", client=FakeClient())


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_non_positive_or_non_finite_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(GenerationConfigurationError, match="timeout 必须大于 0"):
        make_service(FakeClient(), timeout_seconds=timeout)


def test_upstream_failure_is_converted_to_generation_error() -> None:
    upstream_error = RuntimeError("provider failed")
    service = make_service(FakeClient(error=upstream_error))

    with pytest.raises(GenerationError, match="LLM 服务调用失败") as caught:
        service.generate("课程问题")

    assert caught.value.__cause__ is upstream_error


def test_timeout_is_converted_to_generation_error() -> None:
    timeout_error = TimeoutError("request timed out")
    service = make_service(FakeClient(error=timeout_error))

    with pytest.raises(GenerationError, match="LLM 服务调用失败") as caught:
        service.generate("课程问题")

    assert caught.value.__cause__ is timeout_error


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_answer_raises_generation_error(content: object) -> None:
    service = make_service(FakeClient(FakeResponse(content)))

    with pytest.raises(GenerationError, match="LLM 服务返回了空答案"):
        service.generate("课程问题")


def test_error_message_does_not_expose_api_key() -> None:
    api_key = "highly-sensitive-api-key"
    client = FakeClient(error=RuntimeError(f"provider rejected {api_key}"))
    service = make_service(client, api_key=api_key)

    with pytest.raises(GenerationError) as caught:
        service.generate("课程问题")

    assert api_key not in str(caught.value)


def test_request_uses_configured_url_timeout_and_authorization() -> None:
    client = FakeClient()
    service = make_service(
        client,
        base_url="https://llm.example/v1/",
        timeout_seconds=12.5,
    )

    service.generate("课程问题")

    assert client.calls[0]["url"] == "https://llm.example/v1/chat/completions"
    assert client.calls[0]["timeout"] == 12.5
    assert client.calls[0]["headers"]["Authorization"] == (
        "Bearer test-secret-key"
    )


def test_environment_configuration_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "environment-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://environment.example/v1")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "8.5")
    client = FakeClient()
    service = GenerationService(client=client)

    service.generate("课程问题")

    assert service.model_name == "environment-model"
    assert service.timeout_seconds == 8.5
    assert client.calls[0]["url"].startswith("https://environment.example/v1/")


def test_explicit_configuration_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "environment-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://environment.example/v1")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "8.5")
    client = FakeClient()
    service = GenerationService(
        api_key="explicit-key",
        base_url="https://explicit.example/v1",
        model_name="explicit-model",
        timeout_seconds=4,
        client=client,
    )

    service.generate("课程问题")

    assert service.model_name == "explicit-model"
    assert service.timeout_seconds == 4
    assert client.calls[0]["url"].startswith("https://explicit.example/v1/")
    assert client.calls[0]["headers"]["Authorization"] == "Bearer explicit-key"


def test_default_base_url_and_timeout_are_used() -> None:
    client = FakeClient()
    service = make_service(client)

    service.generate("课程问题")

    assert service.base_url == DEFAULT_BASE_URL
    assert service.timeout_seconds == 30.0
