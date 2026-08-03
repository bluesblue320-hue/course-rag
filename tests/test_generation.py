from typing import Any

import httpx
import pytest

import src.generation as generation_module
from src.exceptions import GenerationConfigurationError, GenerationError
from src.generation import DEFAULT_BASE_URL, GenerationService


class FakeResponse:
    def __init__(
        self,
        content: object = "生成的答案",
        error: Exception | None = None,
        json_data: object | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self._error = error
        self._json_error = json_error
        self._json_data = json_data or {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        }

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> object:
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


class FakeClient:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response or FakeResponse()
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.close_count = 0

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self._error is not None:
            raise self._error
        return self._response

    def close(self) -> None:
        self.close_count += 1


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


def test_request_error_is_converted_to_generation_error() -> None:
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    request_error = httpx.RequestError("provider failed", request=request)
    service = make_service(FakeClient(error=request_error))

    with pytest.raises(GenerationError, match="LLM 服务调用失败") as caught:
        service.generate("课程问题")

    assert caught.value.__cause__ is request_error


def test_timeout_is_converted_to_generation_error() -> None:
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    timeout_error = httpx.ReadTimeout("request timed out", request=request)
    service = make_service(FakeClient(error=timeout_error))

    with pytest.raises(GenerationError, match="LLM 请求超时") as caught:
        service.generate("课程问题")

    assert caught.value.__cause__ is timeout_error


def test_http_status_error_is_converted_to_generation_error() -> None:
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    response = httpx.Response(503, request=request)
    status_error = httpx.HTTPStatusError(
        "service unavailable",
        request=request,
        response=response,
    )
    service = make_service(FakeClient(FakeResponse(error=status_error)))

    with pytest.raises(GenerationError, match="LLM 服务调用失败") as caught:
        service.generate("课程问题")

    assert caught.value.__cause__ is status_error


def test_invalid_json_is_converted_to_generation_error() -> None:
    json_error = ValueError("invalid JSON")
    service = make_service(
        FakeClient(FakeResponse(json_error=json_error))
    )

    with pytest.raises(GenerationError, match="LLM 返回格式无效") as caught:
        service.generate("课程问题")

    assert caught.value.__cause__ is json_error


@pytest.mark.parametrize(
    "json_data",
    [
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"unexpected": "shape"},
    ],
    ids=["empty-choices", "missing-content", "missing-choices"],
)
def test_invalid_response_structure_raises_generation_error(
    json_data: object,
) -> None:
    service = make_service(FakeClient(FakeResponse(json_data=json_data)))

    with pytest.raises(GenerationError, match="LLM 返回格式无效"):
        service.generate("课程问题")


def test_non_string_content_raises_generation_error() -> None:
    service = make_service(FakeClient(FakeResponse(content=None)))

    with pytest.raises(GenerationError, match="LLM 返回格式无效"):
        service.generate("课程问题")


@pytest.mark.parametrize("content", ["", "   "])
def test_empty_answer_raises_generation_error(content: str) -> None:
    service = make_service(FakeClient(FakeResponse(content)))

    with pytest.raises(GenerationError, match="LLM 服务返回了空答案"):
        service.generate("课程问题")


def test_error_message_does_not_expose_api_key() -> None:
    api_key = "highly-sensitive-api-key"
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    client = FakeClient(
        error=httpx.RequestError(
            f"provider rejected {api_key}",
            request=request,
        )
    )
    service = make_service(client, api_key=api_key)

    with pytest.raises(GenerationError) as caught:
        service.generate("课程问题")

    assert api_key not in str(caught.value)


def test_unexpected_programming_error_is_not_swallowed() -> None:
    programming_error = RuntimeError("unexpected bug")
    service = make_service(FakeClient(error=programming_error))

    with pytest.raises(RuntimeError) as caught:
        service.generate("课程问题")

    assert caught.value is programming_error


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


def test_close_closes_internally_created_client_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr(generation_module.httpx, "Client", lambda: client)
    service = GenerationService(
        api_key="test-secret-key",
        model_name="test-model",
    )

    service.close()
    service.close()

    assert client.close_count == 1


def test_close_does_not_close_injected_client() -> None:
    client = FakeClient()
    service = make_service(client)

    service.close()
    service.close()

    assert client.close_count == 0


def test_close_accepts_injected_fake_without_close_method() -> None:
    class FakeClientWithoutClose:
        pass

    service = GenerationService(
        api_key="test-secret-key",
        model_name="test-model",
        client=FakeClientWithoutClose(),
    )

    service.close()
    service.close()
