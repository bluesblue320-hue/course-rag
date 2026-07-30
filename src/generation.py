"""Generate answers through an OpenAI-compatible Chat Completions API."""

import math
import os
from typing import Any

import httpx

from src.exceptions import GenerationConfigurationError, GenerationError

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0


class GenerationService:
    """Call a configured LLM without exposing provider errors to callers."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
        client: Any | None = None,
    ) -> None:
        resolved_api_key = self._resolve_text(api_key, "LLM_API_KEY")
        if not resolved_api_key:
            raise GenerationConfigurationError("缺少 LLM API Key 配置")

        resolved_model_name = self._resolve_text(model_name, "LLM_MODEL")
        if not resolved_model_name:
            raise GenerationConfigurationError("缺少 LLM 模型名称配置")

        resolved_base_url = self._resolve_text(base_url, "LLM_BASE_URL")
        self._api_key = resolved_api_key
        self.base_url = (resolved_base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model_name = resolved_model_name
        self.timeout_seconds = self._resolve_timeout(timeout_seconds)
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client()
        self._closed = False

    @staticmethod
    def _resolve_text(explicit_value: str | None, environment_name: str) -> str:
        value = (
            explicit_value
            if explicit_value is not None
            else os.getenv(environment_name, "")
        )
        return value.strip()

    @staticmethod
    def _resolve_timeout(explicit_value: float | None) -> float:
        raw_value: float | str = (
            explicit_value
            if explicit_value is not None
            else os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        )
        try:
            timeout = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise GenerationConfigurationError(
                "LLM timeout 必须是有效数字"
            ) from exc

        if not math.isfinite(timeout) or timeout <= 0:
            raise GenerationConfigurationError("LLM timeout 必须大于 0")
        return timeout

    def generate(self, prompt: str) -> str:
        """Return one stripped answer for a non-empty prompt."""
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("prompt 不能为空")

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": cleaned_prompt}],
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise GenerationError("LLM 请求超时") from exc
        except httpx.HTTPError as exc:
            raise GenerationError("LLM 服务调用失败") from exc

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GenerationError("LLM 返回格式无效") from exc

        if not isinstance(content, str):
            raise GenerationError("LLM 返回格式无效")

        answer = content.strip()
        if not answer:
            raise GenerationError("LLM 服务返回了空答案")
        return answer

    def close(self) -> None:
        """Close the internally owned HTTP client at most once."""
        if self._closed:
            return

        self._closed = True
        if not self._owns_client:
            return

        close = getattr(self._client, "close", None)
        if callable(close):
            close()
