"""loomable.providers.openai - OpenAI and OpenAI-compatible model providers.

:class:`OpenAIProvider` targets the OpenAI Chat Completions API and any
OpenAI-compatible server (vLLM, Together, Groq, Ollama, LM Studio, ...) by pointing
``base_url`` at the compatible endpoint. :class:`AzureOpenAIProvider` targets Azure
OpenAI (deployment + api-version, ``api-key`` header). Both implement the kernel
``ModelProvider`` protocol and reuse the shared OpenAI translation helpers.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from loomable.kernel.models import ModelRequest, ModelResponse

from ._common import _classify_http_error, parse_openai_response, to_openai_messages

_DEFAULT_TIMEOUT = 60.0


class OpenAIProvider:
    """A ``ModelProvider`` for OpenAI and OpenAI-compatible chat endpoints.

    Parameters
    ----------
    model:
        The model name (e.g. ``"gpt-4o-mini"``) sent in the request body.
    api_key:
        The bearer API key. Defaults to the ``OPENAI_API_KEY`` environment variable.
        May be ``None`` for local servers that do not require auth.
    base_url:
        The API base URL. Defaults to ``https://api.openai.com/v1``. Point this at any
        OpenAI-compatible server (e.g. ``http://localhost:11434/v1`` for Ollama).
    organization:
        Optional OpenAI organization id (sent as ``OpenAI-Organization``).
    default_headers:
        Optional extra headers merged into every request.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        organization: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._organization = organization
        self._default_headers = dict(default_headers or {})
        self._timeout = timeout

    @property
    def _provider_id(self) -> str:
        return f"openai:{self.model}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._default_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._organization:
            headers["OpenAI-Organization"] = self._organization
        return headers

    def _build_body(self, request: ModelRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(request.messages),
            "temperature": request.temperature,
        }
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.tools:
            body["tools"] = request.tools
        return body

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send the chat-completion request and return a ``ModelResponse``.

        Raises a classified :class:`~loomable.providers.errors.TransientProviderError`
        or :class:`~loomable.providers.errors.PermanentProviderError` naming the
        provider when the endpoint is unreachable or returns a non-success status.
        """
        url = f"{self._base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=self._build_body(request), headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise _classify_http_error(self._provider_id, exc) from exc
        except ValueError as exc:
            from loomable.kernel.errors import ModelProviderError

            raise ModelProviderError(self._provider_id) from exc
        return parse_openai_response(data)


class AzureOpenAIProvider:
    """A ``ModelProvider`` for Azure OpenAI chat deployments.

    Parameters
    ----------
    deployment:
        The Azure deployment name (used in the request URL).
    endpoint:
        The Azure resource endpoint (e.g. ``https://my-resource.openai.azure.com``).
        Defaults to the ``AZURE_OPENAI_ENDPOINT`` environment variable.
    api_key:
        The Azure API key (sent as the ``api-key`` header). Defaults to the
        ``AZURE_OPENAI_API_KEY`` environment variable.
    api_version:
        The Azure API version query parameter.
    default_headers:
        Optional extra headers merged into every request.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        deployment: str,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-08-01-preview",
        default_headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.deployment = deployment
        resolved_endpoint = endpoint if endpoint is not None else os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not resolved_endpoint:
            raise ValueError(
                "Azure endpoint is required: pass endpoint=... or set AZURE_OPENAI_ENDPOINT."
            )
        self._endpoint = resolved_endpoint.rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("AZURE_OPENAI_API_KEY")
        self._api_version = api_version
        self._default_headers = dict(default_headers or {})
        self._timeout = timeout

    @property
    def _provider_id(self) -> str:
        return f"azure-openai:{self.deployment}"

    def _url(self) -> str:
        return (
            f"{self._endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self._api_version}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._default_headers}
        if self._api_key:
            headers["api-key"] = self._api_key
        return headers

    def _build_body(self, request: ModelRequest) -> dict[str, Any]:
        # Azure carries the model via the deployment in the URL, so no "model" field.
        body: dict[str, Any] = {
            "messages": to_openai_messages(request.messages),
            "temperature": request.temperature,
        }
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.tools:
            body["tools"] = request.tools
        return body

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send the chat-completion request to Azure and return a ``ModelResponse``.

        Raises :class:`~loomable.kernel.errors.ModelProviderError` naming the provider
        when the endpoint is unreachable or returns a non-success status.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url(), json=self._build_body(request), headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise _classify_http_error(self._provider_id, exc) from exc
        except ValueError as exc:
            from loomable.kernel.errors import ModelProviderError

            raise ModelProviderError(self._provider_id) from exc
        return parse_openai_response(data)
