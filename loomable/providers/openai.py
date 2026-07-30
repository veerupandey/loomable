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

from loomable.kernel.models import ModelRequest, ModelResponse, StreamEvent

from ._common import (
    _classify_http_error,
    parse_openai_response,
    parse_openai_sse_line,
    to_openai_messages,
)

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

    async def stream(self, request: ModelRequest):
        """Stream a chat completion, yielding StreamEvents as they arrive.

        Yields text deltas, assembled tool calls, and a terminal end event.
        Falls back to the same error classification as complete().
        """
        from collections.abc import AsyncIterator

        url = f"{self._base_url}/chat/completions"
        body = self._build_body(request)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", url, json=body, headers=self._headers()
                ) as resp:
                    resp.raise_for_status()
                    async for event in self._parse_sse_stream(resp):
                        yield event
        except httpx.HTTPError as exc:
            raise _classify_http_error(self._provider_id, exc) from exc

    async def _parse_sse_stream(self, resp) -> "AsyncIterator[StreamEvent]":
        """Parse SSE lines from an httpx streaming response into StreamEvents."""
        import json as _json

        from loomable.kernel.models import ToolCall

        # Accumulate tool call fragments: {index: {id, name, args_str}}
        tool_call_acc: dict[int, dict[str, str]] = {}

        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = _json.loads(payload)
            except (_json.JSONDecodeError, TypeError):
                continue

            choices = chunk.get("choices", [])
            if not choices:
                usage = chunk.get("usage")
                if usage:
                    yield StreamEvent(
                        kind="end",
                        usage={
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                    )
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            content = delta.get("content")
            if content:
                yield StreamEvent(kind="text", text=content)

            tc_list = delta.get("tool_calls")
            if tc_list:
                for tc in tc_list:
                    idx = tc.get("index", 0)
                    if idx not in tool_call_acc:
                        tool_call_acc[idx] = {"id": tc.get("id", ""), "name": "", "args": ""}
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_call_acc[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_call_acc[idx]["args"] += fn["arguments"]

            if finish_reason and tool_call_acc:
                for _idx, acc in sorted(tool_call_acc.items()):
                    try:
                        args = _json.loads(acc["args"]) if acc["args"] else {}
                    except (_json.JSONDecodeError, TypeError):
                        args = {"_raw": acc["args"]}
                    yield StreamEvent(
                        kind="tool_call",
                        tool_call=ToolCall(id=acc["id"], tool_name=acc["name"], args=args),
                    )
                tool_call_acc.clear()

            usage = chunk.get("usage")
            if usage and finish_reason:
                yield StreamEvent(
                    kind="end",
                    usage={
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                )

        # Ensure terminal event
        yield StreamEvent(kind="end")


class AzureOpenAIProvider:
    """A ``ModelProvider`` for Azure OpenAI chat deployments.

    Parameters
    ----------
    deployment:
        The Azure deployment name (used in the request URL).
        Defaults to the ``AZURE_OPENAI_DEPLOYMENT_NAME`` environment variable.
    endpoint:
        The Azure resource endpoint (e.g. ``https://my-resource.openai.azure.com``).
        Defaults to the ``AZURE_OPENAI_ENDPOINT`` environment variable.
    api_key:
        The Azure API key (sent as the ``api-key`` header). Defaults to the
        ``AZURE_OPENAI_API_KEY`` environment variable.
    api_version:
        The Azure API version query parameter. Defaults to the
        ``AZURE_OPENAI_API_VERSION`` environment variable, or ``2024-08-01-preview``.
    default_headers:
        Optional extra headers merged into every request.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        deployment: str | None = None,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.deployment = (
            deployment if deployment is not None
            else os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        )
        if not self.deployment:
            raise ValueError(
                "Azure deployment is required: pass deployment=... or set AZURE_OPENAI_DEPLOYMENT_NAME."
            )
        resolved_endpoint = endpoint if endpoint is not None else os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not resolved_endpoint:
            raise ValueError(
                "Azure endpoint is required: pass endpoint=... or set AZURE_OPENAI_ENDPOINT."
            )
        self._endpoint = resolved_endpoint.rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("AZURE_OPENAI_API_KEY")
        self._api_version = (
            api_version if api_version is not None
            else os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        )
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

    async def stream(self, request: ModelRequest):
        """Stream a chat completion from Azure, yielding StreamEvents.

        Uses the same SSE wire format as the OpenAI Chat Completions API.
        """
        url = self._url()
        body = self._build_body(request)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", url, json=body, headers=self._headers()
                ) as resp:
                    resp.raise_for_status()
                    async for event in self._parse_sse_stream(resp):
                        yield event
        except httpx.HTTPError as exc:
            raise _classify_http_error(self._provider_id, exc) from exc

    async def _parse_sse_stream(self, resp):
        """Parse Azure SSE stream (same wire format as OpenAI)."""
        import json as _json

        from loomable.kernel.models import ToolCall as _ToolCall

        tool_call_acc: dict[int, dict[str, str]] = {}

        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = _json.loads(payload)
            except (_json.JSONDecodeError, TypeError):
                continue

            choices = chunk.get("choices", [])
            if not choices:
                usage = chunk.get("usage")
                if usage:
                    yield StreamEvent(
                        kind="end",
                        usage={
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                    )
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            content = delta.get("content")
            if content:
                yield StreamEvent(kind="text", text=content)

            tc_list = delta.get("tool_calls")
            if tc_list:
                for tc in tc_list:
                    idx = tc.get("index", 0)
                    if idx not in tool_call_acc:
                        tool_call_acc[idx] = {"id": tc.get("id", ""), "name": "", "args": ""}
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_call_acc[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_call_acc[idx]["args"] += fn["arguments"]

            if finish_reason and tool_call_acc:
                for _idx, acc in sorted(tool_call_acc.items()):
                    try:
                        args = _json.loads(acc["args"]) if acc["args"] else {}
                    except (_json.JSONDecodeError, TypeError):
                        args = {"_raw": acc["args"]}
                    yield StreamEvent(
                        kind="tool_call",
                        tool_call=_ToolCall(id=acc["id"], tool_name=acc["name"], args=args),
                    )
                tool_call_acc.clear()

            usage = chunk.get("usage")
            if usage and finish_reason:
                yield StreamEvent(
                    kind="end",
                    usage={
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                )

        yield StreamEvent(kind="end")
