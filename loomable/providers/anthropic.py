"""loomable.providers.anthropic - Anthropic (Claude) Messages API provider.

:class:`AnthropicProvider` targets the Anthropic Messages API and any
Anthropic-compatible endpoint via a configurable ``base_url``. It implements the
kernel ``ModelProvider`` protocol and reuses the shared Anthropic translation helpers
(system-prompt extraction + content-block conversion).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from loomable.kernel.errors import ModelProviderError
from loomable.kernel.models import ModelRequest, ModelResponse, StreamEvent

from ._common import parse_anthropic_response, split_anthropic_messages, _classify_http_error

_DEFAULT_TIMEOUT = 60.0
#: Anthropic requires an explicit max_tokens; used when the request does not set one.
_DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider:
    """A ``ModelProvider`` for the Anthropic Messages API (Claude models).

    Parameters
    ----------
    model:
        The Claude model name (e.g. ``"claude-3-5-sonnet-latest"``).
    api_key:
        The API key (sent as the ``x-api-key`` header). Defaults to the
        ``ANTHROPIC_API_KEY`` environment variable.
    base_url:
        The API base URL. Defaults to ``https://api.anthropic.com``. Point this at an
        Anthropic-compatible gateway when needed.
    version:
        The ``anthropic-version`` header value.
    max_tokens:
        Default ``max_tokens`` used when a request does not specify one (Anthropic
        requires this field).
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
        base_url: str = "https://api.anthropic.com",
        version: str = "2023-06-01",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        default_headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self._api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._version = version
        self._max_tokens = max_tokens
        self._default_headers = dict(default_headers or {})
        self._timeout = timeout

    @property
    def _provider_id(self) -> str:
        return f"anthropic:{self.model}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self._version,
            **self._default_headers,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def _build_body(self, request: ModelRequest) -> dict[str, Any]:
        system, messages = split_anthropic_messages(request.messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens or self._max_tokens,
            "temperature": request.temperature,
        }
        if system:
            body["system"] = system
        return body

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send the Messages API request and return a ``ModelResponse``.

        Raises :class:`~loomable.kernel.errors.ModelProviderError` naming the provider
        when the endpoint is unreachable or returns a non-success status.
        """
        url = f"{self._base_url}/v1/messages"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=self._build_body(request), headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelProviderError(self._provider_id) from exc
        return parse_anthropic_response(data)

    async def stream(self, request: ModelRequest):
        """Stream a Messages API completion, yielding StreamEvents.

        Parses Anthropic SSE events (content_block_delta, message_delta, etc.)
        into text deltas and a terminal usage event.
        """
        url = f"{self._base_url}/v1/messages"
        body = self._build_body(request)
        body["stream"] = True

        usage: dict[str, int] = {}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", url, json=body, headers=self._headers()
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        import json as _json

                        payload = line[len("data:"):].strip()
                        try:
                            data = _json.loads(payload)
                        except (_json.JSONDecodeError, TypeError):
                            continue

                        event_type = data.get("type", "")

                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield StreamEvent(kind="text", text=text)

                        elif event_type == "message_start":
                            msg = data.get("message", {})
                            msg_usage = msg.get("usage", {})
                            if msg_usage.get("input_tokens"):
                                usage["input_tokens"] = msg_usage["input_tokens"]

                        elif event_type == "message_delta":
                            msg_usage = data.get("usage", {})
                            if msg_usage.get("output_tokens"):
                                usage["output_tokens"] = msg_usage["output_tokens"]

        except httpx.HTTPError as exc:
            raise _classify_http_error(self._provider_id, exc) from exc

        yield StreamEvent(kind="end", usage=usage)
