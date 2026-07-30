"""loomable.providers.embedders - Embedder protocol and built-in implementations.

:class:`Embedder` is a simple protocol: ``async embed(text) -> list[float]``.
:class:`OpenAIEmbedder` targets the OpenAI Embeddings API and any OpenAI-compatible
``/embeddings`` endpoint. :class:`AzureOpenAIEmbedder` targets Azure OpenAI
(deployment + api-version). Both raise :class:`~loomable.kernel.errors.ModelProviderError`
naming the embedder when the endpoint is unavailable.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import httpx

from loomable.kernel.errors import ModelProviderError

_DEFAULT_TIMEOUT = 60.0


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding text into a numeric vector."""

    async def embed(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """An :class:`Embedder` for OpenAI and OpenAI-compatible embedding endpoints.

    Parameters
    ----------
    model:
        The embedding model name (e.g. ``"text-embedding-3-small"``).
    api_key:
        The bearer API key. Defaults to the ``OPENAI_API_KEY`` environment variable.
    base_url:
        The API base URL. Defaults to ``https://api.openai.com/v1``. Point this at
        any OpenAI-compatible server.
    default_headers:
        Optional extra headers merged into every request.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        default_headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._default_headers = dict(default_headers or {})
        self._timeout = timeout

    @property
    def _embedder_id(self) -> str:
        return f"openai-embedder:{self.model}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._default_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_body(self, text: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": text,
        }

    async def embed(self, text: str) -> list[float]:
        """Embed ``text`` into a numeric vector via the /embeddings endpoint.

        Raises :class:`~loomable.kernel.errors.ModelProviderError` naming the
        embedder when the endpoint is unreachable or returns a non-success status.
        """
        url = f"{self._base_url}/embeddings"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url, json=self._build_body(text), headers=self._headers()
                )
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelProviderError(self._embedder_id) from exc
        return _parse_embedding(data, self._embedder_id)


class AzureOpenAIEmbedder:
    """An :class:`Embedder` for Azure OpenAI embedding deployments.

    Parameters
    ----------
    deployment:
        The Azure deployment name (used in the request URL).
        Defaults to the ``AZURE_OPENAI_EMBED_DEPLOYMENT_NAME`` environment variable.
    endpoint:
        The Azure resource endpoint (e.g. ``https://my-resource.openai.azure.com``).
        Defaults to the ``AZURE_OPENAI_ENDPOINT`` environment variable.
    api_key:
        The Azure API key (sent as the ``api-key`` header). Defaults to the
        ``AZURE_OPENAI_API_KEY`` environment variable.
    api_version:
        The Azure API version query parameter. Defaults to the
        ``AZURE_OPENAI_EMBED_API_VERSION`` environment variable, or ``2023-05-15``.
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
            else os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME")
        )
        if not self.deployment:
            raise ValueError(
                "Azure embedding deployment is required: pass deployment=... "
                "or set AZURE_OPENAI_EMBED_DEPLOYMENT_NAME."
            )
        resolved_endpoint = (
            endpoint if endpoint is not None else os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not resolved_endpoint:
            raise ValueError(
                "Azure endpoint is required: pass endpoint=... or set AZURE_OPENAI_ENDPOINT."
            )
        self._endpoint = resolved_endpoint.rstrip("/")
        self._api_key = (
            api_key if api_key is not None else os.environ.get("AZURE_OPENAI_API_KEY")
        )
        self._api_version = (
            api_version if api_version is not None
            else os.environ.get("AZURE_OPENAI_EMBED_API_VERSION", "2023-05-15")
        )
        self._default_headers = dict(default_headers or {})
        self._timeout = timeout

    @property
    def _embedder_id(self) -> str:
        return f"azure-openai-embedder:{self.deployment}"

    def _url(self) -> str:
        return (
            f"{self._endpoint}/openai/deployments/{self.deployment}"
            f"/embeddings?api-version={self._api_version}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._default_headers}
        if self._api_key:
            headers["api-key"] = self._api_key
        return headers

    def _build_body(self, text: str) -> dict[str, Any]:
        return {"input": text}

    async def embed(self, text: str) -> list[float]:
        """Embed ``text`` into a numeric vector via the Azure /embeddings endpoint.

        Raises :class:`~loomable.kernel.errors.ModelProviderError` naming the
        embedder when the endpoint is unreachable or returns a non-success status.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._url(), json=self._build_body(text), headers=self._headers()
                )
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelProviderError(self._embedder_id) from exc
        return _parse_embedding(data, self._embedder_id)


def _parse_embedding(data: dict[str, Any], embedder_id: str) -> list[float]:
    """Extract the embedding vector from an OpenAI-compatible /embeddings response."""
    try:
        return list(data["data"][0]["embedding"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelProviderError(embedder_id) from exc
