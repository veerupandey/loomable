"""loomable.providers.embedders - Embedder protocol and built-in implementations.

:class:`Embedder` is a simple protocol: ``async embed(text) -> list[float]``.
Optional ``async embed_many(texts) -> list[list[float]]`` for batch indexing.

Built-ins: OpenAI, Azure OpenAI, Gemini, Hugging Face (local sentence-transformers
or Inference API).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Protocol, Sequence, runtime_checkable

import httpx

from loomable.kernel.errors import ModelProviderError

_DEFAULT_TIMEOUT = 60.0


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding text into a numeric vector."""

    async def embed(self, text: str) -> list[float]: ...


async def embed_many(embedder: Any, texts: Sequence[str]) -> list[list[float]]:
    """Batch embed via ``embed_many`` if present, else sequential ``embed``."""
    if hasattr(embedder, "embed_many"):
        return await embedder.embed_many(list(texts))
    out: list[list[float]] = []
    for t in texts:
        out.append(await embedder.embed(t))
    return out


class OpenAIEmbedder:
    """OpenAI and OpenAI-compatible ``/embeddings`` endpoints."""

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

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        return vectors[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self._base_url}/embeddings"
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts[0] if len(texts) == 1 else list(texts),
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelProviderError(self._embedder_id) from exc
        return _parse_embeddings_batch(data, self._embedder_id, len(texts))


class AzureOpenAIEmbedder:
    """Azure OpenAI embedding deployments."""

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
            deployment
            if deployment is not None
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
            api_version
            if api_version is not None
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

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        return vectors[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        body = {"input": texts[0] if len(texts) == 1 else list(texts)}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url(), json=body, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelProviderError(self._embedder_id) from exc
        return _parse_embeddings_batch(data, self._embedder_id, len(texts))


class GeminiEmbedder(OpenAIEmbedder):
    """Google Gemini embeddings (OpenAI-compatible API).

    Default model ``gemini-embedding-001`` (3072-d). Auth via ``GEMINI_API_KEY``
    or ``GOOGLE_API_KEY``.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        *,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        default_headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved = (
            api_key
            if api_key is not None
            else os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))
        )
        super().__init__(
            model=model,
            api_key=resolved,
            base_url=base_url,
            default_headers=default_headers,
            timeout=timeout,
        )

    @property
    def _embedder_id(self) -> str:
        return f"gemini-embedder:{self.model}"


class HuggingFaceEmbedder:
    """Local sentence-transformers or Hugging Face Inference API embeddings."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        backend: str = "local",
        api_key: str | None = None,
        token: str | None = None,
        device: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.backend = (backend or "local").strip().lower()
        self._api_key = (
            api_key
            if api_key is not None
            else token
            if token is not None
            else os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
        )
        self._device = device
        self._timeout = timeout
        self._local: Any | None = None
        if self.backend not in {"local", "api"}:
            raise ValueError("backend must be 'local' or 'api'")

    @property
    def _embedder_id(self) -> str:
        return f"huggingface-embedder:{self.backend}:{self.model}"

    def _load_local(self) -> Any:
        if self._local is not None:
            return self._local
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HuggingFaceEmbedder(backend='local') requires sentence-transformers. "
                "Install with: pip install loomable[huggingface]"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self._device:
            kwargs["device"] = self._device
        self._local = SentenceTransformer(self.model, **kwargs)
        return self._local

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        return vectors[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.backend == "local":
            model = self._load_local()

            def _encode() -> list[list[float]]:
                arr = model.encode(list(texts), normalize_embeddings=True)
                return [list(map(float, row)) for row in arr]

            return await asyncio.to_thread(_encode)

        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url, json={"inputs": list(texts)}, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelProviderError(self._embedder_id) from exc
        return _parse_hf_features(data, self._embedder_id, len(texts))


def _parse_embeddings_batch(
    data: dict[str, Any], embedder_id: str, n: int
) -> list[list[float]]:
    try:
        rows = sorted(data["data"], key=lambda r: int(r.get("index", 0)))
        out = [list(map(float, r["embedding"])) for r in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelProviderError(embedder_id) from exc
    if len(out) != n:
        raise ModelProviderError(embedder_id)
    return out


def _parse_hf_features(data: Any, embedder_id: str, n: int) -> list[list[float]]:
    try:
        if n == 1 and data and isinstance(data[0], (int, float)):
            return [list(map(float, data))]
        out: list[list[float]] = []
        for item in data:
            if item and isinstance(item[0], (list, tuple)):
                dim = len(item[0])
                acc = [0.0] * dim
                for tok in item:
                    for i, v in enumerate(tok):
                        acc[i] += float(v)
                out.append([v / max(1, len(item)) for v in acc])
            else:
                out.append(list(map(float, item)))
        if len(out) != n:
            raise ModelProviderError(embedder_id)
        return out
    except (TypeError, IndexError, ValueError) as exc:
        raise ModelProviderError(embedder_id) from exc
