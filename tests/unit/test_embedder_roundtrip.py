# Feature: agent-ergonomics, Property 15
"""Property 15: Embedder round-trip and unavailability.

For any text, a built-in Embedder returns a numeric vector via its endpoint
(mocked HTTP). An unavailable Embedder raises a ModelProviderError identifying
the embedder.

**Validates: Requirements 8.1, 8.4**
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from loomable.kernel.errors import ModelProviderError
from loomable.providers.embedders import (
    AzureOpenAIEmbedder,
    Embedder,
    OpenAIEmbedder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FAKE_REQUEST = httpx.Request("POST", "https://fake/embeddings")


def _make_embedding_response(vector: list[float]) -> httpx.Response:
    """Build a mock httpx.Response mimicking a successful /embeddings response."""
    return httpx.Response(
        status_code=200,
        json={"data": [{"embedding": vector}], "model": "text-embedding-3-small"},
        request=_FAKE_REQUEST,
    )


def _make_error_response(status_code: int = 503) -> httpx.Response:
    """Build a mock httpx.Response for an unavailable endpoint."""
    return httpx.Response(
        status_code=status_code,
        json={"error": {"message": "Service unavailable"}},
        request=_FAKE_REQUEST,
    )


# ---------------------------------------------------------------------------
# Tests — OpenAIEmbedder
# ---------------------------------------------------------------------------


class TestOpenAIEmbedderRoundTrip:
    """OpenAIEmbedder returns a numeric vector when the endpoint succeeds."""

    @pytest.mark.asyncio
    async def test_embed_returns_numeric_vector(self) -> None:
        """A successful call returns a list of floats matching the mock vector."""
        expected_vector = [0.1, 0.2, 0.3, -0.5, 1.0]
        mock_response = _make_embedding_response(expected_vector)

        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await embedder.embed("hello world")

        assert result == expected_vector
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_calls_correct_endpoint(self) -> None:
        """The embedder POSTs to {base_url}/embeddings with the correct body."""
        mock_response = _make_embedding_response([0.0, 1.0])

        embedder = OpenAIEmbedder(
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="https://custom.api.com/v1",
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await embedder.embed("test text")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://custom.api.com/v1/embeddings"
        body = call_args[1]["json"]
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == "test text"

    @pytest.mark.asyncio
    async def test_satisfies_embedder_protocol(self) -> None:
        """OpenAIEmbedder satisfies the Embedder protocol."""
        embedder = OpenAIEmbedder(api_key="test-key")
        assert isinstance(embedder, Embedder)


class TestOpenAIEmbedderUnavailability:
    """OpenAIEmbedder raises ModelProviderError when the endpoint is unavailable."""

    @pytest.mark.asyncio
    async def test_http_error_raises_model_provider_error(self) -> None:
        """A non-success HTTP status raises ModelProviderError identifying the embedder."""
        mock_response = _make_error_response(503)

        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(ModelProviderError) as exc_info:
                await embedder.embed("hello")

        assert "openai-embedder:text-embedding-3-small" in str(exc_info.value)
        assert exc_info.value.provider_id == "openai-embedder:text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_connection_error_raises_model_provider_error(self) -> None:
        """A network-level connection error raises ModelProviderError."""
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(ModelProviderError) as exc_info:
                await embedder.embed("hello")

        assert exc_info.value.provider_id == "openai-embedder:text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_timeout_raises_model_provider_error(self) -> None:
        """A request timeout raises ModelProviderError identifying the embedder."""
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timed out")
            with pytest.raises(ModelProviderError) as exc_info:
                await embedder.embed("hello")

        assert exc_info.value.provider_id == "openai-embedder:text-embedding-3-small"


# ---------------------------------------------------------------------------
# Tests — AzureOpenAIEmbedder
# ---------------------------------------------------------------------------


class TestAzureOpenAIEmbedderRoundTrip:
    """AzureOpenAIEmbedder returns a numeric vector when the endpoint succeeds."""

    @pytest.mark.asyncio
    async def test_embed_returns_numeric_vector(self) -> None:
        """A successful call returns a list of floats matching the mock vector."""
        expected_vector = [0.42, -0.13, 0.99, 0.0, -1.0]
        mock_response = _make_embedding_response(expected_vector)

        embedder = AzureOpenAIEmbedder(
            deployment="my-embed-deployment",
            endpoint="https://my-resource.openai.azure.com",
            api_key="azure-key",
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await embedder.embed("azure embedding test")

        assert result == expected_vector
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_satisfies_embedder_protocol(self) -> None:
        """AzureOpenAIEmbedder satisfies the Embedder protocol."""
        embedder = AzureOpenAIEmbedder(
            deployment="dep",
            endpoint="https://x.openai.azure.com",
            api_key="k",
        )
        assert isinstance(embedder, Embedder)


class TestAzureOpenAIEmbedderUnavailability:
    """AzureOpenAIEmbedder raises ModelProviderError when the endpoint is unavailable."""

    @pytest.mark.asyncio
    async def test_http_error_raises_model_provider_error(self) -> None:
        """A non-success HTTP status raises ModelProviderError identifying the embedder."""
        mock_response = _make_error_response(500)

        embedder = AzureOpenAIEmbedder(
            deployment="my-embed",
            endpoint="https://my-resource.openai.azure.com",
            api_key="azure-key",
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(ModelProviderError) as exc_info:
                await embedder.embed("test")

        assert "azure-openai-embedder:my-embed" in str(exc_info.value)
        assert exc_info.value.provider_id == "azure-openai-embedder:my-embed"

    @pytest.mark.asyncio
    async def test_connection_error_raises_model_provider_error(self) -> None:
        """A network-level connection error raises ModelProviderError."""
        embedder = AzureOpenAIEmbedder(
            deployment="my-embed",
            endpoint="https://my-resource.openai.azure.com",
            api_key="azure-key",
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(ModelProviderError) as exc_info:
                await embedder.embed("test")

        assert exc_info.value.provider_id == "azure-openai-embedder:my-embed"
