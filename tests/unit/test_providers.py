"""Unit tests for the built-in model providers (loomable.providers).

Covers OpenAIProvider, AzureOpenAIProvider, and AnthropicProvider using mocked HTTP
(pytest-httpx) so no live API is required. Verifies request translation, response
parsing, integration with the high-level Agent, and error handling.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from loomable.agent import Agent
from loomable.content import AgentInput, Image, Message, Text
from loomable.kernel.errors import ModelProviderError
from loomable.kernel.models import ModelRequest
from loomable.providers import AnthropicProvider, AzureOpenAIProvider, OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_request(text: str = "hello") -> ModelRequest:
    """A ModelRequest with a single user text message (content-array form)."""
    return ModelRequest(
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}]
    )


def _openai_response(content: str = "hi there") -> dict:
    return {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


def _anthropic_response(content: str = "hi there") -> dict:
    return {
        "model": "claude-3-5-sonnet-latest",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    async def test_complete_parses_response(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            json=_openai_response("the answer"),
        )
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
        resp = await provider.complete(_text_request())

        assert resp.content == "the answer"
        assert resp.usage == {"input_tokens": 5, "output_tokens": 3}

    async def test_sends_model_and_bearer_auth(self, httpx_mock):
        httpx_mock.add_response(json=_openai_response())
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-secret")
        await provider.complete(_text_request("hi"))

        import json as _json

        req = httpx_mock.get_request()
        assert req.headers["authorization"] == "Bearer sk-secret"
        assert _json.loads(req.read())["model"] == "gpt-4o-mini"

    async def test_flattens_text_only_content_to_string(self, httpx_mock):
        httpx_mock.add_response(json=_openai_response())
        provider = OpenAIProvider(model="m", api_key="k")
        await provider.complete(_text_request("just text"))

        import json as _json

        body = _json.loads(httpx_mock.get_request().read())
        # A text-only message is flattened to a plain string for broad compatibility.
        assert body["messages"][0]["content"] == "just text"

    async def test_keeps_image_url_for_multimodal(self, httpx_mock):
        httpx_mock.add_response(json=_openai_response())
        provider = OpenAIProvider(model="m", api_key="k")
        request = ModelRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                }
            ]
        )
        await provider.complete(request)

        import json as _json

        body = _json.loads(httpx_mock.get_request().read())
        content = body["messages"][0]["content"]
        assert isinstance(content, list)
        assert {"type": "text", "text": "look"} in content
        assert any(p["type"] == "image_url" for p in content)

    async def test_custom_base_url_for_compatible_server(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:11434/v1/chat/completions",
            json=_openai_response("local"),
        )
        provider = OpenAIProvider(
            model="llama3", api_key=None, base_url="http://localhost:11434/v1"
        )
        resp = await provider.complete(_text_request())
        assert resp.content == "local"
        # No Authorization header when api_key is None (local servers).
        assert "authorization" not in {k.lower() for k in httpx_mock.get_request().headers}

    async def test_http_error_raises_model_provider_error(self, httpx_mock):
        httpx_mock.add_response(status_code=500)
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="k")
        with pytest.raises(ModelProviderError) as exc:
            await provider.complete(_text_request())
        assert exc.value.provider_id == "openai:gpt-4o-mini"

    async def test_works_with_high_level_agent(self, httpx_mock):
        httpx_mock.add_response(json=_openai_response("agent reply"))
        agent = Agent(model=OpenAIProvider(model="gpt-4o-mini", api_key="k"))
        result = await agent.arun("hello")
        assert result.output.text() == "agent reply"


# ---------------------------------------------------------------------------
# AzureOpenAIProvider
# ---------------------------------------------------------------------------


class TestAzureOpenAIProvider:
    async def test_builds_deployment_url_and_api_key_header(self, httpx_mock):
        httpx_mock.add_response(json=_openai_response("azure reply"))
        provider = AzureOpenAIProvider(
            deployment="gpt-4o-mini",
            endpoint="https://res.openai.azure.com",
            api_key="azkey",
            api_version="2024-08-01-preview",
        )
        resp = await provider.complete(_text_request())

        assert resp.content == "azure reply"
        req = httpx_mock.get_request()
        assert "deployments/gpt-4o-mini/chat/completions" in str(req.url)
        assert "api-version=2024-08-01-preview" in str(req.url)
        assert req.headers["api-key"] == "azkey"
        # Azure carries the model via the URL, so no "model" field in the body.
        import json as _json

        assert "model" not in _json.loads(req.read())

    async def test_missing_endpoint_raises(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="Azure endpoint"):
            AzureOpenAIProvider(deployment="d", api_key="k")

    async def test_error_names_provider(self, httpx_mock):
        httpx_mock.add_response(status_code=404)
        provider = AzureOpenAIProvider(
            deployment="dep", endpoint="https://res.openai.azure.com", api_key="k"
        )
        with pytest.raises(ModelProviderError) as exc:
            await provider.complete(_text_request())
        assert exc.value.provider_id == "azure-openai:dep"


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    async def test_complete_parses_content_blocks(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            json=_anthropic_response("claude says hi"),
        )
        provider = AnthropicProvider(model="claude-3-5-sonnet-latest", api_key="ak")
        resp = await provider.complete(_text_request())

        assert resp.content == "claude says hi"
        assert resp.usage == {"input_tokens": 5, "output_tokens": 3}

    async def test_extracts_system_prompt_and_headers(self, httpx_mock):
        httpx_mock.add_response(json=_anthropic_response())
        provider = AnthropicProvider(model="claude-x", api_key="secret")
        request = ModelRequest(
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "be terse"}]},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ]
        )
        await provider.complete(request)

        import json as _json

        req = httpx_mock.get_request()
        assert req.headers["x-api-key"] == "secret"
        assert req.headers["anthropic-version"] == "2023-06-01"
        body = _json.loads(req.read())
        # System message is lifted to the top-level 'system' field, not a message.
        assert body["system"] == "be terse"
        assert [m["role"] for m in body["messages"]] == ["user"]
        assert body["max_tokens"] == 1024  # default applied

    async def test_image_data_uri_becomes_base64_source(self, httpx_mock):
        httpx_mock.add_response(json=_anthropic_response())
        raw = b"\x89PNG-bytes"
        data_uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        provider = AnthropicProvider(model="claude-x", api_key="k")
        request = ModelRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ]
        )
        await provider.complete(request)

        import json as _json

        body = _json.loads(httpx_mock.get_request().read())
        blocks = body["messages"][0]["content"]
        image_block = next(b for b in blocks if b["type"] == "image")
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] == "image/png"
        assert base64.b64decode(image_block["source"]["data"]) == raw

    async def test_error_names_provider(self, httpx_mock):
        httpx_mock.add_response(status_code=429)
        provider = AnthropicProvider(model="claude-x", api_key="k")
        with pytest.raises(ModelProviderError) as exc:
            await provider.complete(_text_request())
        assert exc.value.provider_id == "anthropic:claude-x"

    async def test_works_with_high_level_agent(self, httpx_mock):
        httpx_mock.add_response(json=_anthropic_response("claude reply"))
        agent = Agent(model=AnthropicProvider(model="claude-x", api_key="k"))
        result = await agent.arun("hello")
        assert result.output.text() == "claude reply"
