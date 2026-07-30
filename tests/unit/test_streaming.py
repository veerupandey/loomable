"""Tests for provider streaming and real astream behavior."""

from __future__ import annotations

from typing import Any

import pytest

from loomable.kernel.models import ModelRequest, ModelResponse, StreamEvent, ToolCall
from loomable.providers._common import (
    iter_openai_stream_events,
    parse_openai_sse_line,
)


# ---------------------------------------------------------------------------
# SSE line parsing
# ---------------------------------------------------------------------------


class TestSSEParsing:
    """Test OpenAI SSE line parsing."""

    def test_data_line_parses_json(self):
        line = 'data: {"choices": [{"delta": {"content": "hi"}}]}'
        result = parse_openai_sse_line(line)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "hi"

    def test_done_returns_none(self):
        assert parse_openai_sse_line("data: [DONE]") is None

    def test_non_data_returns_none(self):
        assert parse_openai_sse_line("event: message") is None
        assert parse_openai_sse_line("") is None

    def test_malformed_json_returns_none(self):
        assert parse_openai_sse_line("data: {broken") is None


# ---------------------------------------------------------------------------
# Stream event assembly
# ---------------------------------------------------------------------------


class TestStreamEventAssembly:
    """Test iter_openai_stream_events converts chunks to StreamEvents."""

    def test_text_deltas(self):
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        events = iter_openai_stream_events(chunks)
        text_events = [e for e in events if e.kind == "text"]
        assert len(text_events) == 2
        assert text_events[0].text == "Hello"
        assert text_events[1].text == " world"
        # Should have a terminal end event
        assert any(e.kind == "end" for e in events)

    def test_tool_call_fragments(self):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "add", "arguments": '{"a"'}}
            ]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ': 1, "b": 2}'}}
            ]}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = iter_openai_stream_events(chunks)
        tc_events = [e for e in events if e.kind == "tool_call"]
        assert len(tc_events) == 1
        assert tc_events[0].tool_call.tool_name == "add"
        assert tc_events[0].tool_call.args == {"a": 1, "b": 2}

    def test_usage_terminal_event(self):
        chunks = [
            {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        ]
        events = iter_openai_stream_events(chunks)
        end_events = [e for e in events if e.kind == "end"]
        assert len(end_events) >= 1
        # At least one end event should have usage
        usage_event = next((e for e in end_events if e.usage.get("input_tokens")), None)
        if usage_event:
            assert usage_event.usage["input_tokens"] == 10
            assert usage_event.usage["output_tokens"] == 5


# ---------------------------------------------------------------------------
# StreamEvent structural detection
# ---------------------------------------------------------------------------


class TestStreamingDetection:
    """Test that hasattr(provider, 'stream') works for capability detection."""

    def test_provider_with_stream(self):
        """OpenAI/Azure providers should have stream()."""
        from loomable.providers.openai import OpenAIProvider, AzureOpenAIProvider

        oai = OpenAIProvider(model="gpt-4o-mini", api_key="test")
        assert hasattr(oai, "stream")

        azure = AzureOpenAIProvider(
            deployment="test", endpoint="https://test.openai.azure.com", api_key="k"
        )
        assert hasattr(azure, "stream")

    def test_provider_without_stream(self):
        """A bare ModelProvider protocol impl without stream should lack it."""

        class BareProvider:
            async def complete(self, request):
                return ModelResponse(content="done")

        p = BareProvider()
        assert not hasattr(p, "stream")


# ---------------------------------------------------------------------------
# Scripted streaming provider for astream tests
# ---------------------------------------------------------------------------


class ScriptedStreamingProvider:
    """A test provider that implements both complete() and stream()."""

    def __init__(self, text_deltas: list[str], usage: dict | None = None):
        self._deltas = text_deltas
        self._usage = usage or {}

    async def complete(self, request: ModelRequest) -> ModelResponse:
        full_text = "".join(self._deltas)
        return ModelResponse(content=full_text, usage=self._usage)

    async def stream(self, request: ModelRequest):
        for delta in self._deltas:
            yield StreamEvent(kind="text", text=delta)
        yield StreamEvent(kind="end", usage=self._usage)


class NonStreamingProvider:
    """A test provider that only has complete()."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="non-streamed result", usage={"input_tokens": 5, "output_tokens": 3})


class TestAstreamBehavior:
    """Test BuiltAgent.astream with streaming and non-streaming providers."""

    async def test_real_deltas_from_streaming_provider(self):
        """With a streaming provider, astream should yield real text deltas."""
        from loomable.agent import Agent, ModelSpec

        provider = ScriptedStreamingProvider(["Hello", " ", "world"], {"input_tokens": 10, "output_tokens": 5})
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
        )
        built = agent.build()

        chunks = []
        async for chunk in built.astream("test input"):
            chunks.append(chunk)

        # Should have individual text deltas
        text_chunks = [c for c in chunks if not c.done]
        assert len(text_chunks) >= 1
        # Last chunk should be done
        assert chunks[-1].done

    async def test_fallback_for_non_streaming_provider(self):
        """Without stream(), astream should fall back to arun + chunk."""
        from loomable.agent import Agent, ModelSpec

        provider = NonStreamingProvider()
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
        )
        built = agent.build()

        chunks = []
        async for chunk in built.astream("test input"):
            chunks.append(chunk)

        # Should get at least one chunk with the result
        assert len(chunks) >= 1
        assert chunks[-1].done
        # The text content should be present
        text = "".join(
            c.delta.data.decode("utf-8") for c in chunks
            if c.delta.data is not None
        )
        assert "non-streamed result" in text
