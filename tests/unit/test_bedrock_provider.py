"""Unit tests for the Amazon Bedrock provider (Converse API).

These exercise request translation, response/stream parsing, error
classification, and the async ``complete`` / ``stream`` paths using an injected
fake ``bedrock-runtime`` client — no AWS calls or credentials required.
"""

from __future__ import annotations

import pytest

from loomable.kernel.models import ModelRequest
from loomable.providers import BedrockProvider
from loomable.providers.bedrock import _iter_converse_stream, _parse_converse_response
from loomable.providers.errors import PermanentProviderError, TransientProviderError
from loomable.providers.resolver import resolve_model


class _FakeClient:
    """Minimal stand-in for a boto3 bedrock-runtime client."""

    def __init__(self, response=None, stream_events=None, raise_exc=None):
        self._response = response or {}
        self._stream_events = stream_events or []
        self._raise_exc = raise_exc
        self.last_kwargs = None

    def converse(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response

    def converse_stream(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return {"stream": list(self._stream_events)}


def _sample_request() -> ModelRequest:
    return ModelRequest(
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "calling"}],
                "tool_calls": [{"id": "t1", "tool_name": "add", "args": {"a": 1, "b": 2}}],
            },
            {"role": "tool", "content": [{"type": "text", "text": "3"}], "tool_call_id": "t1"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "add two ints",
                    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
                },
            }
        ],
        temperature=0.2,
        max_tokens=256,
    )


def test_resolver_maps_bedrock_prefix():
    p = resolve_model("bedrock:anthropic.claude-3-haiku-20240307-v1:0")
    assert isinstance(p, BedrockProvider)
    # split(":", 1) keeps the trailing version segment intact.
    assert p.model == "anthropic.claude-3-haiku-20240307-v1:0"


def test_resolver_aws_alias():
    assert isinstance(resolve_model("aws:amazon.nova-lite-v1:0"), BedrockProvider)


def test_build_kwargs_translation():
    p = BedrockProvider("m", region_name="ca-central-1")
    kwargs = p._build_kwargs(_sample_request())

    assert kwargs["modelId"] == "m"
    assert kwargs["system"] == [{"text": "Be terse."}]
    assert kwargs["inferenceConfig"] == {"maxTokens": 256, "temperature": 0.2}

    roles = [m["role"] for m in kwargs["messages"]]
    assert roles == ["user", "assistant", "user"]

    assistant_blocks = kwargs["messages"][1]["content"]
    tool_use = [b for b in assistant_blocks if "toolUse" in b][0]["toolUse"]
    assert tool_use["name"] == "add"
    assert tool_use["input"] == {"a": 1, "b": 2}

    tool_result = kwargs["messages"][2]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "t1"

    spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "add"
    assert spec["inputSchema"]["json"]["type"] == "object"


def test_transcript_starts_with_user():
    p = BedrockProvider("m")
    # Leading assistant message must be prefixed with a synthetic user turn.
    _system, msgs = p._to_converse_messages(
        [{"role": "assistant", "content": "hello"}]
    )
    assert msgs[0]["role"] == "user"


def test_consecutive_same_role_merged():
    p = BedrockProvider("m")
    _system, msgs = p._to_converse_messages(
        [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
    )
    assert len(msgs) == 1
    assert [blk["text"] for blk in msgs[0]["content"]] == ["a", "b"]


def test_parse_converse_response():
    data = {
        "output": {
            "message": {
                "content": [
                    {"text": "hello"},
                    {"toolUse": {"toolUseId": "u1", "name": "add", "input": {"a": 1}}},
                    {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
                ]
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "stopReason": "tool_use",
    }
    r = _parse_converse_response(data)
    assert r.content == "hello"
    assert r.tool_calls[0].tool_name == "add"
    assert r.tool_calls[0].args == {"a": 1}
    assert r.usage == {"input_tokens": 10, "output_tokens": 5}
    assert r.reasoning == ["thinking..."]
    assert r.metadata["stop_reason"] == "tool_use"


def test_iter_converse_stream_assembles_text_and_tools():
    stream = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "he"}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "llo"}}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "u1", "name": "add"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"a":'}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "1}"}}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 4}}},
    ]
    evs = _iter_converse_stream(stream)
    text = "".join(e.text for e in evs if e.kind == "text")
    tool = [e.tool_call for e in evs if e.kind == "tool_call"][0]
    end = [e for e in evs if e.kind == "end"][0]
    assert text == "hello"
    assert tool.tool_name == "add"
    assert tool.args == {"a": 1}
    assert end.usage == {"input_tokens": 3, "output_tokens": 4}


@pytest.mark.asyncio
async def test_complete_with_fake_client():
    fake = _FakeClient(
        response={
            "output": {"message": {"content": [{"text": "Paris."}]}},
            "usage": {"inputTokens": 8, "outputTokens": 2},
            "stopReason": "end_turn",
        }
    )
    p = BedrockProvider("amazon.nova-lite-v1:0", client=fake)
    resp = await p.complete(ModelRequest(messages=[{"role": "user", "content": "capital of France?"}]))
    assert resp.content == "Paris."
    assert resp.usage == {"input_tokens": 8, "output_tokens": 2}
    assert fake.last_kwargs["modelId"] == "amazon.nova-lite-v1:0"


@pytest.mark.asyncio
async def test_stream_with_fake_client():
    fake = _FakeClient(
        stream_events=[
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "red "}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "green"}}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2}}},
        ]
    )
    p = BedrockProvider("m", client=fake)
    kinds, text = [], ""
    async for ev in p.stream(ModelRequest(messages=[{"role": "user", "content": "colors"}])):
        kinds.append(ev.kind)
        if ev.kind == "text":
            text += ev.text
    assert text == "red green"
    assert kinds[-1] == "end"


@pytest.mark.asyncio
async def test_error_classification_transient_vs_permanent():
    botocore = pytest.importorskip("botocore")
    from botocore.exceptions import ClientError

    throttle = ClientError(
        {"Error": {"Code": "ThrottlingException"}, "ResponseMetadata": {"HTTPStatusCode": 429}},
        "Converse",
    )
    denied = ClientError(
        {"Error": {"Code": "AccessDeniedException"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
        "Converse",
    )

    p_t = BedrockProvider("m", client=_FakeClient(raise_exc=throttle))
    with pytest.raises(TransientProviderError):
        await p_t.complete(ModelRequest(messages=[{"role": "user", "content": "x"}]))

    p_p = BedrockProvider("m", client=_FakeClient(raise_exc=denied))
    with pytest.raises(PermanentProviderError):
        await p_p.complete(ModelRequest(messages=[{"role": "user", "content": "x"}]))
