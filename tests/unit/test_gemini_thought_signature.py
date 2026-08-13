"""Tests for Gemini thought_signature / extra_content preservation."""

from __future__ import annotations

import json

from loomable.kernel.models import ToolCall
from loomable.providers._common import parse_openai_response, to_openai_messages


def test_parse_openai_response_keeps_extra_content() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a":1,"b":2}',
                            },
                            "extra_content": {
                                "google": {"thought_signature": "sig123"}
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "model": "gemini-flash-latest",
    }
    resp = parse_openai_response(data)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "add"
    assert resp.tool_calls[0].metadata["extra_content"]["google"][
        "thought_signature"
    ] == "sig123"


def test_to_openai_messages_replays_extra_content() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "tool_name": "add",
                    "args": {"a": 1, "b": 2},
                    "metadata": {
                        "extra_content": {
                            "google": {"thought_signature": "sig123"}
                        }
                    },
                    "extra_content": {
                        "google": {"thought_signature": "sig123"}
                    },
                }
            ],
        }
    ]
    out = to_openai_messages(messages)
    assert out[0]["tool_calls"][0]["extra_content"]["google"][
        "thought_signature"
    ] == "sig123"
    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {
        "a": 1,
        "b": 2,
    }


def test_toolcall_metadata_default_empty() -> None:
    tc = ToolCall(id="x", tool_name="t", args={})
    assert tc.metadata == {}
