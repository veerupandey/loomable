"""Integration test for transport parity (task 7.1).

Property 15: Transport parity — *for any* ``BuiltAgent`` and equivalent
``AgentInput``, in-process ``arun``, the FastAPI ``/run`` endpoint, and the MCP
run tool SHALL produce equivalent ``AgentOutput``.

Validates: Requirements 9.1, 9.2

The three transports each wrap the *same* ``BuiltAgent`` and call its identical
``arun`` method, so the same kernel loop executes regardless of transport
(Req 9.1). Both adapters operate on the ``BuiltAgent`` produced by the builder
(Req 9.2). To keep the three outputs directly comparable the agent is built
WITHOUT a ``session_id`` so per-run persistence is disabled entirely — no
transport mutates shared session state between calls.

The provider is a deterministic fake whose output is derived purely from the
input text, so equivalent inputs must yield identical outputs across transports.

Feature: agent-api, Property 15
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from loomable.agent import Agent, ModelSpec
from loomable.content import AgentInput
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.serve import FastAPIAdapter, MCPServerAdapter


class DeterministicProvider:
    """Fake ModelProvider returning content derived deterministically from input.

    It concatenates every text part it receives and transforms it with a fixed,
    pure rule. Because the output depends only on the input text (no clocks,
    randomness, or hidden state), equivalent inputs always yield identical
    outputs — the precondition for a clean transport-parity assertion.
    """

    async def complete(self, request: ModelRequest) -> ModelResponse:
        pieces: list[str] = []
        for message in request.messages:
            for part in message.get("content", []):
                if part.get("type") == "text":
                    pieces.append(part.get("text", ""))
        joined = " ".join(p for p in pieces if p)
        return ModelResponse(
            content=f"processed:{joined.upper()}",
            usage={"input": len(joined), "output": 1},
        )


#: The single text input driven through every transport.
_INPUT_TEXT = "parity check please"


def _build_agent():
    """Build one BuiltAgent (no session_id → persistence off) shared by all transports."""
    return Agent(
        model=ModelSpec(provider="fake", provider_impl=DeterministicProvider()),
    ).build()


def _text_from_run_response(body: dict) -> str:
    """Extract concatenated text from a FastAPI ``/run`` JSON body."""
    return "".join(
        part.get("text") or "" for part in body["output"] if part["modality"] == "text"
    )


def _text_from_mcp_result(result) -> str:
    """Extract concatenated text from an MCP ``CallToolResult``."""
    return "".join(
        block.text for block in result.content if getattr(block, "type", None) == "text"
    )


@pytest.mark.integration
async def test_transport_parity_equivalent_output() -> None:
    """In-process, FastAPI, and MCP transports produce equivalent AgentOutput (Req 9.1, 9.2)."""
    built = _build_agent()

    # 1) In-process: drive the BuiltAgent directly.
    in_process_result = await built.arun(AgentInput.from_text(_INPUT_TEXT))
    in_process_text = in_process_result.output.text()

    # 2) Over FastAPI: same BuiltAgent, wrapped by the HTTP adapter.
    app = FastAPIAdapter(built).app()
    client = TestClient(app)
    payload = {
        "messages": [
            {"role": "user", "parts": [{"modality": "text", "text": _INPUT_TEXT}]}
        ]
    }
    http_response = client.post("/run", json=payload)
    assert http_response.status_code == 200
    fastapi_text = _text_from_run_response(http_response.json())

    # 3) Over MCP: same BuiltAgent, wrapped by the MCP adapter.
    mcp_adapter = MCPServerAdapter(built)
    mcp_result = await mcp_adapter.run_tool({"text": _INPUT_TEXT})
    assert mcp_result.is_error is False
    mcp_text = _text_from_mcp_result(mcp_result)

    # Parity: the deterministic provider makes the expected output exact.
    expected = f"processed:{_INPUT_TEXT.upper()}"
    assert in_process_text == expected
    assert fastapi_text == expected
    assert mcp_text == expected

    # And the three transports agree with one another.
    assert in_process_text == fastapi_text == mcp_text
