"""Unit tests for require_tools re-prompt with tools still enabled."""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.agent.tools import tool
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall


class _SkipWriteThenNudgeProvider:
    """Finish with text first; after require_tools nudge, call write then finish."""

    def __init__(self) -> None:
        self.call_count = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        self.requests.append(request)

        # After tools were used, return structured final.
        if any(
            m.get("role") == "tool" for m in request.messages if isinstance(m, dict)
        ):
            return ModelResponse(
                content='{"ok": true}',
                usage={"input_tokens": 3, "output_tokens": 4},
            )

        # Second call should be the require_tools nudge (tools still enabled).
        if self.call_count >= 2 and request.tools:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="w1",
                        tool_name="write_side_effect",
                        args={"path": "out.txt", "body": "hi"},
                    )
                ],
                usage={"input_tokens": 5, "output_tokens": 2},
            )

        # First call: try to finish without the required tool.
        return ModelResponse(
            content='{"ok": true}',
            usage={"input_tokens": 5, "output_tokens": 3},
        )


@tool
def write_side_effect(path: str, body: str) -> str:
    """Write a side-effect artifact."""
    return f"wrote:{path}:{body}"


@pytest.mark.asyncio
async def test_require_tools_nudges_then_calls_tool() -> None:
    provider = _SkipWriteThenNudgeProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        tools=[write_side_effect],
        require_tools=["write_side_effect"],
        max_tool_iterations=8,
    )

    result = await agent.arun("please write the file then answer")

    assert result.metadata.get("require_tools_nudged") is True
    assert "required_tools_missing" not in (result.metadata or {})
    names = [
        (o.result.metadata.get("tool_name") if o.result else None)
        for o in (result.tool_activity or [])
    ]
    assert "write_side_effect" in names
    # Nudge request must still advertise tools.
    assert any(r.tools for r in provider.requests[1:])


@pytest.mark.asyncio
async def test_require_tools_empty_skips_nudge() -> None:
    provider = _SkipWriteThenNudgeProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        tools=[write_side_effect],
        require_tools=[],
        max_tool_iterations=4,
    )

    result = await agent.arun("just answer")

    assert result.metadata.get("require_tools_nudged") is not True
    assert not (result.tool_activity or [])
    assert provider.call_count == 1


class _WriteJsonEmptyFinalProvider:
    """Call write_json then finish with empty text (schema recovered from tool)."""

    def __init__(self) -> None:
        self.call_count = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        if self.call_count == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="j1",
                        tool_name="write_json",
                        args={
                            "path": "out.json",
                            "content": '{"incident_id":"INC-1","severity":"SEV-1"}',
                        },
                    )
                ],
                usage={"input_tokens": 5, "output_tokens": 2},
            )
        # Final (and empty-text nudge): stay empty so recovery uses write_json.
        return ModelResponse(
            content="",
            usage={"input_tokens": 3, "output_tokens": 0},
        )


@tool
def write_json(path: str, content: str) -> str:
    """Write JSON artifact."""
    return f"Successfully wrote validated JSON to {path}"


@pytest.mark.asyncio
async def test_structured_recovers_from_write_json_when_final_empty() -> None:
    from pydantic import BaseModel

    class Packet(BaseModel):
        incident_id: str
        severity: str

    provider = _WriteJsonEmptyFinalProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        tools=[write_json],
        response_model=Packet,
        require_final_text=True,
        max_tool_iterations=6,
    )

    result = await agent.arun("write the packet")

    assert result.metadata.get("structured_from_write_json") is True
    assert result.structured is not None
    assert result.structured.incident_id == "INC-1"
    assert result.structured.severity == "SEV-1"
