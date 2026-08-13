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


class _PartialRequireToolsProvider:
    """Finish early twice: nudge1 → write_a; finish again → nudge2 → write_b."""

    def __init__(self) -> None:
        self.a_done = False
        self.b_done = False

    async def complete(self, request: ModelRequest) -> ModelResponse:
        last = request.messages[-1] if request.messages else {}
        if isinstance(last, dict) and last.get("role") == "tool":
            # After each write, attempt to finish (framework re-nudges if needed).
            return ModelResponse(
                content='{"ok": true}',
                usage={"input_tokens": 2, "output_tokens": 2},
            )

        last_user = ""
        for msg in reversed(request.messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                last_user = " ".join(
                    str(p.get("text", "")) for p in content if isinstance(p, dict)
                )
            else:
                last_user = str(content or "")
            break

        if "required tools" in last_user:
            if not self.a_done:
                self.a_done = True
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="a1", tool_name="write_a", args={"x": "1"})
                    ],
                    usage={"input_tokens": 3, "output_tokens": 2},
                )
            if not self.b_done:
                self.b_done = True
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="b1", tool_name="write_b", args={"x": "2"})
                    ],
                    usage={"input_tokens": 3, "output_tokens": 2},
                )

        return ModelResponse(
            content='{"ok": true}',
            usage={"input_tokens": 5, "output_tokens": 3},
        )


@tool
def write_a(x: str) -> str:
    """Write artifact A."""
    return f"a:{x}"


@tool
def write_b(x: str) -> str:
    """Write artifact B."""
    return f"b:{x}"


@pytest.mark.asyncio
async def test_require_tools_renudges_until_all_called() -> None:
    provider = _PartialRequireToolsProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        tools=[write_a, write_b],
        require_tools=["write_a", "write_b"],
        max_tool_iterations=10,
    )

    result = await agent.arun("write both")

    assert result.metadata.get("require_tools_nudges", 0) >= 2
    assert "required_tools_missing" not in (result.metadata or {})
    names = [
        (o.result.metadata.get("tool_name") if o.result else None)
        for o in (result.tool_activity or [])
    ]
    assert "write_a" in names and "write_b" in names


class _WrongPathThenFixProvider:
    """write_file to wrong path, then after path nudge write correct path."""

    def __init__(self) -> None:
        self.fixed = False

    async def complete(self, request: ModelRequest) -> ModelResponse:
        last = request.messages[-1] if request.messages else {}
        if isinstance(last, dict) and last.get("role") == "tool":
            return ModelResponse(
                content='{"ok": true}',
                usage={"input_tokens": 2, "output_tokens": 2},
            )

        last_user = ""
        for msg in reversed(request.messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                last_user = " ".join(
                    str(p.get("text", "")) for p in content if isinstance(p, dict)
                )
            else:
                last_user = str(content or "")
            break

        if "path matching" in last_user and not self.fixed:
            self.fixed = True
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="w2",
                        tool_name="write_file",
                        args={"path": "output/brief.md", "content": "ok"},
                    )
                ],
                usage={"input_tokens": 3, "output_tokens": 2},
            )

        if not self.fixed:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="w1",
                        tool_name="write_file",
                        args={"path": "wrong.txt", "content": "nope"},
                    )
                ],
                usage={"input_tokens": 3, "output_tokens": 2},
            )

        return ModelResponse(
            content='{"ok": true}',
            usage={"input_tokens": 2, "output_tokens": 2},
        )


@tool
def write_file(path: str, content: str) -> str:
    """Write a text file."""
    return f"wrote:{path}"


def test_path_constraint_met_exact_and_suffix_not_substring() -> None:
    from loomable.agent.builder import _path_constraint_met

    assert _path_constraint_met("output/brief.md", "output/brief.md")
    assert _path_constraint_met("./output/brief.md", "output/brief.md")
    assert _path_constraint_met("workspace/output/brief.md", "output/brief.md")
    # Substring false positive must NOT match
    assert not _path_constraint_met("myoutput/brief.md", "output/brief.md")
    assert not _path_constraint_met("wrong.txt", "output/brief.md")


@pytest.mark.asyncio
async def test_require_tools_path_constraint_rejects_substring_false_positive() -> None:
    """``myoutput/brief.md`` must not satisfy ``write_file:output/brief.md``."""

    class _SubstringThenFix:
        def __init__(self) -> None:
            self.fixed = False

        async def complete(self, request: ModelRequest) -> ModelResponse:
            last = request.messages[-1] if request.messages else {}
            if isinstance(last, dict) and last.get("role") == "tool":
                return ModelResponse(
                    content='{"ok": true}',
                    usage={"input_tokens": 2, "output_tokens": 2},
                )

            last_user = ""
            for msg in reversed(request.messages):
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    last_user = " ".join(
                        str(p.get("text", "")) for p in content if isinstance(p, dict)
                    )
                else:
                    last_user = str(content or "")
                break

            if "path matching" in last_user and not self.fixed:
                self.fixed = True
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="w2",
                            tool_name="write_file",
                            args={"path": "output/brief.md", "content": "ok"},
                        )
                    ],
                    usage={"input_tokens": 3, "output_tokens": 2},
                )

            if not self.fixed:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="w1",
                            tool_name="write_file",
                            args={"path": "myoutput/brief.md", "content": "nope"},
                        )
                    ],
                    usage={"input_tokens": 3, "output_tokens": 2},
                )

            return ModelResponse(
                content='{"ok": true}',
                usage={"input_tokens": 2, "output_tokens": 2},
            )

    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_SubstringThenFix()),
        tools=[write_file],
        require_tools=["write_file:output/brief.md"],
        max_tool_iterations=10,
    )
    result = await agent.arun("write the brief")
    assert result.metadata.get("require_tools_nudged") is True
    assert "required_tools_missing" not in (result.metadata or {})


@pytest.mark.asyncio
async def test_require_tools_path_constraint_renudges() -> None:
    provider = _WrongPathThenFixProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        tools=[write_file],
        require_tools=["write_file:output/brief.md"],
        max_tool_iterations=10,
    )

    result = await agent.arun("write the brief")

    assert result.metadata.get("require_tools_nudged") is True
    assert "required_tools_missing" not in (result.metadata or {})
    paths = []
    for o in result.tool_activity or []:
        # path is not on outcome; ensure at least two write_file calls happened
        if o.result and o.result.metadata.get("tool_name") == "write_file":
            paths.append(o.result.content)
    assert any("output/brief.md" in (c or "") for c in paths)
