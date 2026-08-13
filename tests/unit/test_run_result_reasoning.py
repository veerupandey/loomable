"""Unit tests for RunResult thoughts / plan / reasoning extraction."""

from __future__ import annotations

from loomable.agent.run import RunResult, extract_plan_steps, extract_thoughts
from loomable.content import AgentOutput, Text
from loomable.kernel.models import ToolResult
from loomable.providers._common import _extract_reasoning_segments


class _Outcome:
    def __init__(self, result: ToolResult | None = None, error: object | None = None) -> None:
        self.result = result
        self.error = error


def test_extract_thoughts_from_think_tool() -> None:
    activity = [
        _Outcome(ToolResult(content="step A", metadata={"tool_name": "think"})),
        _Outcome(ToolResult(content="other", metadata={"tool_name": "search"})),
        _Outcome(ToolResult(content="step B", metadata={"tool_name": "think"})),
    ]
    assert extract_thoughts(activity) == ["step A", "step B"]


def test_extract_plan_steps_from_metadata() -> None:
    activity = [
        _Outcome(
            ToolResult(
                content="done",
                metadata={"tool_name": "plan", "plan_steps": ["One", "Two"]},
            )
        )
    ]
    assert extract_plan_steps(activity) == ["One", "Two"]


def test_extract_plan_steps_from_json_content() -> None:
    activity = [
        _Outcome(
            ToolResult(
                content='{"plan_steps": ["A", "B"]}',
                metadata={"tool_name": "plan"},
            )
        )
    ]
    assert extract_plan_steps(activity) == ["A", "B"]


def test_extract_plan_steps_missing_returns_none() -> None:
    assert extract_plan_steps([]) is None
    assert extract_plan_steps(
        [_Outcome(ToolResult(content="x", metadata={"tool_name": "think"}))]
    ) is None


def test_extract_reasoning_segments_openai_compat() -> None:
    message = {"reasoning_content": "native thought"}
    data = {"reasoning": "top-level"}
    segs = _extract_reasoning_segments(message, data)
    assert "native thought" in segs
    assert "top-level" in segs


def test_run_result_fields_default_empty() -> None:
    result = RunResult(
        output=AgentOutput(parts=[Text("hi")]),
        session_id="s1",
    )
    assert result.thoughts == []
    assert result.plan is None
    assert result.reasoning == []
