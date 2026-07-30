# Feature: agent-ergonomics, Property 9
"""Property 9: Loop honors hooks/guardrails.

For any tool call blocked by a configured guardrail or pre-hook, the loop SHALL
NOT execute that call and SHALL record it as blocked, while non-blocked calls
execute normally. The loop continues to function correctly when some calls are
blocked.

**Validates: Requirements 3.5**
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.agent import Agent, ModelSpec, RunResult
from loomable.content import ModelCapabilities
from loomable.kernel.contracts import Tool
from loomable.kernel.guardrails import GuardrailHarness
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: a tool name (valid identifier, short)
tool_names_st = st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True)

# Strategy: generate a list of distinct tool names (at least 2 so we can split
# into blocked and allowed subsets)
distinct_tool_names_st = st.lists(
    tool_names_st, min_size=2, max_size=6, unique=True
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """A model provider that returns a scripted sequence of responses."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._call_index >= len(self._responses):
            return ModelResponse(content="fallback", tool_calls=[])
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp


class RecordingTool(Tool):
    """A tool that records invocations and returns a fixed result."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Tool {name}"
        self.invocations: list[dict[str, Any]] = []

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(args))
        return ToolResult(content=f"result-from-{self.name}")


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestLoopHonorsGuardrails:
    """Property 9: Loop honors hooks/guardrails."""

    @settings(max_examples=100, deadline=None)
    @given(
        tool_names=distinct_tool_names_st,
        blocked_fraction=st.floats(min_value=0.1, max_value=0.9),
    )
    @pytest.mark.asyncio
    async def test_guardrail_blocks_tools_and_allows_others(
        self,
        tool_names: list[str],
        blocked_fraction: float,
    ) -> None:
        """Tools blocked by a GuardrailHarness are never executed, while
        non-blocked tools in the same batch execute normally."""
        # Split tool_names into blocked and allowed subsets
        split_idx = max(1, int(len(tool_names) * blocked_fraction))
        split_idx = min(split_idx, len(tool_names) - 1)  # Ensure at least 1 allowed
        blocked_names = tool_names[:split_idx]
        allowed_names = tool_names[split_idx:]

        assume(len(blocked_names) >= 1)
        assume(len(allowed_names) >= 1)

        # Create recording tools for all names
        tools: dict[str, RecordingTool] = {
            name: RecordingTool(name) for name in tool_names
        }

        # Build model responses: one iteration with all tools called, then final
        call_counter = 0
        tool_calls = []
        for name in tool_names:
            tool_calls.append(
                ToolCall(
                    id=str(uuid.uuid4()),
                    tool_name=name,
                    args={"_idx": call_counter},
                )
            )
            call_counter += 1

        responses = [
            ModelResponse(content="", tool_calls=tool_calls),
            ModelResponse(content="done", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)

        # Configure a guardrail that blocks the blocked_names
        harness = GuardrailHarness([
            {
                "rule_id": "test-block-rule",
                "blocked_tools": blocked_names,
            }
        ])

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=list(tools.values()),
            harness=harness,
        )
        built = agent.build()
        built.max_tool_iterations = 5

        result = await built.arun("test input")

        # (1) Blocked tools were never invoked
        for name in blocked_names:
            assert len(tools[name].invocations) == 0, (
                f"Blocked tool '{name}' should not have been invoked"
            )

        # (2) Allowed tools were invoked exactly once
        for name in allowed_names:
            assert len(tools[name].invocations) == 1, (
                f"Allowed tool '{name}' should have been invoked once, "
                f"got {len(tools[name].invocations)}"
            )

        # (3) tool_activity only contains outcomes for allowed tools
        activity_call_ids = {o.call_id for o in result.tool_activity}
        # Map call_ids to tool names from our original tool_calls
        allowed_call_ids = {
            tc.id for tc in tool_calls if tc.tool_name in allowed_names
        }
        blocked_call_ids = {
            tc.id for tc in tool_calls if tc.tool_name in blocked_names
        }
        assert activity_call_ids == allowed_call_ids, (
            "tool_activity should contain exactly the allowed tool call ids"
        )
        # Blocked calls should not appear in tool_activity
        assert not (activity_call_ids & blocked_call_ids)

        # (4) The loop still terminated with a final answer
        assert isinstance(result, RunResult)
        assert result.output.text() == "done"

    @settings(max_examples=100, deadline=None)
    @given(
        tool_names=distinct_tool_names_st,
        blocked_fraction=st.floats(min_value=0.1, max_value=0.9),
    )
    @pytest.mark.asyncio
    async def test_pre_hook_blocks_tools_and_allows_others(
        self,
        tool_names: list[str],
        blocked_fraction: float,
    ) -> None:
        """Tools rejected by a pre-hook are not executed, while non-rejected
        tools in the same batch execute normally."""
        # Split tool_names into blocked and allowed subsets
        split_idx = max(1, int(len(tool_names) * blocked_fraction))
        split_idx = min(split_idx, len(tool_names) - 1)
        blocked_names = set(tool_names[:split_idx])
        allowed_names = set(tool_names[split_idx:])

        assume(len(blocked_names) >= 1)
        assume(len(allowed_names) >= 1)

        # Create recording tools for all names
        tools: dict[str, RecordingTool] = {
            name: RecordingTool(name) for name in tool_names
        }

        # Build model responses: one iteration with all tools called, then final
        call_counter = 0
        tool_calls = []
        for name in tool_names:
            tool_calls.append(
                ToolCall(
                    id=str(uuid.uuid4()),
                    tool_name=name,
                    args={"_idx": call_counter},
                )
            )
            call_counter += 1

        responses = [
            ModelResponse(content="", tool_calls=tool_calls),
            ModelResponse(content="done", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)

        # Pre-hook that blocks tools in blocked_names by returning False
        def blocking_hook(tool_name: str, call: ToolCall, args: dict) -> object:
            if tool_name in blocked_names:
                return False
            return True

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=list(tools.values()),
            tool_hooks=[blocking_hook],
        )
        built = agent.build()
        built.max_tool_iterations = 5

        result = await built.arun("test input")

        # (1) Blocked tools were never invoked
        for name in blocked_names:
            assert len(tools[name].invocations) == 0, (
                f"Hook-blocked tool '{name}' should not have been invoked"
            )

        # (2) Allowed tools were invoked exactly once
        for name in allowed_names:
            assert len(tools[name].invocations) == 1, (
                f"Allowed tool '{name}' should have been invoked once, "
                f"got {len(tools[name].invocations)}"
            )

        # (3) tool_activity only contains outcomes for allowed tools
        activity_call_ids = {o.call_id for o in result.tool_activity}
        allowed_call_ids = {
            tc.id for tc in tool_calls if tc.tool_name in allowed_names
        }
        blocked_call_ids = {
            tc.id for tc in tool_calls if tc.tool_name in blocked_names
        }
        assert activity_call_ids == allowed_call_ids
        assert not (activity_call_ids & blocked_call_ids)

        # (4) The loop still terminated
        assert isinstance(result, RunResult)
        assert result.output.text() == "done"

    @settings(max_examples=100, deadline=None)
    @given(
        tool_names=distinct_tool_names_st,
    )
    @pytest.mark.asyncio
    async def test_loop_continues_across_iterations_with_blocked_calls(
        self,
        tool_names: list[str],
    ) -> None:
        """When some calls are blocked in an iteration, the loop still continues
        to process subsequent iterations correctly."""
        assume(len(tool_names) >= 2)

        # Block the first tool name
        blocked_name = tool_names[0]
        allowed_names = tool_names[1:]

        tools: dict[str, RecordingTool] = {
            name: RecordingTool(name) for name in tool_names
        }

        # Iteration 1: all tools called (one blocked, rest allowed)
        iter1_calls = [
            ToolCall(
                id=str(uuid.uuid4()),
                tool_name=name,
                args={"iter": 1, "_idx": i},
            )
            for i, name in enumerate(tool_names)
        ]

        # Iteration 2: only allowed tools called (none blocked)
        iter2_calls = [
            ToolCall(
                id=str(uuid.uuid4()),
                tool_name=name,
                args={"iter": 2, "_idx": len(tool_names) + i},
            )
            for i, name in enumerate(allowed_names)
        ]

        responses = [
            ModelResponse(content="", tool_calls=iter1_calls),
            ModelResponse(content="", tool_calls=iter2_calls),
            ModelResponse(content="final answer", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)

        harness = GuardrailHarness([
            {
                "rule_id": "block-first",
                "blocked_tools": [blocked_name],
            }
        ])

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=list(tools.values()),
            harness=harness,
        )
        built = agent.build()
        built.max_tool_iterations = 10

        result = await built.arun("test input")

        # (1) Blocked tool was never invoked (even though requested in iter1)
        assert len(tools[blocked_name].invocations) == 0

        # (2) Allowed tools were invoked in both iterations
        for name in allowed_names:
            # Once in iter1 + once in iter2 = 2
            assert len(tools[name].invocations) == 2, (
                f"Allowed tool '{name}' should have been invoked twice "
                f"(once per iteration), got {len(tools[name].invocations)}"
            )

        # (3) The loop terminated with the final answer
        assert result.output.text() == "final answer"

        # (4) tool_activity should contain outcomes for allowed calls only
        # iter1: len(allowed_names) outcomes + iter2: len(allowed_names) outcomes
        expected_activity_count = len(allowed_names) * 2
        assert len(result.tool_activity) == expected_activity_count
