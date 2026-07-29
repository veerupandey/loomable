"""Unit tests for tool hooks and human-in-the-loop confirmation (task 9.2, Req 14).

Covers the gated tool-dispatch path on ``BuiltAgent``:
- A pre-hook that rejects a specific tool blocks it (not executed) and records the
  rejection, while other calls still run (Req 14.1/14.2/14.3).
- A confirmation-required tool executes only when the injectable approver grants
  approval; the default (headless) approver denies (Req 14.4).
- Post-hooks observe (and may transform) outcomes after execution (Req 14.1/14.2).

All fake tools are built via ``Agent(tools=[...]).build()`` so the tests exercise the
real builder + kernel ToolRuntime wiring, not a stub.
"""

from __future__ import annotations

from typing import Any

import pytest

from loomable.agent import (
    Agent,
    GatedDispatchResult,
    ModelSpec,
    ToolHookRejection,
)
from loomable.content import ModelCapabilities
from loomable.kernel.contracts import Tool
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolOutcome,
    ToolResult,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider (satisfies the structural protocol)."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


class RecordingTool(Tool):
    """A tool that records every invocation so tests can assert (non-)execution."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Records calls to {name}."
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content={"tool": self.name, **args})


def _agent_with_tools(tools: list[Tool], **kwargs: Any):
    """Build a BuiltAgent over the given tools using a fake provider."""
    return Agent(
        model=ModelSpec(provider="fake", provider_impl=_FakeProvider()),
        capabilities=ModelCapabilities(),
        tools=tools,
        **kwargs,
    ).build()


# ---------------------------------------------------------------------------
# Pre-hook rejection (Req 14.1, 14.2, 14.3)
# ---------------------------------------------------------------------------


class TestPreHookRejection:
    async def test_prehook_returning_false_blocks_only_that_tool(self) -> None:
        """A pre-hook rejecting one tool blocks it while siblings still execute."""
        danger = RecordingTool("danger")
        safe = RecordingTool("safe")

        def reject_danger(tool_name: str, call: ToolCall, args: dict) -> object:
            return False if tool_name == "danger" else True

        built = _agent_with_tools([danger, safe], tool_hooks=[reject_danger])

        result = await built.dispatch_tools_gated(
            [
                ToolCall(id="c1", tool_name="danger", args={"x": 1}),
                ToolCall(id="c2", tool_name="safe", args={"y": 2}),
            ]
        )

        assert isinstance(result, GatedDispatchResult)
        # danger never executed
        assert danger.calls == []
        # safe executed and produced an outcome
        assert safe.calls == [{"y": 2}]
        assert [o.call_id for o in result.outcomes] == ["c2"]
        assert result.outcomes[0].result is not None

        # The rejection was recorded as a guardrail violation for the danger tool.
        assert len(result.blocked) == 1
        assert result.blocked[0].action == "danger"
        assert result.blocked[0].rule_id == "tool-hook-rejection"

    async def test_prehook_raising_rejection_blocks_call(self) -> None:
        """A pre-hook that raises ToolHookRejection blocks the call."""
        danger = RecordingTool("danger")

        def reject(tool_name: str, call: ToolCall, args: dict) -> object:
            raise ToolHookRejection(tool_name, reason="not allowed")

        built = _agent_with_tools([danger], tool_hooks=[reject])

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="danger", args={})]
        )

        assert danger.calls == []
        assert result.outcomes == []
        assert len(result.blocked) == 1
        assert result.blocked[0].action == "danger"

    async def test_no_hooks_all_calls_execute(self) -> None:
        """With no hooks or confirmation, every call executes (parity with dispatch)."""
        a = RecordingTool("a")
        b = RecordingTool("b")
        built = _agent_with_tools([a, b])

        result = await built.dispatch_tools_gated(
            [
                ToolCall(id="c1", tool_name="a", args={}),
                ToolCall(id="c2", tool_name="b", args={}),
            ]
        )

        assert result.blocked == []
        assert [o.call_id for o in result.outcomes] == ["c1", "c2"]
        assert a.calls == [{}]
        assert b.calls == [{}]


# ---------------------------------------------------------------------------
# Confirmation gate (Req 14.4)
# ---------------------------------------------------------------------------


class TestConfirmationGate:
    async def test_default_approver_denies_confirmation_tool(self) -> None:
        """Default headless approver denies a confirmation-required tool."""
        risky = RecordingTool("risky")
        built = _agent_with_tools([risky], require_confirmation=["risky"])

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="risky", args={})]
        )

        assert risky.calls == []  # not executed
        assert result.outcomes == []
        assert len(result.blocked) == 1
        assert result.blocked[0].rule_id == "require-confirmation"
        assert result.blocked[0].action == "risky"

    async def test_approver_grant_allows_execution(self) -> None:
        """When the injected approver grants approval, the tool executes."""
        risky = RecordingTool("risky")
        built = _agent_with_tools([risky], require_confirmation=["risky"])
        built.approver = lambda call: True

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="risky", args={"n": 5})]
        )

        assert risky.calls == [{"n": 5}]
        assert result.blocked == []
        assert [o.call_id for o in result.outcomes] == ["c1"]
        assert result.outcomes[0].result is not None

    async def test_approver_can_decide_per_call(self) -> None:
        """The approver may grant some confirmation-required calls and deny others."""
        risky = RecordingTool("risky")
        built = _agent_with_tools([risky], require_confirmation=["risky"])
        # Approve only when args request it.
        built.approver = lambda call: bool(call.args.get("ok"))

        result = await built.dispatch_tools_gated(
            [
                ToolCall(id="yes", tool_name="risky", args={"ok": True}),
                ToolCall(id="no", tool_name="risky", args={"ok": False}),
            ]
        )

        assert risky.calls == [{"ok": True}]
        assert [o.call_id for o in result.outcomes] == ["yes"]
        assert [v.action for v in result.blocked] == ["risky"]

    async def test_non_confirmation_tool_runs_without_approval(self) -> None:
        """Tools not in require_confirmation execute without consulting the approver."""
        free = RecordingTool("free")
        built = _agent_with_tools([free], require_confirmation=["other"])
        # Approver would deny everything, but 'free' is not gated.
        built.approver = lambda call: False

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="free", args={})]
        )

        assert free.calls == [{}]
        assert result.blocked == []
        assert len(result.outcomes) == 1


# ---------------------------------------------------------------------------
# Post-hooks (Req 14.1, 14.2)
# ---------------------------------------------------------------------------


class TestPostHooks:
    async def test_post_hook_observes_outcomes(self) -> None:
        """Post-hooks run after execution and observe each outcome."""
        tool = RecordingTool("t")
        built = _agent_with_tools([tool])

        observed: list[tuple[str, str]] = []

        def observer(tool_name: str, call: ToolCall, outcome: ToolOutcome) -> object:
            observed.append((tool_name, outcome.call_id))
            return None  # observe only

        built.post_tool_hooks = [observer]

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="t", args={})]
        )

        assert observed == [("t", "c1")]
        # Outcome unchanged when the post-hook only observes.
        assert result.outcomes[0].result is not None

    async def test_post_hook_can_transform_outcome(self) -> None:
        """A post-hook returning a ToolOutcome replaces the original."""
        tool = RecordingTool("t")
        built = _agent_with_tools([tool])

        def transformer(
            tool_name: str, call: ToolCall, outcome: ToolOutcome
        ) -> ToolOutcome:
            return ToolOutcome(
                call_id=outcome.call_id,
                result=ToolResult(content={"transformed": True}),
            )

        built.post_tool_hooks = [transformer]

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="t", args={})]
        )

        assert result.outcomes[0].result is not None
        assert result.outcomes[0].result.content == {"transformed": True}

    async def test_post_hooks_only_run_on_executed_calls(self) -> None:
        """Post-hooks see only executed calls, not blocked ones."""
        allowed = RecordingTool("allowed")
        blocked_tool = RecordingTool("blocked")

        def reject_blocked(tool_name: str, call: ToolCall, args: dict) -> object:
            return tool_name != "blocked"

        built = _agent_with_tools(
            [allowed, blocked_tool], tool_hooks=[reject_blocked]
        )

        seen: list[str] = []
        built.post_tool_hooks = [
            lambda tn, call, outcome: seen.append(outcome.call_id)
        ]

        result = await built.dispatch_tools_gated(
            [
                ToolCall(id="c1", tool_name="allowed", args={}),
                ToolCall(id="c2", tool_name="blocked", args={}),
            ]
        )

        assert seen == ["c1"]
        assert [o.call_id for o in result.outcomes] == ["c1"]
        assert [v.action for v in result.blocked] == ["blocked"]
