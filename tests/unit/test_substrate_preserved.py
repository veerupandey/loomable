"""Confirm the thinking tool and subagent primitive remain available and unchanged.

Task 15.3: verify that the consolidation (removal of Pipeline/Orchestrator/AutoPlan)
did NOT touch the kernel SubagentManager or the reasoning tools — they remain available
as substrate for parallel and map execution in the flow engine.

Validates: Requirements 17.1, 17.4
"""

from __future__ import annotations

import inspect

from loomable.agent import make_think_tool, make_plan_tool
from loomable.agent.tools import FunctionTool
from loomable.kernel.subagents import DelegatedTask, SubagentManager, SubagentOutcome


# ---------------------------------------------------------------------------
# Req 17.1 — Thinking tool remains available
# ---------------------------------------------------------------------------


class TestThinkToolPreserved:
    """make_think_tool() produces a usable FunctionTool named 'think'."""

    def test_importable_and_returns_function_tool(self) -> None:
        tool = make_think_tool()
        assert isinstance(tool, FunctionTool)

    def test_tool_name_is_think(self) -> None:
        tool = make_think_tool()
        assert tool.name == "think"

    def test_tool_is_idempotent(self) -> None:
        tool = make_think_tool()
        assert tool.idempotent is True

    def test_think_tool_has_description(self) -> None:
        tool = make_think_tool()
        assert tool.description  # non-empty


# ---------------------------------------------------------------------------
# Req 17.4 — SubagentManager remains available and unchanged as substrate
# ---------------------------------------------------------------------------


class TestSubagentManagerPreserved:
    """SubagentManager is importable with its spawn/run_all interface intact."""

    def test_importable(self) -> None:
        assert SubagentManager is not None

    def test_has_spawn_method(self) -> None:
        assert hasattr(SubagentManager, "spawn")
        assert inspect.iscoroutinefunction(SubagentManager.spawn)

    def test_has_run_all_method(self) -> None:
        assert hasattr(SubagentManager, "run_all")
        assert inspect.iscoroutinefunction(SubagentManager.run_all)

    def test_delegated_task_importable(self) -> None:
        assert DelegatedTask is not None
        # Verify it has the expected fields
        assert "task_id" in DelegatedTask.__dataclass_fields__
        assert "task" in DelegatedTask.__dataclass_fields__
        assert "context" in DelegatedTask.__dataclass_fields__
        assert "agent_factory" in DelegatedTask.__dataclass_fields__

    def test_subagent_outcome_importable(self) -> None:
        assert SubagentOutcome is not None
        # Verify it has the expected fields
        assert "task_id" in SubagentOutcome.__dataclass_fields__
        assert "result" in SubagentOutcome.__dataclass_fields__
        assert "error" in SubagentOutcome.__dataclass_fields__


# ---------------------------------------------------------------------------
# Req 17.2 — make_plan_tool remains importable
# ---------------------------------------------------------------------------


class TestPlanToolPreserved:
    """make_plan_tool is importable from loomable.agent."""

    def test_importable(self) -> None:
        assert callable(make_plan_tool)

    def test_is_defined_in_reasoning_module(self) -> None:
        from loomable.agent.reasoning import make_plan_tool as direct_import

        assert direct_import is make_plan_tool
