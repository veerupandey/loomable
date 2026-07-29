"""loomable.agent.reasoning - Scratchpad and plan-escalation tools.

Provides two reasoning tools that extend the agent's in-context capabilities:

- :func:`make_think_tool` — a zero-side-effect scratchpad that echoes the model's
  thought back into context, improving policy adherence over long tool chains.
- :func:`make_plan_tool` — exposes :class:`~loomable.agent.autoplan.AutoPlan` as a
  callable tool so the model can escalate a simple loop into a dynamic fan-out on
  demand ("dynamic graphs without a graph engine").
"""

from __future__ import annotations

__all__ = ["make_think_tool", "make_plan_tool"]

from typing import TYPE_CHECKING

from .tools import FunctionTool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .builder import BuiltAgent


def make_think_tool() -> FunctionTool:
    """Create a no-side-effect scratchpad tool.

    The tool signature is ``think(thought: str) -> str``. It returns the thought
    straight back so it re-enters context; no memory changes, no control-flow
    changes. Marked ``idempotent=True``.

    This mirrors the Anthropic "think" tool / agno ThinkingTools pattern: giving
    the model an explicit place to reason improves accuracy on multi-step tasks.
    """

    def think(thought: str) -> str:
        """A scratchpad for intermediate reasoning. Returns the thought unchanged."""
        return thought

    return FunctionTool(think, name="think", description="A scratchpad for intermediate reasoning. Returns the thought unchanged.", idempotent=True)


def make_plan_tool(agent: "BuiltAgent") -> FunctionTool:
    """Create a tool that escalates to plan-and-fan-out at runtime.

    The tool signature is ``plan(task: str, max_steps: int = 5) -> str``. It
    invokes :class:`~loomable.agent.autoplan.AutoPlan` internally — plan → parallel
    subagents (via the kernel SubagentManager) → synthesize — and returns the
    synthesized answer as the tool result.

    This lets the model escalate a simple loop into a dynamic graph on demand
    without requiring a separate graph engine.
    """

    async def plan(task: str, max_steps: int = 5) -> str:
        """Decompose a complex task into parallel steps, execute them, and synthesize the results."""
        from .autoplan import AutoPlan

        from loomable.content import AgentInput

        result = await AutoPlan(agent, max_steps=max_steps).run(
            AgentInput.from_text(task)
        )
        return result.output.text()

    return FunctionTool(plan, name="plan", description="Decompose a complex task into parallel steps, execute them, and synthesize the results.", idempotent=True)
