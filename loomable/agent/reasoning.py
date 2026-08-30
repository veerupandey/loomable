"""loomable.agent.reasoning - Scratchpad and plan-escalation tools.

Provides two reasoning tools that extend the agent's in-context capabilities:

- :func:`make_think_tool` — a zero-side-effect scratchpad that echoes the model's
  thought back into context, improving policy adherence over long tool chains.
- :func:`make_plan_tool` — builds a plan→map→synthesize Flow via
  :func:`~loomable.flow.helpers.plan_and_execute` so the model can escalate a
  simple loop into a dynamic fan-out on demand (Req 17.2).
"""

from __future__ import annotations

__all__ = ["make_think_tool", "make_plan_tool"]

from typing import TYPE_CHECKING, Any

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
    """Create a tool that escalates to plan-and-fan-out at runtime via the Flow engine.

    The tool signature is ``plan(task: str, max_steps: int = 5) -> str``. It
    builds a plan→map→synthesize Flow using :func:`~loomable.flow.helpers.plan_and_execute`
    internally (Req 17.2). Planning uses the kernel :class:`~loomable.kernel.planner.Planner`
    when set on the agent; workers use the full tool loop when tools are registered
    (parity with the complexity-router PLAN path).
    """

    async def plan(task: str, max_steps: int = 5) -> Any:
        """Decompose a complex task into parallel steps, execute them, and synthesize the results."""
        from loomable.content import AgentInput
        from loomable.flow.helpers import plan_and_execute
        from loomable.kernel.models import ToolResult

        plan_steps: list[str] = []

        async def _planner(input: Any, **kwargs: Any) -> dict:
            """Produce plan steps — kernel Planner when set, else JSON prompt."""
            nonlocal plan_steps
            if getattr(agent, "planner", None) is not None:
                from loomable.kernel.planner import TaskContext

                exec_plan = await agent.planner.plan(TaskContext(task=task))
                steps = [str(s).strip() for s in exec_plan.steps if str(s).strip()]
                plan_steps = steps[:max_steps] or [task]
                return {"plan_steps": plan_steps}

            from loomable.plan_parse import parse_plan_steps

            plan_prompt = (
                f"You are a planner. Break the user's task into at most {max_steps} "
                "concrete, independent, actionable steps. Return ONLY a JSON array of "
                "short imperative step strings (e.g. [\"Do X\", \"Do Y\"]). "
                "No prose, no markdown, no code fences.\n\n"
                f"Task: {task}"
            )
            result = await agent._run_single(
                AgentInput.from_text(plan_prompt), include_history=False
            )
            steps = parse_plan_steps(result.output.text(), max_steps=max_steps)
            plan_steps = steps
            return {"plan_steps": plan_steps}

        async def _worker(input: Any, **kwargs: Any) -> str:
            """Run a single plan step with the agent's tool loop when tools exist."""
            step = input if isinstance(input, str) else str(input)
            prompt = (
                f"Overall task:\n{task}\n\n"
                f"Complete ONLY this step, concisely and concretely:\n{step}"
            )
            step_input = AgentInput.from_text(prompt)
            if agent.tool_runtime._tools:
                result = await agent._run_tool_loop(
                    step_input,
                    include_history=False,
                    exclude_tools=frozenset({"plan"}),
                )
            else:
                result = await agent._run_single(step_input, include_history=False)
            return result.output.text()

        async def _synthesizer(input: Any, *, context: Any = None, **kwargs: Any) -> str:
            """Combine step results into a final answer."""
            pieces: list[Any] = []
            if context is not None and getattr(context, "shared_state", None) is not None:
                raw = context.shared_state.get("map")
                if isinstance(raw, list):
                    pieces = raw
            if not pieces and isinstance(input, dict):
                pieces = input.get("map", []) or []
            if not pieces:
                pieces = [str(input)]
            combined = "\n".join(f"- {p}" for p in pieces)
            prompt = (
                f"Original task:\n{task}\n\n"
                f"Results from the planned steps:\n{combined}\n\n"
                "Integrate these into one cohesive, well-structured final answer."
            )
            result = await agent._run_single(
                AgentInput.from_text(prompt), include_history=False
            )
            return result.output.text()

        flow = plan_and_execute(
            planner=_planner,
            workers=_worker,
            synthesizer=_synthesizer,
            session_id=agent.session.session_id,
        )
        flow_result = await flow.arun(AgentInput.from_text(task))
        return ToolResult(
            content=flow_result.output.text(),
            metadata={"plan_steps": plan_steps},
        )

    return FunctionTool(
        plan,
        name="plan",
        description="Decompose a complex task into parallel steps, execute them, and synthesize the results.",
        idempotent=False,
    )
