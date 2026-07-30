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
    internally (Req 17.2), replacing the removed ``AutoPlan`` class. The planner,
    worker, and synthesizer are all backed by the agent's single-shot path so the
    agent's session/tools/knowledge remain available.

    This lets the model escalate a simple loop into a dynamic graph on demand
    without requiring a separate graph engine.
    """

    async def plan(task: str, max_steps: int = 5) -> str:
        """Decompose a complex task into parallel steps, execute them, and synthesize the results."""
        import json as _json

        from loomable.content import AgentInput
        from loomable.flow.helpers import plan_and_execute

        from .builder import _input_text

        async def _planner(input: Any, **kwargs: Any) -> dict:
            """Ask the model for a concise plan."""
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
            text = result.output.text().strip()
            # Strip code fences if present.
            if text.startswith("```"):
                text = text.split("\n", 1)[-1] if "\n" in text else text
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json"):
                    text = text[len("json"):].strip()
            try:
                steps = _json.loads(text)
                if not isinstance(steps, list):
                    steps = [text]
            except (ValueError, _json.JSONDecodeError):
                steps = [
                    line.strip().lstrip("-*•0123456789.) ")
                    for line in text.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            return {"plan_steps": steps[:max_steps]}

        async def _worker(input: Any, **kwargs: Any) -> str:
            """Run a single plan step."""
            step = input if isinstance(input, str) else str(input)
            prompt = (
                f"Overall task:\n{task}\n\n"
                f"Complete ONLY this step, concisely and concretely:\n{step}"
            )
            result = await agent._run_single(
                AgentInput.from_text(prompt), include_history=False
            )
            return result.output.text()

        async def _synthesizer(input: Any, **kwargs: Any) -> str:
            """Combine step results into a final answer."""
            state_data = input if isinstance(input, dict) else {}
            pieces = state_data.get("map", []) or []
            combined = "\n".join(f"- {p}" for p in pieces) if pieces else str(input)
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
        return flow_result.output.text()

    return FunctionTool(plan, name="plan", description="Decompose a complex task into parallel steps, execute them, and synthesize the results.", idempotent=True)
