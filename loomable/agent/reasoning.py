"""loomable.agent.reasoning - Scratchpad and plan-escalation tools.

Provides:

- :func:`make_think_tool` — zero-side-effect scratchpad
- :func:`make_plan_tool` — runtime plan→map→synthesize escalation
- :func:`execute_plan_flow` — shared plan pipeline used by BuiltAgent and the plan tool
- :func:`coalesce_map_pieces` / :func:`format_worker_notes` — synthesizer helpers
"""

from __future__ import annotations

__all__ = [
    "make_think_tool",
    "make_plan_tool",
    "coalesce_map_pieces",
    "format_worker_notes",
    "parse_plan_steps",
    "execute_plan_flow",
]

import json
from typing import TYPE_CHECKING, Any

from .tools import FunctionTool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from loomable.agent.context import RunContext
    from loomable.agent.run import RunResult

    from .builder import BuiltAgent


def coalesce_map_pieces(input: Any, context: Any | None = None) -> list[Any]:
    """Extract mapped worker outputs for a synthesizer node.

    Prefers, in order:
    1. A list passed directly as ``input`` (SequentialEngine state["map"])
    2. ``input["map"]`` when ``input`` is a dict
    3. ``context.shared_state["map"]`` when available
    """
    if isinstance(input, list):
        return input
    if isinstance(input, dict):
        pieces = input.get("map")
        if isinstance(pieces, list):
            return pieces
    if context is not None:
        shared = getattr(context, "shared_state", None)
        if shared is not None:
            pieces = shared.get("map")
            if isinstance(pieces, list):
                return pieces
    return []


def format_worker_notes(
    pieces: list[Any],
    steps: list[str] | None = None,
) -> str:
    """Pair plan steps with worker results for the synthesizer."""
    if not pieces and not steps:
        return "(no worker notes)"
    lines: list[str] = []
    count = max(len(pieces), len(steps or []))
    for i in range(count):
        step = (steps or [])[i] if steps and i < len(steps) else f"Step {i + 1}"
        result = pieces[i] if i < len(pieces) else "(missing)"
        lines.append(f"Step {i + 1} — {step}\n{result}")
    return "\n\n".join(lines)


def parse_plan_steps(text: str, *, max_steps: int = 5) -> list[str]:
    """Parse a model plan response into a clean list of step strings."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json") :].strip()

    steps: list[Any]
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and isinstance(data.get("plan_steps"), list):
            steps = data["plan_steps"]
        elif isinstance(data, list):
            steps = data
        else:
            steps = [cleaned]
    except (ValueError, json.JSONDecodeError):
        steps = [
            line.strip().lstrip("-*•0123456789.) ")
            for line in cleaned.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    out = [str(s).strip() for s in steps if str(s).strip()]
    return out[:max_steps]


def planner_prompt(task: str, *, max_steps: int = 5) -> str:
    return (
        "You are a planner for parallel subagents.\n"
        f"Decompose the task into at most {max_steps} steps that can run "
        "independently in parallel (different facets, entities, or workstreams — "
        "not sequential draft revisions of the same answer).\n"
        "Each step must be self-contained: a worker will see the overall task "
        "plus only that step.\n"
        "If the task is really one cohesive write/answer, return a single-element array.\n"
        "Return ONLY a JSON array of short imperative strings.\n"
        "No prose, no markdown, no code fences.\n\n"
        f"Task: {task}"
    )


def worker_prompt(task: str, step: str) -> str:
    return (
        "You are one specialist subagent. Other workers handle other steps in "
        "parallel; do not cover their work and do not write the final combined answer.\n\n"
        f"Overall task (context only):\n{task}\n\n"
        f"Your assigned step (do only this):\n{step}\n\n"
        "Return a self-contained finding: concrete, concise, usable by a synthesizer.\n"
        "No meta commentary about being a step or subagent."
    )


def synthesizer_prompt(task: str, notes: str) -> str:
    return (
        "Answer the user's original task directly for a busy executive reader.\n\n"
        f"Original task:\n{task}\n\n"
        f"Research notes from parallel workers (internal; do not mention them):\n{notes}\n\n"
        "Write one plain-language final answer that fulfills the original task.\n"
        "Rules:\n"
        "- Lead with the answer; no preamble about steps or workers.\n"
        "- Do not list or label worker outputs unless the user asked for a breakdown.\n"
        "- Short sentences, plain English, no jargon unless the task requires it.\n"
        "- Do not invent facts beyond the notes and the task."
    )


async def execute_plan_flow(
    agent: "BuiltAgent",
    task_text: str,
    *,
    max_steps: int = 5,
    output_schema: type | None = None,
    ctx: "RunContext | None" = None,
    plan_trigger: str = "router",
    allow_tools: bool = True,
) -> "RunResult":
    """Run plan → parallel workers → synthesize and return a RunResult.

    Shared by :meth:`BuiltAgent._run_plan` and :func:`make_plan_tool`.
    """
    from loomable.content import AgentInput, AgentOutput, Text
    from loomable.flow.helpers import plan_and_execute

    from .run import RunResult

    plan_steps: list[str] = []

    async def _run_worker_model(prompt: str) -> str:
        agent_input = AgentInput.from_text(prompt)
        if allow_tools and bool(getattr(agent.tool_runtime, "_tools", {})):
            # Prefer tool loop when the agent has tools, so plan workers stay capable.
            result = await agent._run_tool_loop(
                agent_input, output_schema=None, include_history=False, ctx=ctx
            )
        else:
            result = await agent._run_single(
                agent_input, include_history=False, ctx=ctx
            )
        return result.output.text()

    async def _worker(input: Any, **kwargs: Any) -> str:
        step = input if isinstance(input, str) else str(input)
        return await _run_worker_model(worker_prompt(task_text, step))

    async def _synthesizer(input: Any, **kwargs: Any) -> str:
        context = kwargs.get("context")
        pieces = coalesce_map_pieces(input, context)
        steps = list(plan_steps)
        if context is not None and getattr(context, "shared_state", None) is not None:
            raw_steps = context.shared_state.get("plan_steps")
            if isinstance(raw_steps, list) and raw_steps:
                steps = [str(s) for s in raw_steps]
        notes = format_worker_notes(pieces, steps)
        result = await agent._run_single(
            AgentInput.from_text(synthesizer_prompt(task_text, notes)),
            output_schema=output_schema,
            include_history=False,
            ctx=ctx,
        )
        return result.output.text()

    # Empty-plan guard: fall back to a normal single-shot answer.
    probe = await agent._run_single(
        AgentInput.from_text(planner_prompt(task_text, max_steps=max_steps)),
        include_history=False,
        ctx=ctx,
    )
    plan_steps = parse_plan_steps(probe.output.text(), max_steps=max_steps)
    if not plan_steps:
        fallback = await agent._run_single(
            AgentInput.from_text(task_text),
            output_schema=output_schema,
            include_history=False,
            ctx=ctx,
        )
        fallback.metadata["run_strategy"] = "single"
        fallback.metadata["plan_fallback"] = "empty_plan"
        fallback.metadata["plan_trigger"] = plan_trigger
        fallback.metadata["plan_steps"] = []
        fallback.metadata["plan_workers"] = 0
        return fallback

    # Reuse parsed steps — planner node should not pay a second model call.
    cached_steps = list(plan_steps)

    async def _planner_cached(input: Any, **kwargs: Any) -> dict:
        nonlocal plan_steps
        plan_steps = list(cached_steps)
        return {"plan_steps": plan_steps}

    flow = plan_and_execute(
        planner=_planner_cached,
        workers=_worker,
        synthesizer=_synthesizer,
        session_id=agent.session.session_id,
    )
    flow_result = await flow.arun(AgentInput.from_text(task_text))

    output_text = flow_result.output.text() if flow_result.output else ""

    plan_workers = 0
    map_failed = 0
    map_result = (flow_result.sub_results or {}).get("map")
    if map_result is not None and getattr(map_result, "metadata", None):
        outputs = map_result.metadata.get("map_outputs")
        if isinstance(outputs, list):
            plan_workers = len(outputs)
        else:
            plan_workers = int(map_result.metadata.get("map_total") or 0)
        map_failed = int(map_result.metadata.get("map_failed") or 0)

    return RunResult(
        output=AgentOutput(parts=[Text(output_text)]),
        session_id=agent.session.session_id,
        usage=flow_result.usage,
        tool_activity=[],
        structured=None,
        metadata={
            "run_strategy": "plan",
            "plan_trigger": plan_trigger,
            "plan_steps": list(cached_steps),
            "plan_workers": plan_workers,
            "map_failed": map_failed,
        },
    )


def make_think_tool() -> FunctionTool:
    """Create a no-side-effect scratchpad tool."""

    def think(thought: str) -> str:
        """A scratchpad for intermediate reasoning. Returns the thought unchanged."""
        return thought

    return FunctionTool(
        think,
        name="think",
        description="A scratchpad for intermediate reasoning. Returns the thought unchanged.",
        idempotent=True,
    )


def make_plan_tool(agent: "BuiltAgent") -> FunctionTool:
    """Create a tool that escalates to plan-and-fan-out via :func:`execute_plan_flow`."""

    async def plan(task: str, max_steps: int = 5) -> str:
        """Decompose a complex task into parallel steps, execute them, and synthesize."""
        result = await execute_plan_flow(
            agent,
            task,
            max_steps=max_steps,
            plan_trigger="tool",
            allow_tools=True,
        )
        return result.output.text()

    return FunctionTool(
        plan,
        name="plan",
        description=(
            "Decompose a complex task into parallel steps, execute them, "
            "and synthesize the results."
        ),
        idempotent=True,
    )
