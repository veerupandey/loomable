"""loomable.agent.autoplan - Autonomous plan -> parallel subagents -> synthesize.

:class:`AutoPlan` lets a *single* agent decompose a task on its own: it asks its
model for a concise plan, runs each step concurrently as an internal subagent
(reusing the kernel :class:`~loomable.kernel.subagents.SubagentManager` for
concurrency + fault isolation), and synthesizes the step results into one answer —
all through the same :class:`~loomable.agent.builder.BuiltAgent`, so the agent's own
session/memory captures the task and final answer (memory stays centralized on the
one agent).

This is wired to :attr:`OrchestrationMode.PLAN`: with that mode, ``agent.arun(task)``
plans and parallelizes automatically — no pre-supplied ``sub_agents`` needed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loomable.content import AgentInput
from loomable.kernel.models import ModelRequest
from loomable.kernel.subagents import DelegatedTask, SubagentManager

from .builder import _input_text
from .run import RunResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .builder import BuiltAgent


def _parse_steps(text: str, max_steps: int) -> list[str]:
    """Parse a plan response into a clean list of step strings.

    Prefers a JSON array of strings (what the planning prompt asks for). Falls back
    to line parsing that strips markdown headers, list bullets, and numbering, and
    drops empty or heading-only lines. The result is capped to ``max_steps``.
    """
    cleaned = text.strip()
    # Strip a leading/trailing ``` or ```json code fence if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json") :].strip()

    # Try JSON array of strings first.
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            steps = [str(item).strip() for item in data if str(item).strip()]
            if steps:
                return steps[:max_steps]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: line-based parsing.
    steps: list[str] = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip common bullet/number prefixes: "1.", "1)", "-", "*", "•".
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
        # Numbered: "12. text" or "12) text".
        head, sep, tail = line.partition(".")
        if sep and head.strip().isdigit():
            line = tail.strip()
        else:
            head, sep, tail = line.partition(")")
            if sep and head.strip().isdigit():
                line = tail.strip()
        # Skip lines that are pure headings (end with ':' and have no detail).
        if not line or (line.endswith(":") and len(line.split()) <= 4):
            continue
        steps.append(line)
    return steps[:max_steps]


class AutoPlan:
    """Drives autonomous plan -> parallel subagents -> synthesize for one agent."""

    #: The planning instruction — asks for a clean JSON array of concise steps.
    _PLAN_SYSTEM = (
        "You are a planner. Break the user's task into at most {n} concrete, "
        "independent, actionable steps. Return ONLY a JSON array of short imperative "
        "step strings (e.g. [\"Do X\", \"Do Y\"]). No prose, no markdown, no code fences."
    )

    def __init__(self, agent: "BuiltAgent", *, max_steps: int = 5) -> None:
        self._agent = agent
        self._max_steps = max_steps

    async def run(
        self, agent_input: AgentInput, *, output_schema: type | None = None
    ) -> RunResult:
        """Plan the task, run steps concurrently, then synthesize a final answer."""
        task = _input_text(agent_input)

        steps = await self._plan(task)
        if not steps:
            # No decomposition produced — fall back to a single direct run.
            return await self._agent._run_single(agent_input, output_schema=output_schema)

        outcomes = await self._delegate(task, steps)

        sub_results: dict[str, object] = {}
        pieces: list[tuple[str, str]] = []
        usage: dict[str, int] = {}
        for outcome in outcomes:
            if outcome.result is not None:
                sub_results[outcome.task_id] = outcome.result
                pieces.append((outcome.task_id, outcome.result.output.text()))
                for key, value in (outcome.result.usage or {}).items():
                    usage[key] = usage.get(key, 0) + value
            else:
                sub_results[outcome.task_id] = outcome.error

        synth = await self._synthesize(task, pieces, output_schema)
        for key, value in (synth.usage or {}).items():
            usage[key] = usage.get(key, 0) + value

        return RunResult(
            output=synth.output,
            session_id=self._agent.session.session_id,
            usage=usage,
            tool_activity=[],
            sub_results=sub_results,
            structured=synth.structured,
        )

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    async def _plan(self, task: str) -> list[str]:
        """Ask the model for a clean list of steps and parse it robustly."""
        messages: list[dict] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self._PLAN_SYSTEM.format(n=self._max_steps)}
                ],
            }
        ]
        # Include the agent's recent conversation memory so follow-up tasks (e.g.
        # "summarize that") are planned with context (Req 15).
        messages.extend(self._agent._memory_prefix())
        messages.append({"role": "user", "content": [{"type": "text", "text": task}]})

        request = ModelRequest(messages=messages, temperature=0.2, max_tokens=400)
        response = await self._agent.model_interface.invoke(request)
        return _parse_steps(response.content, self._max_steps)

    async def _delegate(self, task: str, steps: list[str]):
        """Run each step as a concurrent internal subagent (fault-isolated)."""
        tasks = [
            DelegatedTask(
                task_id=f"step-{i + 1}",
                task=step,
                context={},
                agent_factory=(lambda s=step: self._run_step(task, s)),
            )
            for i, step in enumerate(steps)
        ]
        return await SubagentManager().run_all(tasks)

    async def _run_step(self, task: str, step: str) -> RunResult:
        """Run a single plan step through the same agent's single-run path."""
        prompt = (
            f"Overall task:\n{task}\n\n"
            f"Complete ONLY this step, concisely and concretely:\n{step}"
        )
        # Focused steps are self-contained: they carry the overall task explicitly and
        # skip conversation history to stay on-task and cheap.
        return await self._agent._run_single(
            AgentInput.from_text(prompt), include_history=False
        )

    async def _synthesize(
        self, task: str, pieces: list[tuple[str, str]], output_schema: type | None
    ) -> RunResult:
        """Merge the step results into one cohesive final answer."""
        combined = "\n".join(f"- {sid}: {text}" for sid, text in pieces)
        prompt = (
            f"Original task:\n{task}\n\n"
            f"Results from the planned steps:\n{combined}\n\n"
            "Integrate these into one cohesive, well-structured final answer."
        )
        return await self._agent._run_single(
            AgentInput.from_text(prompt), output_schema=output_schema
        )
