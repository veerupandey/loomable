"""Tough-task mode — plan → fan-out → act → verify for hard problems.

Easy surface for agents that must decompose work, spawn/map specialists,
and loop until a verifier passes — without forcing users onto low-level
Flow / Edge / MapNode APIs.

Happy path::

    from loomable import ToughTask, Agent

    task = ToughTask(
        model=provider,
        fan_out="spawn",          # or "map" (reuse one worker)
        verify=lambda out, ctx: "SEV-" in out.text(),
        max_iterations=3,
    )
    result = await task.arun("Handle INC-88421 end-to-end")

    # Same idea as a Workflow (graph engineering):
    wf = plan_act_verify(model=provider, until=my_verifier)
    result = await wf.arun("...")

    # Or force an Agent into tough mode:
    agent = Agent(model=provider, mode="tough", verifier=my_verifier)
"""

from __future__ import annotations

__all__ = [
    "ToughTask",
    "plan_act_verify",
    "map_specialists",
    "parse_plan_steps",
]

import asyncio
import json
import re
from typing import Any, Callable, Literal, Sequence

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, Text
from loomable.flow.loop import CallableVerifier, Loop, VerdictResult, Verifier
from loomable.flow.workflow import Workflow

FanOut = Literal["map", "spawn"]


def parse_plan_steps(text: str, *, max_steps: int = 5) -> list[str]:
    """Parse a planner response into a clean list of step strings."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "plan_steps" in data:
            data = data["plan_steps"]
        if isinstance(data, list):
            steps = [str(s).strip() for s in data if str(s).strip()]
            return steps[:max_steps] or [cleaned or "Complete the task"]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    steps: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line:
            steps.append(line)
    return steps[:max_steps] or [cleaned or "Complete the task"]


async def map_specialists(
    steps: Sequence[str],
    *,
    model: Any,
    role: str = "Specialist",
    goal: str = "",
    instructions: str | None = None,
    tools: list[Any] | None = None,
    modalities: str | None = None,
    concurrency: int | None = None,
) -> list[str]:
    """Fan-out: spawn one ephemeral specialist per plan step (parallel)."""
    from loomable.agent.delegation import spawn_specialist

    sem = asyncio.Semaphore(concurrency) if concurrency else None

    async def _one(idx: int, step: str) -> str:
        async def _run() -> str:
            return await spawn_specialist(
                model=model,
                role=f"{role} {idx + 1}",
                goal=goal or f"Execute assigned specialist work as {role}",
                task=step,
                instructions=instructions,
                tools=tools,
                modalities=modalities,
            )

        if sem is None:
            return await _run()
        async with sem:
            return await _run()

    return list(await asyncio.gather(*[_one(i, s) for i, s in enumerate(steps)]))


def _as_verifier(value: Any) -> Verifier | None:
    if value is None:
        return None
    if isinstance(value, Verifier):
        return value
    if callable(value):
        return CallableVerifier(value)
    raise TypeError(f"verifier must be Verifier or callable, got {type(value)!r}")


def _default_agent(
    model: Any,
    *,
    role: str,
    goal: str,
    instructions: str,
    tools: list[Any] | None = None,
    modalities: str | None = None,
) -> Any:
    from loomable.agent.builder import Agent

    kwargs: dict[str, Any] = {
        "model": model,
        "role": role,
        "goal": goal,
        "instructions": instructions,
        "tools": tools or [],
        "modalities": modalities or "text",
    }
    return Agent(**kwargs)


def plan_act_verify(
    planner: Any | None = None,
    worker: Any | None = None,
    synthesizer: Any | None = None,
    *,
    model: Any | None = None,
    verifier: Any | None = None,
    until: Any | None = None,
    max_iterations: int = 3,
    max_steps: int = 5,
    fan_out: FanOut = "map",
    name: str = "tough",
    session_id: str | None = None,
    checkpointer: Any = None,
    tools: list[Any] | None = None,
    modalities: str | None = None,
    concurrency: int | None = None,
) -> Workflow:
    """Build a Workflow: plan → fan-out act → synthesize → optional verify loop.

    Parameters
    ----------
    planner / worker / synthesizer:
        Agents or callables. If omitted, ``model`` builds sensible defaults.
    verifier / until:
        Same Verifier protocol used by Agent and Loop (callable OK).
    fan_out:
        ``"map"`` — reuse ``worker`` over plan steps (MapNode).
        ``"spawn"`` — ephemeral ``spawn_specialist`` per step.
    """
    if model is None and (planner is None or worker is None):
        raise ValueError("plan_act_verify requires model= or explicit planner/worker")

    verify = _as_verifier(until if until is not None else verifier)

    if planner is None:
        planner = _default_agent(
            model,
            role="Planner",
            goal="Decompose hard tasks into concrete steps",
            instructions=(
                f"Break the user task into at most {max_steps} concrete, "
                "independently executable steps. Return ONLY a JSON array of "
                "short imperative strings. No prose, no markdown."
            ),
            modalities=modalities,
        )
    if worker is None and fan_out == "map":
        worker = _default_agent(
            model,
            role="Worker",
            goal="Execute one plan step thoroughly",
            instructions="Complete ONLY the assigned step. Be concrete and concise.",
            tools=tools,
            modalities=modalities,
        )
    if synthesizer is None:
        synthesizer = _default_agent(
            model,
            role="Synthesizer",
            goal="Integrate step results into one final answer",
            instructions=(
                "Merge the step results into one cohesive answer. "
                "Preserve facts; do not invent missing evidence."
            ),
            modalities=modalities,
        )

    async def plan_step(inp: Any, *, context: RunContext | None = None) -> RunResult:
        text = inp.text() if hasattr(inp, "text") and callable(inp.text) else str(inp)
        if hasattr(planner, "arun"):
            result = await planner.arun(text)
            steps = parse_plan_steps(result.output.text(), max_steps=max_steps)
            meta = dict(result.metadata or {})
        else:
            raw = planner(text) if not asyncio.iscoroutinefunction(planner) else await planner(text)
            if isinstance(raw, RunResult):
                steps = parse_plan_steps(raw.output.text(), max_steps=max_steps)
                meta = dict(raw.metadata or {})
            elif isinstance(raw, dict) and "plan_steps" in raw:
                steps = [str(s) for s in raw["plan_steps"]][:max_steps]
                meta = {}
            else:
                steps = parse_plan_steps(str(raw), max_steps=max_steps)
                meta = {}
        payload = {"plan_steps": steps}
        return RunResult(
            output=AgentOutput(parts=[Text(json.dumps(payload))]),
            session_id=session_id or "",
            structured=payload,
            metadata={**meta, "state_updates": payload, "plan_steps": steps},
        )

    async def act_step(inp: Any, *, context: RunContext | None = None) -> RunResult:
        steps: list[str] = []
        if context is not None and context.shared_state is not None:
            raw = context.shared_state.get("plan_steps")
            if isinstance(raw, list):
                steps = [str(s) for s in raw]
        if not steps and callable(getattr(inp, "text", None)):
            steps = parse_plan_steps(inp.text(), max_steps=max_steps)
        if not steps:
            steps = parse_plan_steps(str(inp), max_steps=max_steps)

        if fan_out == "spawn":
            if model is None:
                raise ValueError("fan_out='spawn' requires model=")
            texts = await map_specialists(
                steps,
                model=model,
                tools=tools,
                modalities=modalities,
                concurrency=concurrency,
            )
        else:
            texts = []
            # Parallel map via gather when worker is Agent/callable
            async def _work(step: str) -> str:
                if hasattr(worker, "arun"):
                    r = await worker.arun(step)
                    return r.output.text()
                if asyncio.iscoroutinefunction(worker):
                    r = await worker(step)
                else:
                    r = worker(step)
                if isinstance(r, RunResult):
                    return r.output.text()
                return str(r)

            if concurrency:
                sem = asyncio.Semaphore(concurrency)

                async def _bounded(step: str) -> str:
                    async with sem:
                        return await _work(step)

                texts = list(await asyncio.gather(*[_bounded(s) for s in steps]))
            else:
                texts = list(await asyncio.gather(*[_work(s) for s in steps]))

        if context is not None and context.shared_state is not None:
            context.shared_state.write("map", texts)
            context.shared_state.write("plan_steps", steps)

        body = json.dumps({"plan_steps": steps, "map": texts}, indent=2)
        return RunResult(
            output=AgentOutput(parts=[Text(body)]),
            session_id=session_id or "",
            structured={"plan_steps": steps, "map": texts},
            metadata={
                "state_updates": {"map": texts, "plan_steps": steps},
                "fan_out": fan_out,
                "step_count": len(steps),
            },
        )

    async def synth_step(inp: Any, *, context: RunContext | None = None) -> RunResult:
        pieces: list[str] = []
        steps: list[str] = []
        if context is not None and context.shared_state is not None:
            raw_map = context.shared_state.get("map")
            raw_steps = context.shared_state.get("plan_steps")
            if isinstance(raw_map, list):
                pieces = [str(p) for p in raw_map]
            if isinstance(raw_steps, list):
                steps = [str(s) for s in raw_steps]
        if not pieces:
            text = inp.text() if hasattr(inp, "text") and callable(inp.text) else str(inp)
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    pieces = [str(p) for p in (data.get("map") or [])]
                    steps = [str(s) for s in (data.get("plan_steps") or [])]
            except (json.JSONDecodeError, TypeError):
                pieces = [text]

        combined = "\n".join(
            f"### Step {i + 1}"
            + (f": {steps[i]}" if i < len(steps) else "")
            + f"\n{piece}"
            for i, piece in enumerate(pieces)
        ) or "(no step results)"
        prompt = (
            "Integrate these specialist/worker results into one final answer.\n\n"
            + combined
        )
        if hasattr(synthesizer, "arun"):
            return await synthesizer.arun(prompt)
        if asyncio.iscoroutinefunction(synthesizer):
            raw = await synthesizer(prompt)
        else:
            raw = synthesizer(prompt)
        if isinstance(raw, RunResult):
            return raw
        return RunResult(
            output=AgentOutput(parts=[Text(str(raw))]),
            session_id=session_id or "",
        )

    wf = (
        Workflow(name, session_id=session_id, checkpointer=checkpointer)
        .step("plan", plan_step)
        .step("act", act_step)
    )

    if verify is not None:
        async def synth_body(inp: Any, *, context: RunContext | None = None) -> RunResult:
            feedback = ""
            if isinstance(inp, dict) and "feedback" in inp:
                feedback = str(inp.get("feedback") or "")

            pieces: list[str] = []
            steps: list[str] = []
            if context is not None and context.shared_state is not None:
                raw_map = context.shared_state.get("map")
                raw_steps = context.shared_state.get("plan_steps")
                if isinstance(raw_map, list):
                    pieces = [str(p) for p in raw_map]
                if isinstance(raw_steps, list):
                    steps = [str(s) for s in raw_steps]

            combined = "\n".join(
                f"### Step {i + 1}"
                + (f": {steps[i]}" if i < len(steps) else "")
                + f"\n{piece}"
                for i, piece in enumerate(pieces)
            ) or "(no step results)"
            prompt = (
                "Integrate these specialist/worker results into one final answer.\n\n"
                + combined
            )
            if feedback:
                prompt = (
                    "Previous attempt failed verification.\n"
                    f"Feedback: {feedback}\n\n"
                    "Produce a corrected final answer.\n\n"
                    + combined
                )

            if hasattr(synthesizer, "arun"):
                return await synthesizer.arun(prompt)
            if asyncio.iscoroutinefunction(synthesizer):
                raw = await synthesizer(prompt)
            else:
                raw = synthesizer(prompt)
            if isinstance(raw, RunResult):
                return raw
            return RunResult(
                output=AgentOutput(parts=[Text(str(raw))]),
                session_id=session_id or "",
            )

        loop = Loop(body=synth_body, verifier=verify, max_iterations=max_iterations)
        wf = wf.step("verify_loop", loop)
    else:
        wf = wf.step("synthesize", synth_step)

    return wf


class ToughTask:
    """One-object API for tough agent work: plan → fan-out → synthesize → verify.

    Use this when a single Agent tool-loop is not enough and you want
    deterministic decomposition + parallel specialists + quality gate.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        planner: Any | None = None,
        worker: Any | None = None,
        synthesizer: Any | None = None,
        verifier: Any | None = None,
        verify: Any | None = None,
        until: Any | None = None,
        max_iterations: int = 3,
        max_steps: int = 5,
        fan_out: FanOut = "map",
        name: str = "tough",
        session_id: str | None = None,
        checkpointer: Any = None,
        tools: list[Any] | None = None,
        modalities: str | None = None,
        concurrency: int | None = None,
    ) -> None:
        self._model = model
        self._kwargs = dict(
            planner=planner,
            worker=worker,
            synthesizer=synthesizer,
            verifier=verifier if verifier is not None else (verify if verify is not None else until),
            until=None,
            max_iterations=max_iterations,
            max_steps=max_steps,
            fan_out=fan_out,
            name=name,
            session_id=session_id,
            checkpointer=checkpointer,
            tools=tools,
            modalities=modalities,
            concurrency=concurrency,
        )
        self._workflow: Workflow | None = None

    def as_workflow(self) -> Workflow:
        """Compile to a Workflow (for nesting, HITL, checkpoints)."""
        if self._workflow is None:
            self._workflow = plan_act_verify(model=self._model, **self._kwargs)
        return self._workflow

    async def arun(self, task: Any, **kwargs: Any) -> RunResult:
        """Run the tough pipeline on ``task``."""
        wf = self.as_workflow()
        result = await wf.arun(task, **kwargs)
        # Annotate mode for observability
        meta = dict(result.metadata or {})
        meta.setdefault("tough", True)
        meta.setdefault("fan_out", self._kwargs.get("fan_out"))
        result.metadata = meta
        return result

    def run(self, task: Any, **kwargs: Any) -> RunResult:
        """Sync wrapper around :meth:`arun`."""
        return asyncio.run(self.arun(task, **kwargs))

    def __repr__(self) -> str:
        return (
            f"ToughTask(fan_out={self._kwargs.get('fan_out')!r}, "
            f"max_iterations={self._kwargs.get('max_iterations')})"
        )
