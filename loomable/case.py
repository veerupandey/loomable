"""Case — long-running goal work with a WorkItems board + AG-UI streaming.

Happy path::

    from loomable import Case, Agent
    from loomable.serve import mount_case

    case = Case(
        model=provider,
        goal="Close INC-88421 with SEV packet",
        board=True,
        dispatch="spawn",       # or "reuse"
        accept=lambda out, ctx: "SEV-" in out.text(),
        max_rounds=5,
    )
    result = await case.arun(email)
    async for ev in case.astream_events(email):
        ...
"""

from __future__ import annotations

__all__ = [
    "Case",
    "WorkItem",
    "Board",
    "build_case_workflow",
    "map_specialists",
    "parse_plan_steps",
    "board_tools",
]

import asyncio
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Callable, Literal, Sequence

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, Text
from loomable.flow.loop import CallableVerifier, Loop, Verifier
from loomable.flow.workflow import Workflow
from loomable.stream import (
    RUN_ERROR,
    RUN_FINISHED,
    RUN_STARTED,
    STATE_DELTA,
    STATE_SNAPSHOT,
    AsyncStreamBus,
    StreamEvent,
)

Dispatch = Literal["reuse", "spawn"]


@dataclass
class WorkItem:
    """One card on the Case board."""

    id: str
    title: str
    status: Literal["open", "in_progress", "blocked", "done"] = "open"
    owner: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkItem:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:8]),
            title=str(data.get("title") or ""),
            status=data.get("status") or "open",  # type: ignore[arg-type]
            owner=str(data.get("owner") or ""),
            note=str(data.get("note") or ""),
        )


class Board:
    """WorkItems board: open → in_progress → blocked → done."""

    def __init__(
        self,
        items: list[WorkItem] | None = None,
        *,
        on_change: Callable[[str, WorkItem | None], None] | None = None,
    ) -> None:
        self._items: dict[str, WorkItem] = {}
        self._on_change = on_change
        for it in items or []:
            self._items[it.id] = it

    def set_on_change(self, cb: Callable[[str, WorkItem | None], None] | None) -> None:
        self._on_change = cb

    def _notify(self, op: str, item: WorkItem | None) -> None:
        if self._on_change is not None:
            self._on_change(op, item)

    def list(self) -> list[WorkItem]:
        return list(self._items.values())

    def add(self, title: str, *, owner: str = "", status: str = "open") -> WorkItem:
        item = WorkItem(
            id=uuid.uuid4().hex[:8],
            title=title,
            status=status if status in ("open", "in_progress", "blocked", "done") else "open",  # type: ignore[arg-type]
            owner=owner,
        )
        self._items[item.id] = item
        self._notify("add", item)
        return item

    def update(
        self,
        item_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        owner: str | None = None,
        note: str | None = None,
    ) -> WorkItem | None:
        item = self._items.get(item_id)
        if item is None:
            return None
        if status is not None and status in ("open", "in_progress", "blocked", "done"):
            item.status = status  # type: ignore[assignment]
        if title is not None:
            item.title = title
        if owner is not None:
            item.owner = owner
        if note is not None:
            item.note = note
        self._notify("update", item)
        return item

    def complete(self, item_id: str) -> WorkItem | None:
        return self.update(item_id, status="done")

    def to_dict(self) -> dict[str, Any]:
        return {"items": [i.to_dict() for i in self._items.values()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Board:
        if not data:
            return cls()
        items = [WorkItem.from_dict(x) for x in (data.get("items") or []) if isinstance(x, dict)]
        return cls(items)

    def snapshot_event(self, *, run_id: str = "", session_id: str = "") -> StreamEvent:
        return StreamEvent(
            type=STATE_SNAPSHOT,
            run_id=run_id,
            session_id=session_id,
            data={"board": self.to_dict()},
        )

    def delta_event(
        self,
        *,
        op: str,
        item: WorkItem | None = None,
        run_id: str = "",
        session_id: str = "",
    ) -> StreamEvent:
        return StreamEvent(
            type=STATE_DELTA,
            run_id=run_id,
            session_id=session_id,
            data={
                "path": "board",
                "op": op,
                "item": item.to_dict() if item else None,
                "board": self.to_dict(),
            },
        )


def board_tools(
    board: Board,
    *,
    bus: AsyncStreamBus | None = None,
    run_id: str = "",
    session_id: str = "",
) -> list[Any]:
    """Return Agent tools that mutate ``board`` and optionally emit STATE_DELTA."""
    from loomable.agent import tool

    def _emit(op: str, item: WorkItem | None) -> None:
        if bus is not None:
            bus.emit_sync(board.delta_event(op=op, item=item, run_id=run_id, session_id=session_id))

    @tool
    def work_list() -> str:
        """List all work items on the case board."""
        return json.dumps(board.to_dict(), indent=2)

    @tool
    def work_add(title: str, owner: str = "") -> str:
        """Add a work item to the board (status=open)."""
        item = board.add(title, owner=owner)
        _emit("add", item)
        return json.dumps(item.to_dict())

    @tool
    def work_update(item_id: str, status: str = "", note: str = "", owner: str = "") -> str:
        """Update a work item. status: open|in_progress|blocked|done."""
        kwargs: dict[str, Any] = {}
        if status:
            kwargs["status"] = status
        if note:
            kwargs["note"] = note
        if owner:
            kwargs["owner"] = owner
        item = board.update(item_id, **kwargs)
        if item is None:
            return json.dumps({"error": f"unknown item_id {item_id}"})
        _emit("update", item)
        return json.dumps(item.to_dict())

    @tool
    def work_complete(item_id: str) -> str:
        """Mark a work item done."""
        item = board.complete(item_id)
        if item is None:
            return json.dumps({"error": f"unknown item_id {item_id}"})
        _emit("complete", item)
        return json.dumps(item.to_dict())

    return [work_list, work_add, work_update, work_complete]


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
    """Dispatch: spawn one ephemeral specialist per plan step (parallel)."""
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


def _as_accept(value: Any) -> Verifier | None:
    if value is None:
        return None
    if isinstance(value, Verifier):
        return value
    if callable(value):
        return CallableVerifier(value)
    raise TypeError(f"accept must be Verifier or callable, got {type(value)!r}")


def _normalize_dispatch(dispatch: str | None) -> Dispatch:
    raw = dispatch or "reuse"
    if raw == "reuse":
        return "reuse"
    if raw == "spawn":
        return "spawn"
    raise ValueError(f"dispatch must be 'reuse' or 'spawn', got {raw!r}")


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

    return Agent(
        model=model,
        role=role,
        goal=goal,
        instructions=instructions,
        tools=tools or [],
        modalities=modalities or "text",
    )


def build_case_workflow(
    planner: Any | None = None,
    worker: Any | None = None,
    synthesizer: Any | None = None,
    *,
    model: Any | None = None,
    accept: Any | None = None,
    max_rounds: int = 3,
    max_steps: int = 5,
    dispatch: Dispatch = "reuse",
    name: str = "case",
    session_id: str | None = None,
    checkpointer: Any = None,
    tools: list[Any] | None = None,
    modalities: str | None = None,
    concurrency: int | None = None,
    board: Board | None = None,
) -> Workflow:
    """Build a Workflow: plan → dispatch → synthesize → optional accept loop."""
    if model is None and (planner is None or worker is None):
        raise ValueError("build_case_workflow requires model= or explicit planner/worker")

    gate = _as_accept(accept)
    mode = _normalize_dispatch(dispatch)

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
    if worker is None and mode == "reuse":
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
        state_updates: dict[str, Any] = {"plan_steps": steps}
        if board is not None:
            for step in steps:
                if not any(i.title == step for i in board.list()):
                    board.add(step, status="open")
            state_updates["board"] = board.to_dict()
        payload = {"plan_steps": steps}
        return RunResult(
            output=AgentOutput(parts=[Text(json.dumps(payload))]),
            session_id=session_id or "",
            structured=payload,
            metadata={**meta, "state_updates": state_updates, "plan_steps": steps},
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

        if board is not None:
            for it in board.list():
                if it.title in steps and it.status == "open":
                    board.update(it.id, status="in_progress")

        if mode == "spawn":
            if model is None:
                raise ValueError("dispatch='spawn' requires model=")
            texts = await map_specialists(
                steps,
                model=model,
                tools=tools,
                modalities=modalities,
                concurrency=concurrency,
            )
        else:
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

        if board is not None:
            for it in board.list():
                if it.title in steps and it.status == "in_progress":
                    board.update(it.id, status="done")

        state_updates: dict[str, Any] = {"map": texts, "plan_steps": steps}
        if board is not None:
            state_updates["board"] = board.to_dict()

        if context is not None and context.shared_state is not None:
            context.shared_state.write("map", texts)
            context.shared_state.write("plan_steps", steps)
            if board is not None:
                context.shared_state.write("board", board.to_dict())

        body = json.dumps({"plan_steps": steps, "map": texts}, indent=2)
        return RunResult(
            output=AgentOutput(parts=[Text(body)]),
            session_id=session_id or "",
            structured={"plan_steps": steps, "map": texts},
            metadata={
                "state_updates": state_updates,
                "dispatch": mode,
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

    if gate is not None:
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
                    "Previous attempt failed acceptance.\n"
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

        loop = Loop(body=synth_body, verifier=gate, max_iterations=max_rounds)
        wf = wf.step("accept_loop", loop)
    else:
        wf = wf.step("synthesize", synth_step)

    return wf


class Case:
    """Long-running case: goal + WorkItems board + dispatch + accept gate."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        goal: str = "",
        board: bool | Board = True,
        planner: Any | None = None,
        worker: Any | None = None,
        synthesizer: Any | None = None,
        accept: Any | None = None,
        max_rounds: int = 3,
        max_steps: int = 5,
        dispatch: Dispatch = "reuse",
        name: str = "case",
        session_id: str | None = None,
        checkpointer: Any = None,
        tools: list[Any] | None = None,
        modalities: str | None = None,
        concurrency: int | None = None,
    ) -> None:
        self.goal = goal
        self._model = model
        if board is True:
            self.board: Board | None = Board()
        elif board is False:
            self.board = None
        else:
            self.board = board

        self._kwargs = dict(
            planner=planner,
            worker=worker,
            synthesizer=synthesizer,
            accept=accept,
            max_rounds=max_rounds,
            max_steps=max_steps,
            dispatch=dispatch,
            name=name,
            session_id=session_id,
            checkpointer=checkpointer,
            tools=tools,
            modalities=modalities,
            concurrency=concurrency,
            board=self.board,
        )
        self._workflow: Workflow | None = None
        self.session_id = session_id or ""

    @classmethod
    def from_agent(cls, agent: Any) -> Case:
        """Build a Case from ``Agent(mode='case', ...)`` attributes."""
        dispatch = getattr(agent, "_dispatch", None) or "reuse"
        max_rounds = getattr(agent, "_max_rounds", None)
        if max_rounds is None:
            max_rounds = max(1, int(getattr(agent, "_max_verify_retries", 1) or 1) + 1)
        return cls(
            model=getattr(agent, "_model", None),
            goal=str(getattr(agent, "_goal", "") or ""),
            board=bool(getattr(agent, "_board", True)),
            accept=getattr(agent, "_accept", None) or getattr(agent, "_verifier", None),
            max_rounds=int(max_rounds),
            max_steps=int(getattr(agent, "_max_plan_steps", 5) or 5),
            dispatch=dispatch if dispatch in ("reuse", "spawn") else "reuse",
            tools=list(getattr(agent, "_tools", None) or []),
            modalities=getattr(agent, "_modalities_raw", None),
            session_id=getattr(agent, "_session_id", None),
            name=str(getattr(agent, "_name", None) or "case"),
        )

    def as_workflow(self) -> Workflow:
        """Compile to a Workflow (nesting, HITL, checkpoints)."""
        if self._workflow is None:
            self._workflow = build_case_workflow(model=self._model, **self._kwargs)
        return self._workflow

    async def arun(self, task: Any, **kwargs: Any) -> RunResult:
        """Run the case pipeline on ``task``."""
        text = self._coerce_task_text(task)
        prompt = text
        if self.goal:
            prompt = f"Goal: {self.goal}\n\nTask: {text}"
        await self._hydrate_board_from_checkpoint(resume=kwargs.get("resume"))
        wf = self.as_workflow()
        result = await wf.arun(prompt, **kwargs)
        # Prefer live SharedState board after the run (covers resume + plan writes)
        self._hydrate_board_from_state(getattr(wf, "state", None))
        meta = dict(result.metadata or {})
        meta.setdefault("case", True)
        meta.setdefault("dispatch", self._kwargs.get("dispatch") or "reuse")
        if self.board is not None:
            meta["board"] = self.board.to_dict()
            ss = kwargs.get("session_state")
            if isinstance(ss, dict):
                ss["board"] = self.board.to_dict()
        result.metadata = meta
        return result

    def _hydrate_board_from_state(self, shared: Any) -> None:
        if self.board is None or shared is None:
            return
        raw = None
        if hasattr(shared, "get"):
            raw = shared.get("board")
        elif isinstance(shared, dict):
            raw = shared.get("board")
        if isinstance(raw, dict) and raw.get("items") is not None:
            restored = Board.from_dict(raw)
            # Preserve on_change callback on the live board object
            on_change = getattr(self.board, "_on_change", None)
            self.board._items = restored._items
            self.board.set_on_change(on_change)

    async def _hydrate_board_from_checkpoint(self, *, resume: bool | None = None) -> None:
        if self.board is None:
            return
        checkpointer = self._kwargs.get("checkpointer")
        session_id = self._kwargs.get("session_id") or self.session_id
        if checkpointer is None or not session_id:
            return
        if resume is False:
            return
        try:
            cp = await checkpointer.get(session_id)
        except Exception:  # noqa: BLE001
            return
        if cp is None or getattr(cp, "complete", False):
            return
        ss = getattr(cp, "session_state", None) or {}
        shared = ss.get("shared_state") if isinstance(ss, dict) else None
        board_data = None
        if isinstance(shared, dict):
            # snapshot may nest values under keys directly
            board_data = shared.get("board")
            if board_data is None and "values" in shared and isinstance(shared["values"], dict):
                board_data = shared["values"].get("board")
        if isinstance(board_data, dict):
            self._hydrate_board_from_state({"board": board_data})

    @staticmethod
    def _coerce_task_text(task: Any) -> str:
        if isinstance(task, str):
            return task
        if hasattr(task, "text") and callable(task.text):
            try:
                return str(task.text())
            except Exception:  # noqa: BLE001
                pass
        if hasattr(task, "messages"):
            from loomable.agent.builder import _input_text
            from loomable.content import AgentInput

            if isinstance(task, AgentInput):
                return _input_text(task)
            try:
                return _input_text(task)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                pass
        return str(task)
    def run(self, task: Any, **kwargs: Any) -> RunResult:
        return asyncio.run(self.arun(task, **kwargs))

    async def astream_events(
        self,
        task: Any,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """AG-UI events: lifecycle + board STATE_* + nested workflow NODE_*."""
        rid = run_id or uuid.uuid4().hex
        sid = session_id or self.session_id or ""
        bus = AsyncStreamBus(run_id=rid, session_id=sid)
        await bus.emit(
            StreamEvent(
                type=RUN_STARTED,
                run_id=rid,
                session_id=sid,
                data={"input": str(task)[:500]},
            )
        )
        if self.board is not None:
            await bus.emit(self.board.snapshot_event(run_id=rid, session_id=sid))

            def _on_board(op: str, item: WorkItem | None) -> None:
                bus.emit_sync(
                    self.board.delta_event(  # type: ignore[union-attr]
                        op=op, item=item, run_id=rid, session_id=sid
                    )
                )

            self.board.set_on_change(_on_board)

        async def _runner() -> None:
            try:
                # Reuse arun coercion + goal prefix via as_workflow input
                text = self._coerce_task_text(task)
                prompt = f"Goal: {self.goal}\n\nTask: {text}" if self.goal else text
                wf = self.as_workflow()
                async for ev in wf.astream_events(
                    prompt, session_id=sid, run_id=rid, **kwargs
                ):
                    if ev.type in (RUN_STARTED, RUN_FINISHED, RUN_ERROR):
                        continue
                    await bus.emit(ev)
                if self.board is not None:
                    await bus.emit(self.board.snapshot_event(run_id=rid, session_id=sid))
                await bus.emit(
                    StreamEvent(
                        type=RUN_FINISHED,
                        run_id=rid,
                        session_id=sid,
                        data={"board": self.board.to_dict() if self.board else None},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                await bus.emit(
                    StreamEvent(
                        type=RUN_ERROR,
                        run_id=rid,
                        session_id=sid,
                        data={"message": str(exc)},
                    )
                )
            finally:
                if self.board is not None:
                    self.board.set_on_change(None)
                await bus.close()
        task_h = asyncio.create_task(_runner())
        try:
            async for ev in bus:
                yield ev
        finally:
            if not task_h.done():
                task_h.cancel()
                try:
                    await task_h
                except (asyncio.CancelledError, Exception):
                    pass

    def __repr__(self) -> str:
        return (
            f"Case(dispatch={self._kwargs.get('dispatch')!r}, "
            f"board={self.board is not None}, "
            f"max_rounds={self._kwargs.get('max_rounds')})"
        )
