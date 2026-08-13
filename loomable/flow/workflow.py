"""Workflow — High-level enterprise process orchestrator.

The Workflow is the primary way to build multi-step agentic applications.
It compiles declarative / fluent steps into a durable :class:`~loomable.flow.Flow`
graph at build time.

Happy path (no low-level graph types)::

    from loomable import Agent, Workflow, Step

    wf = (
        Workflow("sev1", session_id="inc-1", checkpointer=cp, memory=True)
        .step("gather", gatherer)
        .parallel(Step("analyst", analyst), Step("visual", visual))
        .branch(when=needs_human, then=approver, else_=auto_close)
        .step("scribe", scribe)
    )
    result = await wf.arun(email)

Complex cases (conditions, loops, nested workflows, map/plan) stay on the
same object — never force frozensets, Edge lists, or engine enums on users.
"""

from __future__ import annotations

__all__ = ["Workflow"]

import asyncio
from typing import Any, Callable, TYPE_CHECKING

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.compiler import WorkflowCompiler
from loomable.flow.flow import Flow, FlowPlan
from loomable.flow.nodes import FlowConfigError
from loomable.flow.state import SharedState

if TYPE_CHECKING:
    from loomable.persist.checkpoint import Checkpointer


def _as_steps(value: Any) -> list[Any]:
    """Normalize a branch / parallel argument into a list of composable elements."""
    from loomable.flow.step import Step

    if value is None:
        return []
    if isinstance(value, list):
        return [_wrap_runnable(v) for v in value]
    return [_wrap_runnable(value)]


def _wrap_runnable(value: Any, *, default_name: str | None = None) -> Any:
    """Accept Step / Workflow / Loop / Condition / Parallel_Group / Agent / callable."""
    from loomable.flow.condition import Condition
    from loomable.flow.loop import Loop
    from loomable.flow.parallel_group import Parallel_Group
    from loomable.flow.step import Step

    if isinstance(value, (Step, Condition, Parallel_Group, Loop, Workflow)):
        return value
    if isinstance(value, dict):
        # {"name": runnable, ...} → Parallel_Group of Steps
        steps = [Step(str(k), v) for k, v in value.items()]
        return Parallel_Group(*steps)
    name = default_name
    if name is None:
        name = getattr(value, "name", None) or getattr(value, "_name", None) or getattr(value, "_role", None)
        if callable(name):
            name = None
        name = str(name) if name else "step"
    return Step(name, value)


class Workflow:
    """Enterprise process: sequential / parallel / branch / loop over Agents & Teams.

    Parameters
    ----------
    name:
        Human-readable workflow name.
    steps:
        Optional initial list of steps (declarative style). Prefer fluent
        ``.step()`` / ``.parallel()`` / ``.branch()`` / ``.loop()``.
    session_id:
        Scopes memory and checkpoints (LangGraph-style thread id).
    checkpointer:
        Durable resume backend (JsonFile / SQLite / InMemory).
    memory:
        ``True`` for auto TieredMemoryStore, or a MemoryStore instance.
    deps:
        Shared dependency injection object for all steps.
    """

    def __init__(
        self,
        name: str = "workflow",
        steps: list[Any] | None = None,
        *,
        deps: Any = None,
        memory: bool | Any = False,
        session_id: str | None = None,
        checkpointer: "Checkpointer | None" = None,
        events: Any = None,
    ) -> None:
        self._name = name
        self._steps: list[Any] = list(steps) if steps is not None else []
        self._deps = deps
        self._session_id = session_id
        self._checkpointer = checkpointer
        self._events = events
        self._compiled_flow: Flow | None = None
        self._last_state: SharedState | None = None
        self._step_counter = 0

        # Explicit empty steps=[] still fails fast; steps=None allows fluent .step()
        if steps is not None and not steps:
            raise ValueError("At least one step is required")

        # Resolve memory configuration (store created once; Flow rebuilt lazily)
        memory_store: Any = None
        if memory is True:
            from loomable.flow.memory import TieredMemoryStore

            memory_store = TieredMemoryStore(session_id=session_id)
        elif memory and memory is not False:
            memory_store = memory
        self._memory = memory_store

        if self._steps:
            self._validate_no_duplicate_names(self._steps)

    # ------------------------------------------------------------------
    # Fluent builders (return self for chaining)
    # ------------------------------------------------------------------

    def step(
        self,
        name: str | Any,
        agent: Any | None = None,
        *,
        description: str = "",
        deps: Any = None,
    ) -> "Workflow":
        """Append a named step. ``.step("gather", agent)`` or ``.step(Step(...))``."""
        from loomable.flow.step import Step

        if agent is None:
            element = _wrap_runnable(name)
        else:
            if not isinstance(name, str) or not name:
                raise ValueError("step name must be a non-empty string")
            element = Step(name, agent, description=description, deps=deps)
        self._steps.append(element)
        self._invalidate()
        return self

    def then(self, *agents: Any) -> "Workflow":
        """Append one or more agents/callables as auto-named sequential steps."""
        for agent in agents:
            self._step_counter += 1
            self._steps.append(_wrap_runnable(agent, default_name=f"step_{self._step_counter}"))
        self._invalidate()
        return self

    def parallel(
        self,
        *steps: Any,
        name: str | None = None,
        **named: Any,
    ) -> "Workflow":
        """Run steps concurrently. Accepts Steps, agents, or ``name=agent`` kwargs."""
        from loomable.flow.parallel_group import Parallel_Group
        from loomable.flow.step import Step

        elements: list[Any] = []
        for s in steps:
            if isinstance(s, dict):
                elements.extend(Step(str(k), v) for k, v in s.items())
            else:
                elements.append(_wrap_runnable(s))
        for key, value in named.items():
            elements.append(Step(key, value))
        if not elements:
            raise ValueError("parallel() requires at least one step")
        self._steps.append(Parallel_Group(*elements, name=name))
        self._invalidate()
        return self

    def branch(
        self,
        when: Callable[[SharedState], bool],
        then: Any,
        else_: Any | None = None,
        *,
        name: str | None = None,
    ) -> "Workflow":
        """Conditional branch. ``when`` receives SharedState and returns bool."""
        from loomable.flow.condition import Condition

        then_steps = _as_steps(then)
        else_steps = _as_steps(else_) if else_ is not None else None
        # name is reserved for future RouterNode labeling; Condition has no name today
        _ = name
        self._steps.append(Condition(when, then_steps, else_steps))
        self._invalidate()
        return self

    def loop(
        self,
        body: Any,
        *,
        until: Any | None = None,
        max_iterations: int = 3,
        name: str | None = None,
    ) -> "Workflow":
        """Repeat ``body`` until verifier passes or ``max_iterations``."""
        from loomable.flow.loop import AlwaysOkVerifier, CallableVerifier, Loop, Verifier
        from loomable.flow.step import Step

        runnable = body
        if not hasattr(body, "arun"):
            loop_name = name or "loop_body"
            runnable = Step(loop_name, body)

        verifier = until
        if verifier is None:
            verifier = AlwaysOkVerifier()
        elif callable(verifier) and not isinstance(verifier, Verifier):
            verifier = CallableVerifier(verifier)

        self._steps.append(Loop(body=runnable, verifier=verifier, max_iterations=max_iterations))
        self._invalidate()
        return self

    def map(
        self,
        worker: Any,
        *,
        planner: Any | None = None,
        synthesizer: Any | None = None,
        name: str | None = None,
    ) -> "Workflow":
        """Plan → fan-out map → synthesize (complex dynamic decomposition)."""
        from loomable.flow.helpers import plan_and_execute
        from loomable.flow.step import Step

        flow = plan_and_execute(
            planner=planner or worker,
            workers=worker,
            synthesizer=synthesizer or worker,
            session_id=self._session_id,
            deps=self._deps,
            memory=self._memory,
        )
        step_name = name or "plan_and_execute"
        self._steps.append(Step(step_name, flow))
        self._invalidate()
        return self

    def add(self, *elements: Any) -> "Workflow":
        """Append pre-built composable elements (Step, Condition, nested Workflow…)."""
        for el in elements:
            self._steps.append(_wrap_runnable(el))
        self._invalidate()
        return self

    # ------------------------------------------------------------------
    # Compile / run
    # ------------------------------------------------------------------

    def _invalidate(self) -> None:
        self._compiled_flow = None

    def build(self) -> "Workflow":
        """Eagerly compile the graph (also happens automatically on arun/explain)."""
        self._ensure_compiled()
        return self

    def _ensure_compiled(self) -> Flow:
        if not self._steps:
            raise ValueError("At least one step is required — use .step() or pass steps=")
        self._validate_no_duplicate_names(self._steps)
        if self._compiled_flow is None:
            self._compiled_flow = WorkflowCompiler.compile(
                self._steps,
                name=self._name,
                deps=self._deps,
                memory=self._memory,
                session_id=self._session_id,
                checkpointer=self._checkpointer,
                events=self._events,
            )
        return self._compiled_flow

    @staticmethod
    def _validate_no_duplicate_names(steps: list[Any]) -> None:
        from loomable.flow.parallel_group import Parallel_Group
        from loomable.flow.step import Step

        seen: set[str] = set()
        for element in steps:
            name: str | None = None
            if isinstance(element, Step):
                name = element.name
            elif isinstance(element, Parallel_Group):
                name = element.name
            elif hasattr(element, "name"):
                attr = element.name
                if callable(attr) and not isinstance(attr, property):
                    attr = attr()
                if attr:
                    name = str(attr)
            if name is not None:
                if name in seen:
                    raise FlowConfigError(f"Duplicate step name: '{name}'")
                seen.add(name)

    async def arun(
        self,
        input: Any = None,  # noqa: A002
        *,
        context: RunContext | None = None,
        resume: bool | None = None,
    ) -> RunResult:
        """Execute the workflow.

        With a ``checkpointer`` + ``session_id``, incomplete runs resume
        automatically. Pass ``resume=True`` to require a checkpoint, or
        ``resume=False`` to start fresh.
        """
        flow = self._ensure_compiled()
        result = await flow.arun(input, context=context, resume=resume)
        if context is not None and context.shared_state is not None:
            self._last_state = context.shared_state
        else:
            self._last_state = SharedState()
        return result

    async def approve(
        self,
        node_id: str,
        *,
        status: str = "approved",
    ) -> None:
        """Approve or reject a HITL-paused node, then call ``arun(resume=True)``."""
        if self._checkpointer is None or self._session_id is None:
            raise RuntimeError("approve() requires checkpointer and session_id")
        cp = await self._checkpointer.get(self._session_id)
        if cp is None or not cp.pending:
            raise RuntimeError(f"No pending HITL action for session {self._session_id!r}")
        for pa in cp.pending:
            if pa.tool_name == node_id or pa.args.get("node_id") == node_id:
                pa.status = status
        await self._checkpointer.put(cp)

    async def clear_checkpoint(self) -> None:
        """Mark any incomplete checkpoint complete so the next run starts fresh."""
        if self._checkpointer is None or self._session_id is None:
            return
        cp = await self._checkpointer.get(self._session_id)
        if cp is not None and not cp.complete:
            from loomable.persist.checkpoint import Checkpoint

            await self._checkpointer.put(
                Checkpoint(
                    thread_id=self._session_id,
                    step=cp.step,
                    session_state=dict(cp.session_state),
                    complete=True,
                )
            )

    def run(self, input: Any = None) -> RunResult:  # noqa: A002
        """Synchronous convenience wrapper around ``arun``."""
        return asyncio.run(self.arun(input))

    def explain(self) -> FlowPlan:
        """Inspect compiled topology before running."""
        return self._ensure_compiled().explain()

    @property
    def state(self) -> SharedState:
        """SharedState from the most recent run (empty before first run)."""
        if self._last_state is None:
            return SharedState()
        return self._last_state

    @property
    def name(self) -> str:
        return self._name

    @property
    def steps(self) -> list[Any]:
        return list(self._steps)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def flow(self) -> Flow:
        """Compiled low-level Flow (advanced escape hatch)."""
        return self._ensure_compiled()

    def __repr__(self) -> str:
        return f"Workflow(name={self._name!r}, steps={len(self._steps)})"
