"""Workflow — High-level enterprise process orchestrator.

The Workflow is the primary way to build multi-step agentic applications.
It compiles declarative / fluent steps into a durable :class:`~loomable.flow.Flow`
graph at build time.

Happy path (no low-level graph types)::

    from loomable import Agent, Workflow, Step

    wf = (
        Workflow("sev1", session_id="inc-1", checkpointer=cp)
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
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING

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
    if value is None:
        return []
    if isinstance(value, list):
        return [_wrap_runnable(v) for v in value]
    return [_wrap_runnable(value)]


def _wrap_runnable(value: Any, *, default_name: str | None = None) -> Any:
    """Accept Step / Workflow / Loop / Condition / Route / Parallel_Group / Agent / callable."""
    from loomable.flow.condition import Condition
    from loomable.flow.loop import Loop
    from loomable.flow.parallel_group import Parallel_Group
    from loomable.flow.route import Route
    from loomable.flow.step import Step

    if isinstance(value, (Step, Condition, Parallel_Group, Loop, Workflow, Route)):
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


def _collect_confirm_names(steps: list[Any]) -> list[str]:
    """Names of steps marked ``confirm=True`` / ``require_confirmation``."""
    names: list[str] = []
    for element in steps:
        if getattr(element, "require_confirmation", False):
            name = getattr(element, "name", None) or getattr(element, "_name", None)
            names.append(str(name) if name else "step")
        nested = getattr(element, "_steps", None)
        if nested:
            names.extend(_collect_confirm_names(list(nested)))
        for attr in ("_then_steps", "_else_steps"):
            child = getattr(element, attr, None)
            if child:
                names.extend(_collect_confirm_names(list(child)))
        body = getattr(element, "_body", None)
        if body is not None:
            names.extend(_collect_confirm_names([body]))
    return names


def _unsupported_confirm_sites(steps: list[Any]) -> list[str]:
    """HITL sites that compile but never pause (branch / loop / parallel / route)."""
    from loomable.flow.condition import Condition
    from loomable.flow.loop import Loop
    from loomable.flow.parallel_group import Parallel_Group
    from loomable.flow.route import Route
    from loomable.flow.step import Step

    sites: list[str] = []
    for element in steps:
        if isinstance(element, Parallel_Group):
            names = _collect_confirm_names(list(element._steps))
            if names:
                sites.append(f"parallel ({', '.join(names)})")
        elif isinstance(element, Condition):
            nested: list[Any] = list(element._then_steps or [])
            nested.extend(element._else_steps or [])
            names = _collect_confirm_names(nested)
            if names:
                sites.append(f"branch ({', '.join(names)})")
        elif isinstance(element, Route):
            nested = []
            for branch in element.choices.values():
                if isinstance(branch, list):
                    nested.extend(branch)
                else:
                    nested.append(branch)
            names = _collect_confirm_names(nested)
            if names:
                sites.append(f"route ({', '.join(names)})")
        elif isinstance(element, Loop):
            body = getattr(element, "_body", None)
            names = _collect_confirm_names([body] if body is not None else [])
            if names:
                sites.append(f"loop ({', '.join(names)})")
        elif isinstance(element, Step):
            inner = getattr(element, "_agent", None)
            if inner is not None and hasattr(inner, "_steps"):
                sites.extend(_unsupported_confirm_sites(list(inner._steps)))
        elif type(element).__name__ == "Workflow" and hasattr(element, "_steps"):
            sites.extend(_unsupported_confirm_sites(list(element._steps)))
    return sites


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
        Scopes memory and checkpoints (thread id).
    checkpointer:
        Durable resume backend (JsonFile / SQLite / InMemory).
    memory:
        ``True`` for auto TieredMemoryStore, or a MemoryStore instance.
    knowledge_base:
        Shared vector-DB knowledge base inherited by Agent steps that do not
        already have one (same object as ``Agent(knowledge_base=...)``).
    retrievers / embedder:
        Extra search tools / embedder inherited the same way.
    require_tools / strict_require_tools:
        Inherited by Agent steps that do not already set ``require_tools``.
        Same semantics as ``Agent(require_tools=..., strict_require_tools=...)``.
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
        knowledge_base: Any = None,
        retrievers: Any = None,
        embedder: Any = None,
        require_tools: list[str] | None = None,
        strict_require_tools: bool = False,
        reducers: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._steps: list[Any] = list(steps) if steps is not None else []
        self._deps = deps
        self._session_id = session_id
        self._checkpointer = checkpointer
        self._events = events
        self._knowledge_base = knowledge_base
        self._retrievers = retrievers
        self._embedder = embedder
        self._require_tools = list(require_tools) if require_tools else []
        self._strict_require_tools = bool(strict_require_tools)
        self._reducers = dict(reducers) if reducers else None
        self._compiled_flow: Flow | None = None
        self._last_state: SharedState | None = None
        self._clear_checkpoint_pending = False
        self._step_counter = 0
        self._active_ctx: RunContext | None = None

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
            if (
                knowledge_base is not None
                or retrievers is not None
                or embedder is not None
            ):
                from loomable.agent.memory_opts import apply_knowledge_base

                apply_knowledge_base(
                    self._steps,
                    knowledge_base=knowledge_base,
                    retrievers=retrievers,
                    embedder=embedder,
                )
            if self._require_tools or self._strict_require_tools:
                from loomable.agent.memory_opts import apply_require_tools

                apply_require_tools(
                    self._steps,
                    require_tools=self._require_tools or None,
                    strict_require_tools=self._strict_require_tools or None,
                )

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
        require_confirmation: bool = False,
        confirm: bool | None = None,
        require_tools: list[str] | None = None,
        strict_require_tools: bool | None = None,
        on_failure: str = "raise",
        max_retries: int | None = None,
        fallback: Any | None = None,
        reads: str | None = None,
        complexity: str | None = None,
    ) -> "Workflow":
        """Append a named step. ``.step("gather", agent)`` or ``.step(Step(...))``.

        Pass ``confirm=True`` (or ``require_confirmation=True``) to pause the
        workflow before the step until ``approve(name)`` + ``arun(resume=True)``.
        ``require_tools`` / ``strict_require_tools`` apply when ``agent`` is an
        :class:`~loomable.agent.builder.Agent`.

        Graph-engineering knobs:

        - ``on_failure`` — ``raise`` / ``retry`` / ``skip`` / ``fallback`` / ``stop``
        - ``reads`` — SharedState key this step consumes (edge data contract)
        - ``complexity`` — ``"low"`` / ``"high"`` cost hint for model tiers
        """
        from loomable.flow.step import Step

        if confirm is not None:
            require_confirmation = confirm

        if agent is None:
            element = _wrap_runnable(name)
            if require_confirmation and isinstance(element, Step):
                element.require_confirmation = True
            if isinstance(element, Step):
                if on_failure != "raise":
                    element.on_failure = on_failure  # type: ignore[assignment]
                if max_retries is not None:
                    element.max_retries = max_retries
                if fallback is not None:
                    element._fallback = Step._as_runnable(fallback, label="fallback")
                if reads is not None:
                    element.reads = reads
                if complexity is not None:
                    element.complexity = complexity  # type: ignore[assignment]
        else:
            if not isinstance(name, str) or not name:
                raise ValueError("step name must be a non-empty string")
            element = Step(
                name,
                agent,
                description=description,
                deps=deps,
                require_confirmation=require_confirmation,
                on_failure=on_failure,  # type: ignore[arg-type]
                max_retries=max_retries,
                fallback=fallback,
                reads=reads,
                complexity=complexity,  # type: ignore[arg-type]
            )
        inner = getattr(element, "_agent", None)
        if require_tools is not None or strict_require_tools is True:
            from loomable.agent.memory_opts import inherit_agent_require_tools

            if inner is not None and hasattr(inner, "_require_tools"):
                inherit_agent_require_tools(
                    inner,
                    require_tools=require_tools,
                    strict_require_tools=strict_require_tools,
                    overwrite=True,
                )
            elif require_tools:
                raise TypeError(
                    "Workflow.step(..., require_tools=) only applies when the step is an Agent"
                )
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
        for el in elements:
            if getattr(el, "require_confirmation", False):
                from loomable.flow.nodes import FlowConfigError

                raise FlowConfigError(
                    "confirm=True is not supported inside Workflow.parallel(); "
                    "put HITL on a sequential .step(..., confirm=True) after the group."
                )
        self._steps.append(Parallel_Group(*elements, name=name))
        self._invalidate()
        return self

    def branch(
        self,
        when: Callable[[SharedState], bool],
        then: Any,
        else_: Any | None = None,
    ) -> "Workflow":
        """Conditional branch. ``when`` receives SharedState and returns bool."""
        from loomable.flow.condition import Condition

        then_steps = _as_steps(then)
        else_steps = _as_steps(else_) if else_ is not None else None
        self._steps.append(Condition(when, then_steps, else_steps))
        self._invalidate()
        return self

    def route(
        self,
        chooser: Any,
        choices: dict[str, Any] | None = None,
        *,
        handoff: bool = False,
        **named: Any,
    ) -> "Workflow":
        """N-way route (Agno Router / LangGraph multi-edge).

        ``chooser`` returns a choice name, list of names, or a
        :class:`~loomable.flow.command.Command` with ``goto=``::

            wf.route(
                classify,
                quick=quick_agent,
                full=full_audit,
                human=human_review,
            )
        """
        from loomable.flow.route import Route

        merged: dict[str, Any] = {}
        if choices:
            merged.update(choices)
        merged.update(named)
        if not merged:
            raise ValueError("route() requires at least one choice")
        # Normalize each choice into list of wrapped steps for the compiler
        normalized: dict[str, Any] = {}
        for key, value in merged.items():
            if isinstance(value, list):
                normalized[key] = [_wrap_runnable(v) for v in value]
            else:
                normalized[key] = _wrap_runnable(value, default_name=key)
        self._steps.append(Route(chooser, normalized, handoff=handoff))
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

    def verify(
        self,
        body: Any,
        *,
        check: Any,
        max_retries: int = 2,
        name: str | None = None,
    ) -> "Workflow":
        """Verify ``body`` output before it moves downstream (bounded repair cycle).

        Graph shape::

            WORK -> VERIFY -> PASS -> next
                       |
                       -> FAIL -> feedback -> WORK  (up to max_retries)

        This is the graph-engineering form of ``.loop(..., until=)``: a
        dedicated verifier gate with a hard iteration budget
        (``max_retries + 1`` total attempts).
        """
        if check is None:
            raise ValueError("verify() requires check= (Verifier or callable)")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        return self.loop(
            body,
            until=check,
            max_iterations=max_retries + 1,
            name=name,
        )

    def map(
        self,
        worker: Any,
        *,
        planner: Any | None = None,
        synthesizer: Any | None = None,
        over: str = "plan_steps",
        name: str | None = None,
    ) -> "Workflow":
        """Plan → fan-out map → synthesize (complex dynamic decomposition)."""
        from loomable.flow.helpers import plan_and_execute
        from loomable.flow.step import Step

        flow = plan_and_execute(
            planner=planner or worker,
            workers=worker,
            synthesizer=synthesizer or worker,
            over=over,
            session_id=self._session_id,
            deps=self._deps,
            memory=self._memory,
        )
        step_name = name or "plan_and_execute"
        self._steps.append(Step(step_name, flow))
        self._invalidate()
        return self

    def map_over(
        self,
        worker: Any,
        *,
        over: str,
        concurrency: int | None = None,
        name: str | None = None,
    ) -> "Workflow":
        """Fan out ``worker`` over ``SharedState[over]`` (LangGraph Send / map parity).

        Populate ``over`` with plain values or :class:`~loomable.flow.send.Send`
        instances (``Send.node`` is metadata; ``Send.arg`` is worker input).
        """
        from loomable.flow.flow import Flow
        from loomable.flow.helpers import _ensure_runnable
        from loomable.flow.nodes import MapNode, Node
        from loomable.flow.step import Step

        body = _ensure_runnable(worker)
        map_node = MapNode(body, over=over, concurrency=concurrency)
        flow = Flow(
            {
                "map": Node(node_id="map", runnable=map_node),
            },
            edges=[],
            engine="sequential",
            session_id=self._session_id,
            deps=self._deps,
            memory=self._memory,
            events=self._events,
            reducers=self._reducers,
        )
        step_name = name or f"map_over_{over}"
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
        if self._knowledge_base is not None or self._retrievers is not None or self._embedder is not None:
            from loomable.agent.memory_opts import apply_knowledge_base

            apply_knowledge_base(
                self._steps,
                knowledge_base=self._knowledge_base,
                retrievers=self._retrievers,
                embedder=self._embedder,
            )
        if self._require_tools or self._strict_require_tools:
            from loomable.agent.memory_opts import apply_require_tools

            apply_require_tools(
                self._steps,
                require_tools=self._require_tools or None,
                strict_require_tools=self._strict_require_tools or None,
            )

    def build(self) -> "Workflow":
        """Eagerly compile the graph (also happens automatically on arun/explain)."""
        self._ensure_compiled()
        return self

    def _ensure_compiled(self) -> Flow:
        if not self._steps:
            raise ValueError("At least one step is required — use .step() or pass steps=")
        self._validate_no_duplicate_names(self._steps)
        if self._compiled_flow is None:
            from loomable.agent.memory_opts import apply_knowledge_base

            apply_knowledge_base(
                self._steps,
                knowledge_base=self._knowledge_base,
                retrievers=self._retrievers,
                embedder=self._embedder,
            )
            if self._require_tools or self._strict_require_tools:
                from loomable.agent.memory_opts import apply_require_tools

                apply_require_tools(
                    self._steps,
                    require_tools=self._require_tools or None,
                    strict_require_tools=self._strict_require_tools or None,
                )
            unsupported = _unsupported_confirm_sites(self._steps)
            if unsupported:
                from loomable.flow.nodes import FlowConfigError

                raise FlowConfigError(
                    "confirm=True is not supported inside Workflow.parallel(), "
                    ".branch(), or .loop(); put HITL on a sequential "
                    f".step(..., confirm=True). Found: {'; '.join(unsupported)}"
                )
            confirm_names = _collect_confirm_names(self._steps)
            if confirm_names and (self._checkpointer is None or not self._session_id):
                from loomable.flow.nodes import FlowConfigError

                raise FlowConfigError(
                    "confirm=True requires Workflow(..., checkpointer=..., session_id=...); "
                    f"HITL steps: {', '.join(confirm_names)}"
                )
            self._compiled_flow = WorkflowCompiler.compile(
                self._steps,
                name=self._name,
                deps=self._deps,
                memory=self._memory,
                session_id=self._session_id,
                checkpointer=self._checkpointer,
                events=self._events,
                reducers=self._reducers,
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
        if self._clear_checkpoint_pending and self._checkpointer is not None:
            await self.clear_checkpoint()
            self._clear_checkpoint_pending = False
        flow = self._ensure_compiled()
        ctx = context or RunContext()
        self._active_ctx = ctx
        try:
            result = await flow.arun(input, context=ctx, resume=resume)
        finally:
            # Preserve SharedState even when a node raises (e.g. StepFailed),
            # so callers can inspect completed work after a hard stop.
            if ctx.shared_state is not None:
                self._last_state = ctx.shared_state
            elif self._last_state is None:
                self._last_state = SharedState()
            if self._active_ctx is ctx:
                self._active_ctx = None
        return result

    def cancel(self) -> bool:
        """Request cooperative cancellation of the in-flight workflow run."""
        hit = False
        ctx = self._active_ctx
        if ctx is not None:
            ctx.cancel()
            hit = True
        flow = self._compiled_flow
        if flow is not None and hasattr(flow, "cancel"):
            hit = flow.cancel() or hit
        return hit

    def bind_session(
        self,
        session_id: str | None,
        *,
        resume: bool | None = None,
    ) -> None:
        """Bind checkpoint thread id (HTTP / multi-turn workflows).

        Invalidates the compiled graph when the session changes so checkpoints
        align with the new thread.
        """
        if session_id is None:
            return
        if session_id != self._session_id:
            self._compiled_flow = None
        self._session_id = session_id
        if resume is False and self._checkpointer is not None:
            self._clear_checkpoint_pending = True

    async def astream_events(
        self,
        input: Any = None,  # noqa: A002
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        context: RunContext | None = None,
        resume: bool | None = None,
    ) -> AsyncIterator[Any]:
        """Yield AG-UI events (NODE_* lifecycle + nested run frames)."""
        flow = self._ensure_compiled()
        ctx = context or RunContext()
        self._active_ctx = ctx
        try:
            async for event in flow.astream_events(
                input,
                session_id=session_id or self._session_id,
                run_id=run_id,
                context=ctx,
                resume=resume,
            ):
                yield event
        finally:
            if ctx.shared_state is not None:
                self._last_state = ctx.shared_state
            elif self._last_state is None:
                self._last_state = SharedState()
            if self._active_ctx is ctx:
                self._active_ctx = None

    async def approve(
        self,
        node_id: str,
        *,
        status: str = "approved",
    ) -> None:
        """Approve or reject a HITL-paused node, then call ``arun(resume=True)``.

        ``status`` must be ``"approved"`` or ``"rejected"``.
        """
        if status not in ("approved", "rejected"):
            raise ValueError(
                f"approve status must be 'approved' or 'rejected', got {status!r}"
            )
        if self._checkpointer is None or self._session_id is None:
            raise RuntimeError("approve() requires checkpointer and session_id")
        cp = await self._checkpointer.get(self._session_id)
        if cp is None or not cp.pending:
            raise RuntimeError(f"No pending HITL action for session {self._session_id!r}")
        matched = False
        for pa in cp.pending:
            if pa.tool_name == node_id or pa.args.get("node_id") == node_id:
                pa.status = status
                matched = True
        if not matched:
            raise RuntimeError(f"No pending HITL action matching node {node_id!r}")
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

    async def get_state(self) -> dict[str, Any]:
        """Return the current workflow state (LangGraph-style control plane).

        Prefers an incomplete checkpoint when ``checkpointer`` + ``session_id``
        are set; otherwise returns the in-memory state from the last run.
        """
        cp = None
        if self._checkpointer is not None and self._session_id is not None:
            cp = await self._checkpointer.get(self._session_id)

        if cp is not None:
            session = cp.session_state or {}
            return {
                "values": dict(session.get("shared_state") or {}),
                "completed": list(session.get("completed_node_ids") or []),
                "pending": [
                    {
                        "tool_name": p.tool_name,
                        "call_id": p.call_id,
                        "args": dict(p.args),
                        "status": p.status,
                    }
                    for p in (cp.pending or [])
                ],
                "complete": bool(cp.complete),
                "step": cp.step,
                "thread_id": cp.thread_id,
                "next": None if cp.complete else self._infer_next(session),
            }

        values: dict[str, Any] = {}
        if self._last_state is not None:
            values = self._last_state.snapshot()
        return {
            "values": values,
            "completed": [],
            "pending": [],
            "complete": bool(values),
            "step": 0,
            "thread_id": self._session_id,
            "next": None,
        }

    async def update_state(
        self,
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> dict[str, Any]:
        """Patch SharedState on the current checkpoint (LangGraph update_state).

        Requires ``checkpointer`` + ``session_id``. Writes an incomplete
        checkpoint so the next ``arun(resume=True)`` picks up the patch.
        When ``as_node`` is set, that node id is also marked completed.
        """
        if self._checkpointer is None or self._session_id is None:
            raise RuntimeError(
                "update_state() requires Workflow(..., checkpointer=..., session_id=...)"
            )
        from loomable.persist.checkpoint import Checkpoint

        existing = await self._checkpointer.get(self._session_id)
        session: dict[str, Any] = {}
        step = 0
        pending: list[Any] = []
        if existing is not None:
            session = dict(existing.session_state or {})
            step = existing.step
            pending = list(existing.pending or [])
            for pa in pending:
                if pa.status == "pending" and (
                    pa.tool_name == as_node or pa.args.get("node_id") == as_node
                ):
                    raise RuntimeError(
                        f"update_state(as_node={as_node!r}) blocked: node has pending HITL"
                    )

        shared = dict(session.get("shared_state") or {})
        shared.update(values)
        session["shared_state"] = shared

        completed = list(session.get("completed_node_ids") or [])
        if as_node and as_node not in completed:
            completed.append(as_node)
            session["completed_node_ids"] = completed

        cp = Checkpoint(
            thread_id=self._session_id,
            step=step,
            session_state=session,
            complete=False,
            pending=pending,
        )
        await self._checkpointer.put(cp)

        # Mirror into in-memory state for immediate get_state / inspection
        if self._last_state is None:
            self._last_state = SharedState(reducers=self._reducers)
        for key, value in values.items():
            self._last_state.write(key, value)

        return await self.get_state()

    async def list_states(self) -> list[dict[str, Any]]:
        """List checkpoint history for this workflow's session (time-travel)."""
        if self._checkpointer is None or self._session_id is None:
            if self._last_state is not None:
                return [await self.get_state()]
            return []
        if not hasattr(self._checkpointer, "list"):
            current = await self.get_state()
            return [current] if current.get("values") or current.get("pending") else []
        checkpoints = await self._checkpointer.list(self._session_id)
        out: list[dict[str, Any]] = []
        for cp in checkpoints:
            session = cp.session_state or {}
            out.append(
                {
                    "values": dict(session.get("shared_state") or {}),
                    "completed": list(session.get("completed_node_ids") or []),
                    "pending": [
                        {
                            "tool_name": p.tool_name,
                            "call_id": p.call_id,
                            "args": dict(p.args),
                            "status": p.status,
                        }
                        for p in (cp.pending or [])
                    ],
                    "complete": bool(cp.complete),
                    "step": cp.step,
                    "thread_id": cp.thread_id,
                    "timestamp": getattr(cp, "timestamp", None),
                }
            )
        return out

    async def fork_session(self, new_session_id: str) -> dict[str, Any]:
        """Fork the current session checkpoint into a new session id (time-travel).

        Requires a checkpointer that implements ``fork``. Returns the forked
        state's ``get_state()`` view under the new session.
        """
        if self._checkpointer is None or self._session_id is None:
            raise RuntimeError(
                "fork_session() requires Workflow(..., checkpointer=..., session_id=...)"
            )
        if not hasattr(self._checkpointer, "fork"):
            raise RuntimeError(
                f"{type(self._checkpointer).__name__} does not support fork()"
            )
        forked = await self._checkpointer.fork(self._session_id, new_session_id)
        if forked is None:
            raise RuntimeError(
                f"No checkpoint to fork for session_id={self._session_id!r}"
            )
        # Point this workflow at the forked thread for subsequent resume/get_state
        self._session_id = new_session_id
        if self._compiled_flow is not None:
            self._compiled_flow._session_id = new_session_id
        return await self.get_state()

    def _infer_next(self, session: dict[str, Any]) -> list[str] | None:
        """Best-effort next node ids from incomplete checkpoint."""
        completed = set(session.get("completed_node_ids") or [])
        try:
            flow = self._ensure_compiled()
        except Exception:  # noqa: BLE001
            return None
        remaining = [nid for nid in flow.nodes if nid not in completed]
        return remaining[:1] if remaining else []

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
