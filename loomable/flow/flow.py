"""Flow: the logical plan — a directed graph of Runnables.

A Flow is itself a Runnable so it composes (nests inside another Flow, serves
over transports, runs identically to an agent).
"""

from __future__ import annotations

__all__ = ["Flow", "FlowPlan"]

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TYPE_CHECKING

from loomable.flow.nodes import Edge, FlowConfigError, Node
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.state import Reducer


@dataclass
class FlowPlan:
    """Inspectable representation of a Flow's execution plan.

    Records the original and optimized node/edge topologies, the selected
    engine, and the optimization rules that were applied (Req 9.6, 10.8, 13.4).

    At the pre-optimizer stage, optimized_nodes/optimized_edges mirror the
    originals, applied_rules is empty, and engine shows the configured value.
    """

    original_nodes: list[str]
    original_edges: list[tuple[str, str]]
    optimized_nodes: list[str]
    optimized_edges: list[tuple[str, str]]
    engine: str
    applied_rules: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable before/after representation for print(flow.explain())."""
        lines: list[str] = []
        lines.append("=== Flow Plan ===")
        lines.append("")
        lines.append(f"Engine: {self.engine}")
        lines.append("")

        # Original topology
        lines.append("Original graph:")
        lines.append(f"  Nodes: {', '.join(self.original_nodes)}")
        if self.original_edges:
            edge_strs = [f"{s} -> {t}" for s, t in self.original_edges]
            lines.append(f"  Edges: {', '.join(edge_strs)}")
        else:
            lines.append("  Edges: (none)")

        # Optimized topology (may differ after optimizer runs)
        if (
            self.optimized_nodes != self.original_nodes
            or self.optimized_edges != self.original_edges
        ):
            lines.append("")
            lines.append("Optimized graph:")
            lines.append(f"  Nodes: {', '.join(self.optimized_nodes)}")
            if self.optimized_edges:
                edge_strs = [f"{s} -> {t}" for s, t in self.optimized_edges]
                lines.append(f"  Edges: {', '.join(edge_strs)}")
            else:
                lines.append("  Edges: (none)")

        # Applied optimization rules
        if self.applied_rules:
            lines.append("")
            lines.append(f"Applied rules: {', '.join(self.applied_rules)}")
        else:
            lines.append("")
            lines.append("Applied rules: (none)")

        return "\n".join(lines)

if TYPE_CHECKING:
    from loomable.agent.context import RunContext
    from loomable.agent.events import AgentEvents
    from loomable.agent.run import RunResult
    from loomable.flow.memory import MemoryStore
    from loomable.persist.checkpoint import Checkpointer


class Flow:
    """The logical plan: a directed graph of Runnables with shared state.

    A Flow is itself a Runnable so it can nest inside another Flow, be served
    over transports, and run identically to an agent.

    Parameters
    ----------
    nodes:
        Either a ``dict[str, Runnable]`` mapping node_ids to Runnables (graph
        mode), or a ``list[Runnable]`` for sequential shorthand where nodes
        are auto-chained in order (Req 6.6).
    edges:
        Explicit edges connecting nodes. Required for graph mode unless the
        graph has no dependencies. Ignored/overridden in list shorthand mode.
    engine:
        The execution engine to use, or ``"auto"`` for automatic selection.
    optimizer:
        An Optimizer instance or ``True`` to enable the default optimizer.
        ``False`` (default) disables optimization.
    memory:
        A shared MemoryStore instance available to all nodes.
    checkpointer:
        A Checkpointer for durable state persistence at engine boundaries.
    events:
        An AgentEvents emitter for observability.
    session_id:
        An optional session identifier for memory/checkpoint scoping.
    deps:
        Typed dependency injection object shared across all nodes.
    reducers:
        Per-key reducers for SharedState merge semantics.
    """

    def __init__(
        self,
        nodes: dict[str, Runnable] | list[Runnable],
        *,
        edges: list[Edge] | None = None,
        engine: Any | str = "auto",
        optimizer: Any | bool = False,
        memory: Any | None = None,
        checkpointer: Any | None = None,
        events: Any | None = None,
        session_id: str | None = None,
        deps: Any = None,
        reducers: dict[str, Reducer] | None = None,
        knowledge_base: Any = None,
        retrievers: Any = None,
        embedder: Any = None,
    ) -> None:
        # Store configuration parameters
        self._engine = engine
        self._optimizer = optimizer
        self._memory = memory
        self._checkpointer = checkpointer
        self._events = events
        self._session_id = session_id
        self._deps = deps
        self._reducers = reducers
        self._knowledge_base = knowledge_base
        self._retrievers = retrievers
        self._embedder = embedder

        # Build internal node dict and edge list from the input
        if isinstance(nodes, list):
            self._nodes, self._edges = self._build_from_list(nodes)
        elif isinstance(nodes, dict):
            self._nodes, self._edges = self._build_from_dict(nodes, edges)
        else:
            raise FlowConfigError(
                "Flow 'nodes' must be a dict[str, Runnable] or list[Runnable], "
                f"got {type(nodes).__name__}"
            )

        # Validate: no duplicate node_ids (already prevented by dict construction,
        # but explicit for list mode where we generate ids)
        self._validate_no_duplicate_nodes()
        # Validate: all edge endpoints reference existing nodes
        self._validate_edge_endpoints()
        if knowledge_base is not None or retrievers is not None or embedder is not None:
            from loomable.agent.memory_opts import apply_knowledge_base

            apply_knowledge_base(
                list(self._nodes.values()),
                knowledge_base=knowledge_base,
                retrievers=retrievers,
                embedder=embedder,
            )
        self._active_ctx: Any | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_node_id(runnable: Any, index: int) -> str:
        """Derive a node_id for a runnable in list shorthand.

        Uses the function name if available (for plain functions or
        FunctionRunnable wrappers), otherwise falls back to "node_{index}".
        """
        # If it's a FunctionRunnable, use the wrapped function's name
        if isinstance(runnable, FunctionRunnable):
            fn = runnable._fn
            name = getattr(fn, "__name__", None)
            if name and name != "<lambda>":
                return name
            return f"node_{index}"

        # If it's a plain callable (function), use its __name__
        if callable(runnable) and hasattr(runnable, "__name__"):
            name = runnable.__name__
            if name and name != "<lambda>":
                return name
            return f"node_{index}"

        # Fallback: positional index
        return f"node_{index}"

    @staticmethod
    def _ensure_runnable(obj: Any) -> Runnable:
        """Wrap a plain function into a FunctionRunnable if needed."""
        if isinstance(obj, Runnable):
            return obj
        # If it's a callable (sync or async function), wrap it
        if callable(obj):
            return FunctionRunnable(obj)
        raise FlowConfigError(
            f"Cannot use {type(obj).__name__!r} as a node; "
            "it must be a Runnable or a callable."
        )

    def _build_from_list(
        self, runnables: list[Runnable],
    ) -> tuple[dict[str, Node], list[Edge]]:
        """Build nodes and auto-chain edges from a sequential list (Req 6.6).

        Node IDs are derived from function names or auto-generated as
        "node_0", "node_1", etc. Edges connect each node to the next in order.
        """
        nodes: dict[str, Node] = {}
        node_ids: list[str] = []
        used_ids: set[str] = set()

        for i, item in enumerate(runnables):
            runnable = self._ensure_runnable(item)
            node_id = self._derive_node_id(item, i)

            # Handle potential duplicate derived names by appending index
            if node_id in used_ids:
                node_id = f"{node_id}_{i}"

            used_ids.add(node_id)
            node_ids.append(node_id)
            nodes[node_id] = Node(node_id=node_id, runnable=runnable)

        # Auto-chain edges: node_0 → node_1 → node_2 → ...
        edges: list[Edge] = []
        for j in range(len(node_ids) - 1):
            edges.append(Edge(source=node_ids[j], target=node_ids[j + 1]))

        return nodes, edges

    def _build_from_dict(
        self,
        nodes_dict: dict[str, Runnable],
        edges: list[Edge] | None,
    ) -> tuple[dict[str, Node], list[Edge]]:
        """Build Node objects from a dict of node_id → Runnable (or Node).

        Each value is wrapped into a Node unless it is already a Node instance.
        Duplicate keys are impossible in a dict literal but checked in
        validation for safety (e.g. programmatic construction).
        """
        nodes: dict[str, Node] = {}
        seen_ids: set[str] = set()

        for node_id, runnable_or_callable in nodes_dict.items():
            if node_id in seen_ids:
                raise FlowConfigError(
                    f"Duplicate node_id {node_id!r} in Flow definition."
                )
            seen_ids.add(node_id)

            # If already a Node, use it directly (preserves require_confirmation etc.)
            if isinstance(runnable_or_callable, Node):
                node = runnable_or_callable
                # Ensure the node_id matches the dict key
                if node.node_id != node_id:
                    node.node_id = node_id
                nodes[node_id] = node
            else:
                runnable = self._ensure_runnable(runnable_or_callable)
                nodes[node_id] = Node(node_id=node_id, runnable=runnable)

        return nodes, list(edges) if edges else []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_no_duplicate_nodes(self) -> None:
        """Raise FlowConfigError if duplicate node_ids exist (Req 6.2).

        For dict input this is structurally impossible, but list input
        generates IDs that could collide if logic has bugs. This is a
        defensive check.
        """
        # Already handled during construction — nodes is a dict, so keys
        # are unique. This method exists for completeness and future safety.
        pass

    def _validate_edge_endpoints(self) -> None:
        """Raise FlowConfigError if any edge references a nonexistent node (Req 6.4)."""
        valid_ids = set(self._nodes.keys())
        for edge in self._edges:
            if edge.source not in valid_ids:
                raise FlowConfigError(
                    f"Edge references unknown source node_id {edge.source!r}. "
                    f"Available nodes: {sorted(valid_ids)}"
                )
            if edge.target not in valid_ids:
                raise FlowConfigError(
                    f"Edge references unknown target node_id {edge.target!r}. "
                    f"Available nodes: {sorted(valid_ids)}"
                )

    # ------------------------------------------------------------------
    # Runnable interface (Req 6.7 — Flow is a Runnable)
    # ------------------------------------------------------------------

    async def arun(
        self,
        input: Any,  # noqa: A002
        *,
        context: "RunContext | None" = None,
        resume: bool | None = None,
    ) -> "RunResult":
        """Execute the flow end-to-end.

        Parameters
        ----------
        resume:
            ``True`` — require an incomplete checkpoint for ``session_id`` and
            continue from it (raises if none).
            ``False`` — ignore any incomplete checkpoint and start fresh.
            ``None`` (default) — auto-resume when an incomplete checkpoint exists.
        """
        from loomable.agent.context import RunContext
        from loomable.flow.engines.sequential import SequentialEngine
        from loomable.flow.optimizer import Optimizer
        from loomable.flow.state import SharedState

        # 1. Build SharedState
        state = SharedState(reducers=self._reducers)

        # 2. Set up RunContext (use provided or create default)
        ctx = context or RunContext()
        # Attach shared_state to context so nodes can access it
        ctx.shared_state = state
        # Attach deps if configured
        if self._deps is not None and ctx.deps is None:
            ctx.deps = self._deps
        # Attach memory store if configured (Req 12.2)
        if self._memory is not None and ctx.memory is None:
            ctx.memory = self._memory
        # Attach flow-level events when context still has the default NoOp emitter
        if self._events is not None:
            from loomable.agent.events import NoOpEvents

            if context is None or isinstance(ctx.events, NoOpEvents):
                ctx.events = self._events

        # 3. Checkpoint restore: if checkpointer configured and a checkpoint
        #    exists for this session, restore SharedState and completed_node_ids (Req 13.2)
        completed_node_ids: set[str] | None = None
        pending_decisions: dict[str, str] | None = None
        if self._checkpointer is not None and self._session_id is not None:
            existing_cp = await self._checkpointer.get(self._session_id)
            has_incomplete = existing_cp is not None and not existing_cp.complete
            if resume is True and not has_incomplete:
                raise RuntimeError(
                    f"resume=True but no incomplete checkpoint for session_id="
                    f"{self._session_id!r}"
                )
            if resume is False and has_incomplete:
                # Start fresh: mark old run complete so it won't auto-restore
                from loomable.persist.checkpoint import Checkpoint

                await self._checkpointer.put(
                    Checkpoint(
                        thread_id=self._session_id,
                        step=existing_cp.step if existing_cp else 0,
                        session_state=dict(existing_cp.session_state) if existing_cp else {},
                        complete=True,
                    )
                )
                has_incomplete = False
            if has_incomplete and existing_cp is not None:
                cp_session = existing_cp.session_state
                if "shared_state" in cp_session:
                    state = SharedState.restore(
                        cp_session["shared_state"],
                        reducers=self._reducers,
                    )
                    ctx.shared_state = state
                if "completed_node_ids" in cp_session:
                    completed_node_ids = set(cp_session["completed_node_ids"])
                # HITL resume: extract pending action decisions (Req 16.3)
                if existing_cp.pending:
                    pending_decisions = {}
                    for pa in existing_cp.pending:
                        if pa.status in ("approved", "rejected"):
                            pending_decisions[pa.tool_name] = pa.status

        # 4. Apply optimizer if enabled (Req 10.1, 10.2, 10.7, 10.8)
        optimized_flow = self
        applied_rules: list[str] = []
        optimizer = self._resolve_optimizer()
        if optimizer is not None:
            optimized_flow, applied_rules = optimizer.optimize(self)

        # 5. Resolve engine (on the potentially optimized flow)
        engine = optimized_flow._resolve_engine()

        # 6. Engine drives execution on the optimized flow
        #    Pass checkpoint-related kwargs to engines that support them.
        #    Custom engines may not accept these kwargs, so we handle gracefully.
        engine_kwargs: dict[str, Any] = {}
        if completed_node_ids is not None:
            engine_kwargs["completed_node_ids"] = completed_node_ids
        if self._checkpointer is not None:
            engine_kwargs["checkpointer"] = self._checkpointer
            engine_kwargs["session_id"] = self._session_id
        if pending_decisions is not None:
            engine_kwargs["pending_decisions"] = pending_decisions

        try:
            self._active_ctx = ctx
            result = await engine.run(
                optimized_flow,
                input,
                state,
                ctx,
                **engine_kwargs,
            )
        except TypeError:
            # Custom engine doesn't accept checkpoint kwargs — run without them
            result = await engine.run(optimized_flow, input, state, ctx)
        finally:
            if self._active_ctx is ctx:
                self._active_ctx = None

        # 7. Write a final (complete) checkpoint if checkpointer is configured
        #    Skip when cancelled so resume can continue from the last node.
        if self._checkpointer is not None and not ctx.cancelled:
            from loomable.persist.checkpoint import Checkpoint

            final_cp = Checkpoint(
                thread_id=self._session_id or "default",
                step=len(self._nodes),
                session_state={
                    "shared_state": state.snapshot(),
                    "completed_node_ids": sorted(self._nodes.keys()),
                },
                complete=True,
            )
            await self._checkpointer.put(final_cp)

        # 8. Attach the executed FlowPlan with before/after to the result
        plan = self._build_executed_plan(optimized_flow, applied_rules, engine)
        result.metadata["flow_plan"] = plan
        if completed_node_ids:
            result.metadata["resumed"] = True
            result.metadata["skipped_nodes"] = sorted(completed_node_ids)

        return result

    def cancel(self) -> bool:
        """Request cooperative cancellation of the in-flight flow run."""
        ctx = self._active_ctx
        if ctx is None:
            return False
        ctx.cancel()
        return True

    async def astream_events(
        self,
        input: Any = None,  # noqa: A002
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        context: "RunContext | None" = None,
        resume: bool | None = None,
    ) -> AsyncIterator[Any]:
        """Yield AG-UI events for this flow (NODE_* + RUN_* lifecycle)."""
        from loomable.agent.context import RunContext
        from loomable.stream import (
            RUN_ERROR,
            RUN_FINISHED,
            RUN_STARTED,
            AsyncStreamBus,
            StreamBridge,
            StreamEvent,
        )

        rid = run_id or uuid.uuid4().hex
        sid = session_id or self._session_id or ""
        bus = AsyncStreamBus(run_id=rid, session_id=sid)
        bridge = StreamBridge(bus, run_id=rid, session_id=sid, inner=self._events)
        ctx = context or RunContext()
        ctx.events = bridge

        # Bind checkpoint thread to the stream session_id for this run.
        prev_session = self._session_id
        if session_id:
            self._session_id = session_id

        async def _runner() -> None:
            try:
                bridge.publish(RUN_STARTED, {"input": str(input)[:500] if input is not None else ""})
                result = await self.arun(input, context=ctx, resume=resume)
                text = ""
                if result.output is not None and hasattr(result.output, "text"):
                    text = result.output.text() or ""
                bridge.publish(RUN_FINISHED, {"text": text[:2000]})
            except Exception as exc:  # noqa: BLE001
                bridge.publish(RUN_ERROR, {"message": str(exc), "error_type": type(exc).__name__})
            finally:
                self._session_id = prev_session
                await bus.close()

        task = asyncio.create_task(_runner())
        try:
            async for event in bus:
                yield event
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            self._session_id = prev_session

    def _resolve_engine(self) -> Any:
        """Resolve which engine to use for this flow run.

        Supports "auto", "sequential", "parallel", and "hierarchical" string
        names. Custom engine objects are used directly if they satisfy the
        ExecutionEngine protocol.

        When engine="auto", the EngineSelector inspects the topology to pick
        the best engine (Req 9.1–9.5). Explicit engine strings bypass the
        selector (Req 9.5).
        """
        from loomable.flow.engines.hierarchical import HierarchicalEngine
        from loomable.flow.engines.parallel import ParallelEngine
        from loomable.flow.engines.selector import EngineSelector
        from loomable.flow.engines.sequential import SequentialEngine

        if isinstance(self._engine, str):
            if self._engine == "auto":
                return EngineSelector.select(self._nodes, self._edges)
            if self._engine == "sequential":
                return SequentialEngine()
            if self._engine == "parallel":
                return ParallelEngine()
            if self._engine == "hierarchical":
                return HierarchicalEngine()
            raise FlowConfigError(
                f"Unknown engine name {self._engine!r}. "
                f"Supported: 'auto', 'sequential', 'parallel', 'hierarchical'."
            )

        # Custom engine object — use directly
        return self._engine

    def _resolve_optimizer(self) -> "Any | None":
        """Resolve the optimizer configuration to an Optimizer instance or None.

        Returns None if optimization is disabled (Req 10.1 — no-op when not
        enabled). Supports:
        - False / None → disabled (no-op)
        - True → default Optimizer with no rules (placeholder for tasks 10.2/10.3)
        - An Optimizer instance → used directly
        """
        from loomable.flow.optimizer import Optimizer

        if self._optimizer is False or self._optimizer is None:
            return None
        if self._optimizer is True:
            # Default optimizer with no rules — effectively a no-op until
            # rules are registered in tasks 10.2/10.3
            return Optimizer(rules=[], enabled=True)
        if isinstance(self._optimizer, Optimizer):
            return self._optimizer
        # If it's some other truthy value, treat as an Optimizer instance
        return self._optimizer

    def _build_executed_plan(
        self, optimized_flow: "Flow", applied_rules: list[str], engine: Any
    ) -> "FlowPlan":
        """Build a FlowPlan capturing both original and optimized topologies.

        This is used after optimization and engine resolution to produce the
        before/after plan representation (Req 10.8).
        """
        # Original topology (from self — the pre-optimization flow)
        original_node_ids = list(self._nodes.keys())
        original_edge_tuples = [(e.source, e.target) for e in self._edges]

        # Optimized topology (from the potentially rewritten flow)
        optimized_node_ids = list(optimized_flow._nodes.keys())
        optimized_edge_tuples = [(e.source, e.target) for e in optimized_flow._edges]

        return FlowPlan(
            original_nodes=original_node_ids,
            original_edges=original_edge_tuples,
            optimized_nodes=optimized_node_ids,
            optimized_edges=optimized_edge_tuples,
            engine=type(engine).__name__,
            applied_rules=applied_rules,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> dict[str, Node]:
        """The validated internal node dict."""
        return dict(self._nodes)

    @property
    def edges(self) -> list[Edge]:
        """The validated internal edge list."""
        return list(self._edges)

    def explain(self) -> FlowPlan:
        """Return an inspectable FlowPlan describing the execution plan.

        When an optimizer is configured, applies it and shows both the
        original and rewritten topologies (Req 10.8). When no optimizer
        is configured, optimized topology mirrors the original.
        """
        node_ids = list(self._nodes.keys())
        edge_tuples = [(e.source, e.target) for e in self._edges]

        # Determine engine name string
        engine_name: str
        if isinstance(self._engine, str):
            engine_name = self._engine
        else:
            # Custom engine object — use its class name
            engine_name = type(self._engine).__name__

        # Apply optimizer if configured to show before/after
        optimizer = self._resolve_optimizer()
        if optimizer is not None:
            optimized_flow, applied_rules = optimizer.optimize(self)
            optimized_node_ids = list(optimized_flow._nodes.keys())
            optimized_edge_tuples = [
                (e.source, e.target) for e in optimized_flow._edges
            ]
        else:
            optimized_node_ids = list(node_ids)
            optimized_edge_tuples = list(edge_tuples)
            applied_rules = []

        return FlowPlan(
            original_nodes=node_ids,
            original_edges=edge_tuples,
            optimized_nodes=optimized_node_ids,
            optimized_edges=optimized_edge_tuples,
            engine=engine_name,
            applied_rules=applied_rules,
        )

    def __repr__(self) -> str:
        n = len(self._nodes)
        e = len(self._edges)
        return f"Flow(nodes={n}, edges={e}, engine={self._engine!r})"
