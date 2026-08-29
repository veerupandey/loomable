"""Compilers — Translate high-level workflow constructs into Flow graphs.

WorkflowCompiler: translates a steps list (Step, Condition, Parallel_Group,
Loop, Workflow) into low-level Flow primitives (Node, Edge, RouterNode).

FlowClassCompiler: introspects decorated methods on a FlowClass instance
and compiles them into a Flow graph based on @start, @listen, @router metadata.

Both compilers handle:
- Sequential connections / edge construction
- Condition / router routing
- Parallel groups / fan-out
- Cycle detection and validation

All compilation happens at construction time for early error detection and
explain() availability.
"""

from __future__ import annotations

__all__ = ["WorkflowCompiler", "FlowClassCompiler"]

from typing import Any, TYPE_CHECKING

from loomable.flow.flow import Flow
from loomable.flow.nodes import Edge, Node, RouterNode
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.state import Reducer, SharedState

if TYPE_CHECKING:
    from loomable.flow.condition import ComposableElement
    from loomable.flow.memory import MemoryStore


class WorkflowCompiler:
    """Compiles a steps list into a Flow graph.

    This is an internal implementation detail — not exported publicly.
    The Workflow class uses it at construction time to translate a
    high-level steps list into the existing Flow engine primitives.
    """

    @staticmethod
    def compile(
        steps: list[Any],
        *,
        name: str,
        deps: Any = None,
        memory: "MemoryStore | None" = None,
        session_id: str | None = None,
        reducers: dict[str, Reducer] | None = None,
        checkpointer: Any = None,
        events: Any = None,
    ) -> Flow:
        """Walk the steps list and build a Flow graph.

        Algorithm:
        1. Walk the steps list in order
        2. For each Step: create a Node with the step's name as node_id
        3. For each Parallel_Group: create a single sub-Flow node (engine="parallel")
        4. For each Condition: create a RouterNode + branch sub-flows,
           connect both branches back to next element
        5. For each Loop: wrap as a single node (it's already a Runnable)
        6. For each nested Workflow: treat as a single node (already a Runnable)
        7. Connect sequential elements with edges in declaration order
        8. Return the constructed Flow

        Parameters
        ----------
        steps:
            The list of composable elements to compile.
        name:
            The name of the workflow (used for identification).
        deps:
            Optional dependency injection object for all nodes.
        memory:
            Optional MemoryStore instance for memory-enabled workflows.
        session_id:
            Optional session identifier for memory/checkpoint scoping.
        reducers:
            Optional per-key reducers for SharedState merge semantics.

        Returns
        -------
        Flow
            The compiled Flow graph ready for execution.
        """
        from loomable.flow.condition import Condition
        from loomable.flow.loop import Loop
        from loomable.flow.parallel_group import Parallel_Group
        from loomable.flow.step import Step

        nodes: dict[str, Node | RouterNode] = {}
        edges: list[Edge] = []

        # Track the node_ids in sequence order for edge connection.
        # Each element in the steps list produces one or more node_ids
        # that represent its "entry" and "exit" points for edge connection.
        # For simple elements (Step, Loop, Parallel_Group, Workflow),
        # entry == exit == the single node_id.
        # For Condition, entry is the router node, exit is the join node.

        # We collect (entry_id, exit_id) tuples for each step element.
        element_endpoints: list[tuple[str, str]] = []

        for i, element in enumerate(steps):
            if isinstance(element, Step):
                node_id = element.name
                nodes[node_id] = Node(
                    node_id=node_id,
                    runnable=element,
                    require_confirmation=bool(
                        getattr(element, "require_confirmation", False)
                    ),
                )
                element_endpoints.append((node_id, node_id))

            elif isinstance(element, Parallel_Group):
                node_id = element.name
                # Inherit durability so on_failure="stop" inside a parallel
                # group still checkpoints successful siblings.
                inner = getattr(element, "_compiled_flow", None)
                if inner is not None and checkpointer is not None:
                    inner._checkpointer = checkpointer
                    # Scoped thread id avoids colliding with the parent flow.
                    base = session_id or "default"
                    inner._session_id = f"{base}::parallel::{node_id}"
                # Treat as a single composite node in the outer flow.
                nodes[node_id] = Node(node_id=node_id, runnable=element)
                element_endpoints.append((node_id, node_id))

            elif isinstance(element, Condition):
                # A Condition compiles to:
                # 1. A RouterNode that evaluates the predicate
                # 2. A "then" branch node (sub-flow of then_steps)
                # 3. An optional "else" branch node (sub-flow of else_steps)
                # 4. A join node that reconnects the branches
                #
                # The router selects which branch to execute based on the
                # condition predicate evaluated against SharedState.

                router_id = f"_condition_{i}_router"
                then_id = f"_condition_{i}_then"
                else_id = f"_condition_{i}_else"
                join_id = f"_condition_{i}_join"

                # Build choices list for the router
                choices = [then_id]
                if element.else_steps is not None:
                    choices.append(else_id)

                # Create the chooser function that evaluates the condition
                # predicate against SharedState from context.
                condition_fn = element.condition

                def _make_chooser(
                    cond_fn: Any, then_target: str, else_target: str | None
                ) -> Any:
                    """Create a chooser callable for the RouterNode."""

                    async def chooser(
                        input: Any, *, context: Any = None  # noqa: A002
                    ) -> Any:
                        from loomable.agent.run import RunResult
                        from loomable.content import AgentOutput, MediaPart, Modality

                        # Evaluate the condition against SharedState
                        state = SharedState()
                        if context is not None and context.shared_state is not None:
                            state = context.shared_state

                        result = cond_fn(state)
                        if result:
                            selected = then_target
                            reason = "condition_true"
                        elif else_target is not None:
                            selected = else_target
                            reason = "condition_false"
                        else:
                            # No else branch — still route to then (will be
                            # a passthrough). Actually, we need a passthrough node.
                            # We'll route to the join node directly.
                            selected = then_target
                            reason = "condition_true_no_else"

                        output = AgentOutput(
                            parts=[
                                MediaPart(
                                    modality=Modality.TEXT,
                                    media_type="text/plain",
                                    data=selected.encode("utf-8"),
                                )
                            ]
                        )
                        return RunResult(
                            output=output,
                            session_id="",
                            metadata={"selection": selected, "reason": reason},
                        )

                    return chooser

                chooser_fn = _make_chooser(condition_fn, then_id, else_id if element.else_steps else None)

                # Create the RouterNode
                router_node = RouterNode(
                    chooser=chooser_fn,
                    choices=choices,
                )
                nodes[router_id] = router_node  # type: ignore[assignment]

                # Create the "then" branch as a single node wrapping the Condition
                # for that branch. We compile then_steps into a mini sub-execution.
                then_branch = _BranchRunnable(element.then_steps)
                nodes[then_id] = Node(node_id=then_id, runnable=then_branch)

                # Create edges from router to branches
                edges.append(Edge(
                    source=router_id,
                    target=then_id,
                    condition=lambda state: state.get("_router_selection") == then_id,
                ))

                if element.else_steps is not None:
                    else_branch = _BranchRunnable(element.else_steps)
                    nodes[else_id] = Node(node_id=else_id, runnable=else_branch)
                    edges.append(Edge(
                        source=router_id,
                        target=else_id,
                        condition=lambda state: state.get("_router_selection") == else_id,
                    ))

                # Create a passthrough join node that simply passes input through.
                join_node = Node(
                    node_id=join_id,
                    runnable=_PassthroughRunnable(),
                )
                nodes[join_id] = join_node

                # Connect branches to the join node
                edges.append(Edge(source=then_id, target=join_id))
                if element.else_steps is not None:
                    edges.append(Edge(source=else_id, target=join_id))

                # The condition element's entry is the router, exit is the join
                element_endpoints.append((router_id, join_id))

            elif isinstance(element, Loop):
                # Loop is already a Runnable — wrap as a single node.
                node_id = f"_loop_{i}"
                nodes[node_id] = Node(node_id=node_id, runnable=element)
                element_endpoints.append((node_id, node_id))

            else:
                # Nested Workflow or any other Runnable — treat as a single node.
                # Use its name attribute if available, otherwise generate one.
                node_id = _get_element_name(element, i)
                nodes[node_id] = Node(node_id=node_id, runnable=element)
                element_endpoints.append((node_id, node_id))

        # Connect sequential elements with edges in declaration order.
        # Each element's exit connects to the next element's entry.
        # When the target Step declares ``reads=``, the edge carries that
        # SharedState key as its payload contract.
        for j in range(len(element_endpoints) - 1):
            _, exit_id = element_endpoints[j]
            entry_id, _ = element_endpoints[j + 1]
            payload_key = getattr(steps[j + 1], "reads", None)
            edges.append(
                Edge(source=exit_id, target=entry_id, payload_key=payload_key)
            )

        # Build the final Flow with all nodes and edges.
        # Convert the nodes dict to use Runnable values (Flow expects dict[str, Runnable]).
        # RouterNode and Node both have arun, but Flow's dict mode expects Runnable | Node.
        flow_nodes: dict[str, Any] = {}
        for nid, node in nodes.items():
            flow_nodes[nid] = node

        return Flow(
            nodes=flow_nodes,
            edges=edges,
            engine="sequential",
            memory=memory,
            session_id=session_id,
            deps=deps,
            reducers=reducers,
            checkpointer=checkpointer,
            events=events,
        )


# ---------------------------------------------------------------------------
# Internal helper classes
# ---------------------------------------------------------------------------


def _get_element_name(element: Any, index: int) -> str:
    """Extract a name from a composable element, falling back to index-based."""
    if hasattr(element, "name"):
        name = element.name
        if callable(name) and not isinstance(name, property):
            name = name()
        if name:
            return str(name)
    if hasattr(element, "_name"):
        name = element._name
        if name:
            return str(name)
    return f"_element_{index}"


class _PassthroughRunnable:
    """A Runnable that passes its input through unchanged.

    Used as a join node after Condition branches to reconnect the flow.
    Preserves :class:`~loomable.content.AgentOutput` so ``result.output.text()``
    is the branch agent's text, not ``str(AgentOutput(...))``.
    """

    async def arun(
        self, input: Any, *, context: Any = None  # noqa: A002
    ) -> Any:
        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, MediaPart, Modality

        if isinstance(input, AgentOutput):
            return RunResult(output=input, session_id="")
        output_attr = getattr(input, "output", None)
        if isinstance(output_attr, AgentOutput):
            return RunResult(output=output_attr, session_id="")
        text_fn = getattr(input, "text", None)
        if callable(text_fn):
            try:
                text = text_fn() or ""
            except TypeError:
                text = str(input) if input is not None else ""
        else:
            text = str(input) if input is not None else ""
        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=str(text).encode("utf-8"),
                )
            ]
        )
        return RunResult(output=output, session_id="")


class _BranchRunnable:
    """A Runnable that executes a list of steps sequentially.

    Used to wrap the then_steps or else_steps of a Condition into a
    single Runnable node for the compiled Flow.
    """

    def __init__(self, steps: list[Any]) -> None:
        self._steps = steps

    async def arun(
        self, input: Any, *, context: Any = None  # noqa: A002
    ) -> Any:
        """Run steps sequentially, piping output to next input."""
        from loomable.agent.run import RunResult

        current_input = input
        last_result: Any = None

        for step in self._steps:
            last_result = await step.arun(current_input, context=context)
            # Extract text output for next step's input
            if last_result and hasattr(last_result, "output") and last_result.output:
                current_input = last_result.output.text()
            else:
                current_input = ""

        if last_result is None:
            # Should not happen since Condition validates non-empty then_steps
            from loomable.content import AgentOutput, MediaPart, Modality

            output = AgentOutput(
                parts=[
                    MediaPart(
                        modality=Modality.TEXT,
                        media_type="text/plain",
                        data=b"",
                    )
                ]
            )
            return RunResult(output=output, session_id="")

        return last_result


# ---------------------------------------------------------------------------
# FlowClassCompiler
# ---------------------------------------------------------------------------


class FlowClassCompiler:
    """Compiles decorated methods on a FlowClass instance into a Flow graph.

    This is an internal implementation detail — not exported publicly.
    The FlowClass base class uses it at instantiation time to translate
    decorated methods (``@start``, ``@listen``, ``@router``) into the
    existing Flow engine primitives.

    The compilation algorithm:
    1. Introspect all methods for ``@start``, ``@listen``, ``@router`` decorators
       (look for ``_flow_meta`` attribute).
    2. Validate: at least one ``@start`` method exists.
    3. Validate: all ``@listen``/``@router`` source references point to
       existing decorated methods.
    4. Detect cycles in the listener graph (DFS-based).
    5. Create nodes: each decorated method becomes a Node (node_id = method name).
    6. Create edges: ``@listen(source)`` → ``Edge(source, method_name)``.
    7. For ``@router``: create a RouterNode routing based on return value.
    8. Multiple listeners on same source → parallel edges (fan-out).
    9. Return the constructed Flow.
    """

    @staticmethod
    def compile(cls_instance: Any) -> "Flow":
        """Compile a FlowClass instance's decorated methods into a Flow.

        Parameters
        ----------
        cls_instance:
            An instance of a FlowClass subclass whose methods are
            decorated with ``@start()``, ``@listen(source)``, or
            ``@router(source)``.

        Returns
        -------
        Flow
            The compiled Flow graph ready for execution.

        Raises
        ------
        FlowConfigError
            If no ``@start`` method exists, if a source reference is
            invalid, or if cycles are detected in the listener graph.
        """
        from loomable.flow.flow_class import _ListenMeta, _RouterMeta, _StartMeta
        from loomable.flow.nodes import FlowConfigError

        # -----------------------------------------------------------------
        # Step 1: Introspect all methods for decorator metadata
        # -----------------------------------------------------------------
        start_methods: dict[str, Any] = {}  # name -> bound method
        listen_methods: dict[str, tuple[Any, _ListenMeta]] = {}  # name -> (method, meta)
        router_methods: dict[str, tuple[Any, _RouterMeta]] = {}  # name -> (method, meta)

        # Iterate over all attributes of the instance's class (and the instance)
        for attr_name in dir(cls_instance):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(cls_instance, attr_name)
            except (AttributeError, Exception):
                continue
            if not callable(attr):
                continue

            # Check for _flow_meta on the underlying function
            # For bound methods, the metadata is on the __func__
            fn = getattr(attr, "__func__", attr)
            meta = getattr(fn, "_flow_meta", None)
            if meta is None:
                continue

            if isinstance(meta, _StartMeta):
                start_methods[attr_name] = attr
            elif isinstance(meta, _ListenMeta):
                listen_methods[attr_name] = (attr, meta)
            elif isinstance(meta, _RouterMeta):
                router_methods[attr_name] = (attr, meta)

        # -----------------------------------------------------------------
        # Step 2: Validate at least one @start method exists
        # -----------------------------------------------------------------
        if not start_methods:
            raise FlowConfigError(
                "At least one @start method is required"
            )

        # Build a set of all decorated method names (valid sources)
        all_method_names: set[str] = set(start_methods.keys())
        all_method_names.update(listen_methods.keys())
        all_method_names.update(router_methods.keys())

        # -----------------------------------------------------------------
        # Step 3: Validate all @listen/@router source references
        # -----------------------------------------------------------------
        for method_name, (_, meta) in listen_methods.items():
            if meta.source not in all_method_names:
                raise FlowConfigError(
                    f"@listen/@router references unknown source: '{meta.source}'"
                )

        for method_name, (_, meta) in router_methods.items():
            if meta.source not in all_method_names:
                raise FlowConfigError(
                    f"@listen/@router references unknown source: '{meta.source}'"
                )

        # -----------------------------------------------------------------
        # Step 4: Detect cycles in the listener graph using DFS
        # -----------------------------------------------------------------
        # Build adjacency list: source -> list of targets
        adjacency: dict[str, list[str]] = {name: [] for name in all_method_names}
        for method_name, (_, meta) in listen_methods.items():
            adjacency[meta.source].append(method_name)
        for method_name, (_, meta) in router_methods.items():
            adjacency[meta.source].append(method_name)

        cycle_path = FlowClassCompiler._detect_cycle(adjacency)
        if cycle_path is not None:
            cycle_str = " -> ".join(cycle_path)
            raise FlowConfigError(
                f"Cycle detected in listener graph: {cycle_str}"
            )

        # -----------------------------------------------------------------
        # Step 5-8: Create nodes and edges
        # -----------------------------------------------------------------
        nodes: dict[str, Any] = {}
        edges: list[Edge] = []

        # Step 5: Create nodes for each decorated method.
        # For @start and @listen methods, wrap the bound method in a
        # FunctionRunnable so it can be executed as a node.
        for method_name, method in start_methods.items():
            runnable = _MethodRunnable(method)
            nodes[method_name] = Node(node_id=method_name, runnable=runnable)

        for method_name, (method, meta) in listen_methods.items():
            runnable = _MethodRunnable(method)
            nodes[method_name] = Node(node_id=method_name, runnable=runnable)

        # Step 7: For @router methods, create a RouterNode.
        # The router method itself acts as the chooser — it returns a string
        # naming the next node to execute.
        for method_name, (method, meta) in router_methods.items():
            # Determine the downstream choices: all methods that listen to
            # or route from this router's output (i.e., methods whose source
            # is this router method).
            downstream: list[str] = adjacency.get(method_name, [])

            # If there are no explicit downstream listeners, the router's
            # return value names arbitrary nodes — use all method names as
            # potential choices (the RouterNode will validate at runtime).
            choices = downstream if downstream else [
                n for n in all_method_names if n != method_name
            ]

            router_runnable = _MethodRunnable(method)
            router_node = RouterNode(
                chooser=router_runnable,
                choices=choices,
            )
            nodes[method_name] = router_node

        # Step 6: Create edges for @listen methods.
        # @listen(source) → Edge(source, method_name)
        for method_name, (_, meta) in listen_methods.items():
            edges.append(Edge(source=meta.source, target=method_name))

        # Step 7 (continued): For @router, create edges from the router's
        # source to the router node itself.
        for method_name, (_, meta) in router_methods.items():
            edges.append(Edge(source=meta.source, target=method_name))

        # Step 8: Multiple listeners on same source → parallel edges.
        # This is naturally handled by having multiple edges from the same
        # source node — the engine handles fan-out.

        # Step 9: Return the constructed Flow.
        return Flow(
            nodes=nodes,
            edges=edges,
            engine="sequential",
        )

    @staticmethod
    def _detect_cycle(adjacency: dict[str, list[str]]) -> list[str] | None:
        """Detect a cycle in the directed graph using iterative DFS.

        Parameters
        ----------
        adjacency:
            A mapping from node name to list of successor node names.

        Returns
        -------
        list[str] | None
            The cycle path as a list of node names (e.g., ["A", "B", "A"])
            if a cycle is found, or None if the graph is acyclic.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {node: WHITE for node in adjacency}
        # parent tracking for path reconstruction
        parent: dict[str, str | None] = {node: None for node in adjacency}

        for start_node in adjacency:
            if color[start_node] != WHITE:
                continue

            # Iterative DFS using an explicit stack
            stack: list[tuple[str, int]] = [(start_node, 0)]
            color[start_node] = GRAY

            while stack:
                node, child_idx = stack[-1]
                neighbors = adjacency.get(node, [])

                if child_idx < len(neighbors):
                    # Advance to next child
                    stack[-1] = (node, child_idx + 1)
                    neighbor = neighbors[child_idx]

                    if color[neighbor] == GRAY:
                        # Found a back edge — reconstruct the cycle path
                        cycle = [neighbor]
                        for frame_node, _ in reversed(stack):
                            cycle.append(frame_node)
                            if frame_node == neighbor:
                                break
                        cycle.reverse()
                        return cycle
                    elif color[neighbor] == WHITE:
                        color[neighbor] = GRAY
                        parent[neighbor] = node
                        stack.append((neighbor, 0))
                else:
                    # All children processed
                    color[node] = BLACK
                    stack.pop()

        return None


class _MethodRunnable:
    """Adapts a bound method (sync or async) to the Runnable protocol.

    Used by FlowClassCompiler to wrap decorated FlowClass methods as
    node runnables in the compiled Flow graph.
    """

    def __init__(self, method: Any) -> None:
        import inspect
        self._method = method
        self._is_async = inspect.iscoroutinefunction(method)

    async def arun(
        self, input: Any, *, context: Any = None  # noqa: A002
    ) -> Any:
        """Execute the bound method and wrap the result in a RunResult."""
        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, MediaPart, Modality

        if self._is_async:
            raw = await self._method(input)
        else:
            raw = self._method(input)

        # If it already returns a RunResult, use it as-is.
        if isinstance(raw, RunResult):
            return raw

        # Wrap a plain return value into a RunResult with a text AgentOutput.
        text = str(raw) if raw is not None else ""
        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=text.encode("utf-8"),
                )
            ]
        )
        return RunResult(output=output, session_id="")
