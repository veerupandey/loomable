"""Node, Edge, MapNode, RouterNode, and FlowConfigError definitions.

Nodes wrap a single Runnable identified by a unique node_id. Edges connect
nodes with optional condition predicates over SharedState. MapNode fans out
a Runnable over a runtime list with concurrency control and per-item fault
isolation. RouterNode selects downstream node(s) via a predicate or model-driven
chooser, supporting handoff semantics.
"""

from __future__ import annotations

__all__ = [
    "Edge",
    "FlowConfigError",
    "MapNode",
    "Node",
    "RouterNode",
]

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from loomable.kernel.errors import LoomableError
from loomable.kernel.subagents import DelegatedTask, SubagentManager

if TYPE_CHECKING:
    from loomable.agent.context import RunContext
    from loomable.agent.run import RunResult
    from loomable.flow.runnable import Runnable
    from loomable.flow.state import SharedState


@dataclass
class Edge:
    """A directed connection between two nodes in a Flow.

    Parameters
    ----------
    source:
        The ``node_id`` of the source node.
    target:
        The ``node_id`` of the target node.
    condition:
        An optional predicate over SharedState. When present, the Flow
        traverses this edge only when the predicate evaluates truthy
        against the current SharedState (Req 6.5).
    payload_key:
        Optional SharedState key whose value is the input for ``target``.
        When set, the edge carries a real data contract: the engine feeds
        ``state[payload_key]`` instead of the source node's ambient output.
    """

    source: str
    target: str
    condition: Callable[["SharedState"], bool] | None = None
    payload_key: str | None = None


class Node:
    """Wraps exactly one Runnable identified by a unique node_id.

    Parameters
    ----------
    node_id:
        A unique string identifier for this node within a Flow.
    runnable:
        The Runnable that this node executes.
    require_confirmation:
        When True, the Flow pauses before executing this node for
        human-in-the-loop approval (Req 16.1).
    manager:
        When True, signals to the EngineSelector that this node acts
        as a manager in a Hierarchical engine layout (Req 9.4).
    reads:
        Optional SharedState key this node consumes (edge data contract).
        Engines prefer ``state[reads]`` over ambient previous-node output,
        including when this node is a nested-flow root with no incoming edge.
    """

    def __init__(
        self,
        node_id: str,
        runnable: "Runnable",
        *,
        require_confirmation: bool = False,
        manager: bool = False,
        reads: str | None = None,
    ) -> None:
        self.node_id = node_id
        self.runnable = runnable
        self.require_confirmation = require_confirmation
        self.manager = manager
        self.reads = reads

    def __repr__(self) -> str:
        flags = []
        if self.require_confirmation:
            flags.append("hitl")
        if self.manager:
            flags.append("manager")
        extra = f" [{', '.join(flags)}]" if flags else ""
        return f"Node({self.node_id!r}{extra})"


class FlowConfigError(LoomableError):
    """Raised for Flow configuration errors detected before any run.

    Covers duplicate node_ids (Req 6.2), edges referencing nonexistent
    nodes (Req 6.4), and illegal cycles (Req 8.6).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class MapNode:
    """Fan-out a Runnable over a runtime list (Req 11.1–11.3).

    MapNode is itself a Runnable. It reads a list from
    ``context.shared_state.get(over)``, runs ``body.arun(item, context=context)``
    for each item concurrently via SubagentManager, and fans individual results
    back into a single collection.

    Per-item failures are isolated: one item failing does not cancel others.
    Failed items are recorded with error metadata in the results list.

    Parameters
    ----------
    body:
        The Runnable to execute for each item in the list.
    over:
        The SharedState key whose value is the list to iterate over.
    concurrency:
        Maximum number of concurrent executions. When ``None``, all items
        run concurrently without limit.
    """

    def __init__(
        self,
        body: "Runnable",
        *,
        over: str,
        concurrency: int | None = None,
    ) -> None:
        self.body = body
        self.over = over
        self.concurrency = concurrency

    async def arun(
        self, input: Any, *, context: "RunContext | None" = None  # noqa: A002
    ) -> "RunResult":
        """Execute the body Runnable for each item in the list from SharedState.

        Returns a RunResult whose output summarizes the fan-out and whose
        metadata contains per-item results (successes and failures).
        """
        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, MediaPart, Modality

        # 1. Read the list of items from shared state
        if context is None or context.shared_state is None:
            raise FlowConfigError(
                f"MapNode over={self.over!r}: requires RunContext.shared_state"
            )
        raw_value = context.shared_state.get(self.over)
        if raw_value is None:
            raise FlowConfigError(
                f"MapNode over={self.over!r}: key missing from SharedState"
            )
        if not isinstance(raw_value, list):
            raise FlowConfigError(
                f"MapNode over={self.over!r}: expected list, got {type(raw_value).__name__}"
            )
        from loomable.flow.send import send_args

        items = send_args(raw_value)

        if not items:
            # Empty list is a valid explicit no-op.
            context.shared_state.write("map", [])
            output = AgentOutput(
                parts=[
                    MediaPart(
                        modality=Modality.TEXT,
                        media_type="text/plain",
                        data=b"",
                    )
                ]
            )
            return RunResult(
                output=output,
                session_id="",
                metadata={
                    "map_results": [],
                    "map_errors": [],
                    "map_total": 0,
                    "map_succeeded": 0,
                    "map_failed": 0,
                    "state_updates": {"map": []},
                },
            )

        # 2. Build delegated tasks with concurrency control
        semaphore = (
            asyncio.Semaphore(self.concurrency) if self.concurrency else None
        )

        async def _run_item(item: Any, idx: int) -> "RunResult":
            """Run body for a single item, respecting the semaphore."""
            if semaphore:
                async with semaphore:
                    return await self.body.arun(item, context=context)
            return await self.body.arun(item, context=context)

        manager = SubagentManager()
        tasks: list[DelegatedTask] = []
        for idx, item in enumerate(items):
            task_id = f"map_item_{idx}"
            # Capture idx and item in the closure via default args
            tasks.append(
                DelegatedTask(
                    task_id=task_id,
                    task=f"MapNode item {idx}",
                    context={"item": item, "index": idx},
                    agent_factory=self._make_factory(item, idx, context, semaphore),
                )
            )

        # 3. Run all tasks concurrently via SubagentManager (inherits isolation)
        outcomes = await manager.run_all(tasks)

        # 4. Fan results into a collection, isolating per-item failures
        map_results: list[dict[str, Any]] = []
        map_errors: list[dict[str, Any]] = []
        successful_results: list["RunResult"] = []

        for idx, outcome in enumerate(outcomes):
            if outcome.error is not None:
                map_errors.append(
                    {
                        "index": idx,
                        "task_id": outcome.task_id,
                        "error": str(outcome.error),
                        "cause": str(outcome.error.__cause__) if outcome.error.__cause__ else None,
                    }
                )
                map_results.append(
                    {
                        "index": idx,
                        "task_id": outcome.task_id,
                        "success": False,
                        "error": str(outcome.error),
                    }
                )
            else:
                result = outcome.result
                successful_results.append(result)
                map_results.append(
                    {
                        "index": idx,
                        "task_id": outcome.task_id,
                        "success": True,
                        "result": result,
                    }
                )

        # 5. Assemble final RunResult + write texts into SharedState["map"]
        #    so synthesizers / Case can consume step outputs easily.
        total = len(items)
        succeeded = len(successful_results)
        failed = len(map_errors)
        summary = f"MapNode: {succeeded}/{total} succeeded, {failed}/{total} failed"
        map_texts: list[str] = []
        for item_result in successful_results:
            try:
                map_texts.append(item_result.output.text())
            except Exception:  # noqa: BLE001
                map_texts.append(str(item_result))
        if context is not None and context.shared_state is not None:
            context.shared_state.write("map", map_texts)

        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=summary.encode("utf-8"),
                )
            ]
        )
        return RunResult(
            output=output,
            session_id="",
            metadata={
                "map_results": map_results,
                "map_errors": map_errors,
                "map_total": total,
                "map_succeeded": succeeded,
                "map_failed": failed,
                "state_updates": {"map": map_texts},
            },
        )

    def _make_factory(
        self,
        item: Any,
        idx: int,
        context: "RunContext | None",
        semaphore: asyncio.Semaphore | None,
    ) -> Callable[[], Any]:
        """Create an async factory for SubagentManager that respects the semaphore."""

        async def _factory() -> "RunResult":
            if semaphore:
                async with semaphore:
                    return await self.body.arun(item, context=context)
            return await self.body.arun(item, context=context)

        return _factory

    def __repr__(self) -> str:
        conc = f", concurrency={self.concurrency}" if self.concurrency else ""
        return f"MapNode(over={self.over!r}{conc})"


class RouterNode:
    """Choose downstream node(s) from declared candidates (Req 11.4–11.6).

    RouterNode is itself a Runnable. It uses a ``chooser`` (either a Runnable
    or a plain Callable) to select one or more node_ids from the declared
    ``choices``. The selection is written to SharedState under the key
    ``_router_selection`` so downstream edge conditions can gate on it.

    When ``handoff=True``, the chosen node owns the final output: the
    RouterNode's RunResult metadata records ``"router_handoff": True`` and
    ``"router_selected"`` so the engine knows to use the selected node's
    output as the flow's final output.

    Parameters
    ----------
    chooser:
        A Runnable or Callable that takes the input and returns the selected
        node_id (str) or list of node_ids. If a Callable, it is automatically
        adapted (like FunctionRunnable).
    choices:
        The declared candidate node_ids. The chooser must select from among
        these; selections outside this list raise FlowConfigError.
    handoff:
        When True, the chosen node owns the final output for this path.
    """

    def __init__(
        self,
        chooser: "Runnable | Callable",
        *,
        choices: list[str],
        handoff: bool = False,
    ) -> None:
        from loomable.flow.runnable import FunctionRunnable, Runnable as RunnableProto

        self.choices = choices
        self.handoff = handoff

        # Adapt a plain callable to Runnable if needed
        if isinstance(chooser, RunnableProto):
            self._chooser: "Runnable" = chooser
        elif callable(chooser):
            self._chooser = FunctionRunnable(chooser)
        else:
            raise TypeError(
                f"RouterNode chooser must be a Runnable or Callable, got {type(chooser).__name__}"
            )

    async def arun(
        self, input: Any, *, context: "RunContext | None" = None  # noqa: A002
    ) -> "RunResult":
        """Run the chooser and write the selection to SharedState.

        Returns a RunResult whose metadata records the selected node_id(s)
        and whether handoff is active.
        """
        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, MediaPart, Modality

        # 1. Run the chooser to get the selection
        chooser_result = await self._chooser.arun(input, context=context)

        # 2. Extract the selection from the chooser's output
        selected = self._extract_selection(chooser_result)

        # 2b. Apply Command.update patches before validating selection
        from loomable.flow.command import Command

        cmd = Command.from_metadata(chooser_result.metadata)
        if cmd and cmd.update and context is not None and context.shared_state is not None:
            for key, value in cmd.update.items():
                context.shared_state.write(key, value)
        if cmd and cmd.goto is not None and (
            not chooser_result.metadata or "selection" not in chooser_result.metadata
        ):
            selected = cmd.goto

        # 3. Validate that selection(s) are in declared choices
        self._validate_selection(selected)

        # 4. Build an inspectable route decision (classifier may be soft;
        #    allowed routes remain the declared choices).
        reason = self._extract_reason(chooser_result)
        decision: dict[str, Any] = {
            "selected": selected,
            "choices": list(self.choices),
            "reason": reason,
            "handoff": bool(self.handoff),
        }

        # 5. Write selection + decision to SharedState for edge gating / audit
        if context is not None and context.shared_state is not None:
            context.shared_state.write("_router_selection", selected)
            context.shared_state.write("_route_decision", decision)
            # Preserve the pre-router payload so gated branches receive the
            # original user/upstream input (not the router summary text).
            context.shared_state.write("_route_input", input)

        # 6. Build the RouterNode's RunResult
        if isinstance(selected, list):
            selection_str = ", ".join(selected)
        else:
            selection_str = selected

        summary = f"RouterNode: selected [{selection_str}]"
        if reason:
            summary = f"{summary} ({reason})"
        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=summary.encode("utf-8"),
                )
            ]
        )

        metadata: dict[str, Any] = {
            "router_selected": selected,
            "route_decision": decision,
        }
        if self.handoff:
            metadata["router_handoff"] = True

        return RunResult(
            output=output,
            session_id="",
            metadata=metadata,
        )

    def _extract_selection(self, chooser_result: "RunResult") -> "str | list[str]":
        """Extract the node_id selection from the chooser's RunResult.

        The chooser may return:
        - A RunResult whose output text is the selected node_id or a
          comma-separated list of node_ids.
        - A RunResult whose metadata contains a "selection" key with a
          str or list[str].
        """
        # Check metadata first for structured selection
        if chooser_result.metadata and "selection" in chooser_result.metadata:
            return chooser_result.metadata["selection"]

        # Fall back to extracting from output text
        text = self._output_text(chooser_result)
        # Try splitting by comma for multi-selection
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) == 0:
            return ""
        if len(parts) == 1:
            return parts[0]
        return parts

    @staticmethod
    def _extract_reason(chooser_result: "RunResult") -> str:
        """Extract an optional human-readable reason for the route choice."""
        meta = chooser_result.metadata or {}
        for key in ("reason", "route_reason", "selection_reason"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _output_text(self, result: "RunResult") -> str:
        """Extract plain text from a RunResult's output."""
        if result.output and result.output.parts:
            part = result.output.parts[0]
            if hasattr(part, "data"):
                data = part.data
                if isinstance(data, bytes):
                    return data.decode("utf-8").strip()
                return str(data).strip()
        return ""

    def _validate_selection(self, selected: "str | list[str]") -> None:
        """Ensure the selection(s) are within the declared choices."""
        if isinstance(selected, list):
            invalid = [s for s in selected if s not in self.choices]
        else:
            invalid = [selected] if selected not in self.choices else []

        if invalid:
            raise FlowConfigError(
                f"RouterNode chooser selected {invalid!r} which is not among "
                f"declared choices {self.choices!r}"
            )

    def __repr__(self) -> str:
        handoff_str = ", handoff=True" if self.handoff else ""
        return f"RouterNode(choices={self.choices!r}{handoff_str})"
