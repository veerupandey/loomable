"""ParallelEngine: Bulk Synchronous Parallel (BSP) superstep execution.

Runs all dependency-satisfied nodes concurrently within each superstep and
commits writes at the barrier before the next superstep. Built on the kernel
SubagentManager for concurrency and per-node fault isolation (Req 8.3, 8.7, 17.4).

The barrier applies buffered writes in node_id order so that concurrent writes
to the same key are merged deterministically via the key's reducer (Req 7.2).
"""

from __future__ import annotations

__all__ = ["ParallelEngine"]

from typing import Any

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.engines.base import detect_cycle, level_sets
from loomable.flow.nodes import Edge, Node
from loomable.flow.state import SharedState
from loomable.kernel.subagents import DelegatedTask, SubagentManager, SubagentOutcome


def _apply_state_updates(result: RunResult, state: SharedState) -> None:
    """Merge planner/callable state_updates (and structured dict) into SharedState."""
    updates = (result.metadata or {}).get("state_updates")
    if isinstance(updates, dict):
        for key, value in updates.items():
            state.write(key, value)
    elif isinstance(getattr(result, "structured", None), dict):
        for key, value in result.structured.items():
            state.write(key, value)


class ParallelEngine:
    """Execute a Flow using Bulk Synchronous Parallel supersteps (Req 8.3).

    Each superstep runs all dependency-satisfied nodes concurrently via
    SubagentManager.run_all. After all nodes in a superstep complete,
    writes are committed at the barrier in node_id order (deterministic
    reducer application per Req 7.2). Per-node fault isolation is inherited
    from SubagentManager (Req 8.7): one node failing does not cancel siblings.

    Algorithm:
    1. Detect cycles (Req 8.6).
    2. Compute level_sets — each level is one superstep.
    3. For each superstep:
       a. Filter nodes whose incoming edge conditions are satisfied.
       b. Run ready nodes concurrently via SubagentManager.run_all.
       c. Collect outcomes; buffer writes.
       d. Barrier: apply buffered writes to SharedState in node_id order.
    4. Assemble and return the final RunResult.
    """

    async def run(
        self,
        flow: Any,
        input: Any,  # noqa: A002
        state: SharedState,
        context: RunContext,
        *,
        completed_node_ids: set[str] | None = None,
        checkpointer: Any | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Drive the flow through BSP supersteps.

        Parameters
        ----------
        completed_node_ids:
            Node IDs already completed in a previous run (from checkpoint).
            These nodes are skipped on resume (Req 13.2).
        checkpointer:
            A Checkpointer instance for persisting state at engine boundaries
            (Req 13.1). None means no persistence overhead.
        session_id:
            Session/thread identifier used as the checkpoint thread_id.
        """
        nodes: dict[str, Node] = flow._nodes
        edges: list[Edge] = flow._edges

        # 1. Validate: detect cycles (Req 8.6)
        detect_cycle(nodes, edges)

        # 2. Compute level sets (supersteps)
        levels = level_sets(nodes, edges)

        # 3. Build incoming-edge map for condition checking
        incoming_edges: dict[str, list[Edge]] = {nid: [] for nid in nodes}
        for edge in edges:
            incoming_edges[edge.target].append(edge)

        # Track completed nodes (union of pre-existing + newly completed)
        completed: set[str] = set(completed_node_ids) if completed_node_ids else set()

        # 4. Execute supersteps
        sub_results: dict[str, RunResult] = {}
        last_result: RunResult | None = None
        manager = SubagentManager()

        for level in levels:
            if context.cancelled:
                break

            # 4a. Filter: skip completed nodes and check edge conditions
            ready = [
                nid for nid in level
                if nid not in completed
                and self._should_execute(nid, incoming_edges, state)
            ]

            if not ready:
                continue

            # 4b. Build delegated tasks for concurrent execution
            # NODE_* events fire inside each factory so duration is per-node.
            tasks = [
                DelegatedTask(
                    task_id=nid,
                    task=f"Execute node {nid}",
                    context={},
                    agent_factory=self._make_factory(
                        nodes[nid], nid, incoming_edges, state, input, context
                    ),
                )
                for nid in ready
            ]

            # 4c. Run all concurrently via SubagentManager (fault-isolated)
            outcomes: list[SubagentOutcome] = await manager.run_all(tasks)

            # 4d. Barrier first: durable sibling writes must land even when a
            # hard-stop policy later aborts the graph.
            self._barrier_commit(outcomes, state, sub_results)

            # Track last successful result for the final output
            for nid in sorted(ready):
                if nid in sub_results and sub_results[nid].output is not None:
                    meta = getattr(sub_results[nid], "metadata", None) or {}
                    if "error" not in meta:
                        last_result = sub_results[nid]

            # Mark newly completed (successful) nodes and checkpoint
            for outcome in outcomes:
                if outcome.error is None:
                    completed.add(outcome.task_id)
            if checkpointer is not None:
                await self._write_checkpoint(
                    checkpointer, state, completed, session_id
                )

            # Hard-stop policies (Step.on_failure="stop") halt after siblings
            # are committed — failure stays local until the barrier, then escalates.
            for outcome in outcomes:
                err = outcome.error
                if err is None:
                    continue
                from loomable.flow.step import StepFailed

                if isinstance(err, StepFailed):
                    raise err
                cause = getattr(err, "__cause__", None)
                if isinstance(cause, StepFailed):
                    raise cause

        # 5. Assemble final RunResult
        if last_result is None:
            # Edge case: all nodes were skipped or failed
            output = AgentOutput(
                parts=[
                    MediaPart(
                        modality=Modality.TEXT,
                        media_type="text/plain",
                        data=b"",
                    )
                ]
            )
            last_result = RunResult(output=output, session_id="")

        final = RunResult(
            output=last_result.output,
            session_id=last_result.session_id,
            usage=last_result.usage,
            tool_activity=list(getattr(last_result, "tool_activity", None) or []),
            structured=getattr(last_result, "structured", None),
            sub_results=sub_results,
            metadata=dict(getattr(last_result, "metadata", None) or {}),
            thoughts=list(getattr(last_result, "thoughts", None) or []),
            plan=getattr(last_result, "plan", None),
            reasoning=list(getattr(last_result, "reasoning", None) or []),
        )
        if context.cancelled:
            from loomable.agent.context import StopReason

            final.metadata["stop_reason"] = StopReason.CANCELLED
        final.metadata["completed_node_ids"] = sorted(completed)
        return final

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_execute(
        node_id: str,
        incoming_edges: dict[str, list[Edge]],
        state: SharedState,
    ) -> bool:
        """Determine whether a node should execute based on edge conditions.

        A node executes if:
        - It has no incoming edges (root node), OR
        - It has at least one unconditional incoming edge (always reachable), OR
        - At least one conditional incoming edge evaluates truthy.

        If ALL incoming edges have conditions and ALL evaluate falsy, the
        node is skipped.
        """
        edges = incoming_edges[node_id]
        if not edges:
            return True

        has_unconditional = any(e.condition is None for e in edges)
        if has_unconditional:
            return True

        # All edges are conditional — execute if at least one is truthy
        for edge in edges:
            if edge.condition is not None and edge.condition(state):
                return True

        return False

    @staticmethod
    def _make_factory(
        node: Node,
        node_id: str,
        incoming_edges: dict[str, list[Edge]],
        state: SharedState,
        initial_input: Any,
        context: RunContext,
    ) -> Any:
        """Create the async factory for a DelegatedTask.

        The factory resolves the node's input and runs its runnable.
        Returns a callable that produces an awaitable.
        """
        # Resolve input: use upstream node output from state if available,
        # otherwise use initial flow input.
        node_input = ParallelEngine._resolve_input(
            node_id, incoming_edges, state, initial_input
        )

        async def _run() -> RunResult:
            from loomable.flow.observability import emit_node_end, emit_node_start

            start_t = emit_node_start(context.events, node_id)
            try:
                return await node.runnable.arun(node_input, context=context)
            finally:
                emit_node_end(context.events, node_id, start_t)

        return _run

    @staticmethod
    def _resolve_input(
        node_id: str,
        incoming_edges: dict[str, list[Edge]],
        state: SharedState,
        initial_input: Any,
    ) -> Any:
        """Resolve the input for a node.

        Prefer an incoming edge's ``payload_key`` (data contract). Otherwise
        look at upstream outputs in state (sorted by node_id for determinism).
        Fall back to the initial input.
        """
        edges = incoming_edges[node_id]
        if not edges:
            return initial_input

        for edge in edges:
            if edge.payload_key:
                value = state.get(edge.payload_key)
                if value is not None:
                    return value

        # Check predecessors sorted by node_id for determinism
        predecessors = sorted(set(e.source for e in edges))
        for pred_id in predecessors:
            value = state.get(pred_id)
            if value is not None:
                return value

        return initial_input

    @staticmethod
    def _barrier_commit(
        outcomes: list[SubagentOutcome],
        state: SharedState,
        sub_results: dict[str, RunResult],
    ) -> None:
        """Apply superstep results to state at the barrier.

        Commits writes in node_id order (sorted) so that concurrent writes
        to the same key are merged deterministically via reducers (Req 7.2).

        Failed nodes are recorded in sub_results with error metadata but
        do not write to state.
        """
        # Sort outcomes by task_id (= node_id) for deterministic ordering
        sorted_outcomes = sorted(outcomes, key=lambda o: o.task_id)

        for outcome in sorted_outcomes:
            node_id = outcome.task_id

            if outcome.error is not None:
                # Record the failure in sub_results without writing to state.
                # Build a RunResult that carries the error in metadata.
                error_output = AgentOutput(
                    parts=[
                        MediaPart(
                            modality=Modality.TEXT,
                            media_type="text/plain",
                            data=f"Error: {outcome.error}".encode("utf-8"),
                        )
                    ]
                )
                sub_results[node_id] = RunResult(
                    output=error_output,
                    session_id="",
                    metadata={"error": str(outcome.error)},
                )
            else:
                # Successful outcome — the result is a RunResult from node.arun
                result: RunResult = outcome.result
                sub_results[node_id] = result

                # Write to SharedState (applies the key's reducer)
                state.write(node_id, result.output)
                _apply_state_updates(result, state)

    # ------------------------------------------------------------------
    # Checkpoint helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _write_checkpoint(
        checkpointer: Any,
        state: SharedState,
        completed: set[str],
        session_id: str | None,
    ) -> None:
        """Write a checkpoint with the current state snapshot and completed nodes."""
        from loomable.persist.checkpoint import Checkpoint

        cp = Checkpoint(
            thread_id=session_id or "default",
            step=len(completed),
            session_state={
                "shared_state": state.snapshot(),
                "completed_node_ids": sorted(completed),
            },
            complete=False,
        )
        await checkpointer.put(cp)
