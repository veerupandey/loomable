"""SequentialEngine: executes nodes in topological order, one at a time.

The simplest engine: detect cycles, topologically sort, then run each node
in order, writing each result to SharedState. Downstream nodes can read
upstream outputs through the state.

Supports flow-level HITL: when a node has ``require_confirmation=True``, the
engine pauses before executing it, records a ``PendingAction``, checkpoints,
and raises ``FlowPaused``. On resume (via checkpoint restore), a pending
action with status "approved" causes the node to execute, while "rejected"
skips it. Already-completed nodes are never re-run (Req 16.1–16.5).
"""

from __future__ import annotations

__all__ = ["SequentialEngine"]

import uuid
from typing import Any

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.engines.base import detect_cycle, toposort
from loomable.flow.hitl import FlowPaused
from loomable.flow.nodes import Edge, Node
from loomable.flow.state import SharedState
from loomable.persist.checkpoint import PendingAction


class SequentialEngine:
    """Execute a Flow's nodes in topological order, one at a time (Req 8.2).

    For each node:
    1. Check incoming edge conditions — skip the node if any required
       incoming edge condition evaluates falsy.
    2. If the node has ``require_confirmation=True`` and no pending decision
       exists, pause (record PendingAction, checkpoint, raise FlowPaused).
    3. If a pending decision exists for this node: execute on "approved",
       skip on "rejected".
    4. Determine the node's input (from state or the initial flow input).
    5. Run the node's runnable.
    6. Write the result to ``state[node_id]`` (Req 7.1).
    7. Store the per-node RunResult in sub_results.
    8. Write a checkpoint if a checkpointer is configured (Req 13.1).

    The final RunResult uses the output from the last executed node.
    """

    async def run(
        self,
        flow: Any,
        input: Any,  # noqa: A002
        state: "SharedState",
        context: "RunContext",
        *,
        completed_node_ids: set[str] | None = None,
        checkpointer: Any | None = None,
        session_id: str | None = None,
        pending_decisions: dict[str, str] | None = None,
    ) -> "RunResult":
        """Drive the flow sequentially through topological order.

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
        pending_decisions:
            Mapping of node_id → decision ("approved" or "rejected") for nodes
            that were paused awaiting confirmation. When present, the engine
            uses the decision to execute or skip the node without re-pausing
            (Req 16.3).
        """
        nodes: dict[str, Node] = flow._nodes
        edges: list[Edge] = flow._edges

        # 1. Validate: detect cycles (Req 8.6)
        detect_cycle(nodes, edges)

        # 2. Topological sort
        order = toposort(nodes, edges)

        # 3. Build incoming-edge map for condition checking
        incoming_edges: dict[str, list[Edge]] = {nid: [] for nid in nodes}
        for edge in edges:
            incoming_edges[edge.target].append(edge)

        # Track completed nodes (union of pre-existing + newly completed)
        completed: set[str] = set(completed_node_ids) if completed_node_ids else set()

        # Decisions for require_confirmation nodes (from caller on resume)
        decisions: dict[str, str] = dict(pending_decisions) if pending_decisions else {}

        # 4. Execute nodes in order
        sub_results: dict[str, RunResult] = {}
        last_result: RunResult | None = None

        for node_id in order:
            # Skip nodes already completed in a previous run (Req 13.2)
            if node_id in completed:
                continue

            node = nodes[node_id]

            # Check edge conditions: skip if any incoming conditional edge
            # evaluates falsy. Unconditional edges (condition=None) always pass.
            if not self._should_execute(node_id, incoming_edges, state):
                continue

            # --- HITL: require_confirmation gate (Req 16.1–16.5) ---
            if node.require_confirmation:
                decision = decisions.get(node_id)
                if decision is None:
                    # No decision yet — pause the flow
                    pending = PendingAction(
                        tool_name=node_id,
                        call_id=uuid.uuid4().hex,
                        args={"node_id": node_id},
                        status="pending",
                    )
                    thread_id = session_id or "default"
                    # Checkpoint before raising (Req 16.2)
                    if checkpointer is not None:
                        await self._write_hitl_checkpoint(
                            checkpointer, state, completed, thread_id, pending
                        )
                    raise FlowPaused(pending=pending, thread_id=thread_id)
                elif decision == "rejected":
                    # Skip the node on rejection (Req 16.3)
                    completed.add(node_id)
                    if checkpointer is not None:
                        await self._write_checkpoint(
                            checkpointer, state, completed, session_id
                        )
                    continue
                # else: decision == "approved" → fall through and execute

            # Determine input for this node:
            # - If there are upstream nodes with results in state, use the
            #   output from the most recent upstream node (by topo order).
            # - Otherwise (first node or no upstream results), use the
            #   initial flow input.
            node_input = self._resolve_input(
                node_id, incoming_edges, state, input, order
            )

            # Emit node_start event (Req 13.3)
            from loomable.flow.observability import emit_node_start, emit_node_end

            start_t = emit_node_start(context.events, node_id)

            # Run the node's runnable
            result = await node.runnable.arun(node_input, context=context)

            # Emit node_end event (Req 13.3)
            emit_node_end(context.events, node_id, start_t)

            # Write output to state[node_id] (Req 7.1)
            state.write(node_id, result.output)

            # Store in sub_results
            sub_results[node_id] = result
            last_result = result

            # Mark as completed and write checkpoint (Req 13.1)
            completed.add(node_id)
            if checkpointer is not None:
                await self._write_checkpoint(
                    checkpointer, state, completed, session_id
                )

        # 5. Assemble final RunResult
        if last_result is None:
            # Edge case: all nodes were skipped
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

        # Return with sub_results attached
        final = RunResult(
            output=last_result.output,
            session_id=last_result.session_id,
            usage=last_result.usage,
            sub_results=sub_results,
            metadata=last_result.metadata,
        )
        return final

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _write_checkpoint(
        checkpointer: Any,
        state: "SharedState",
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

    @staticmethod
    async def _write_hitl_checkpoint(
        checkpointer: Any,
        state: "SharedState",
        completed: set[str],
        thread_id: str,
        pending: "PendingAction",
    ) -> None:
        """Write a checkpoint with HITL pending action before pausing (Req 16.2).

        This checkpoint captures the flow state *before* the confirmation node
        runs, so the process may exit safely and later resume with a decision.
        """
        from loomable.persist.checkpoint import Checkpoint

        cp = Checkpoint(
            thread_id=thread_id,
            step=len(completed),
            session_state={
                "shared_state": state.snapshot(),
                "completed_node_ids": sorted(completed),
            },
            complete=False,
            pending=[pending],
        )
        await checkpointer.put(cp)

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
        - All incoming edges with conditions evaluate truthy, OR
        - It has at least one unconditional incoming edge (always reachable).

        If ALL incoming edges have conditions and ALL evaluate falsy, the
        node is skipped.
        """
        edges = incoming_edges[node_id]
        if not edges:
            # No incoming edges — always execute (root node)
            return True

        # Check: if there's any unconditional edge, always execute
        has_unconditional = any(e.condition is None for e in edges)
        if has_unconditional:
            return True

        # All edges are conditional — execute if at least one is truthy
        for edge in edges:
            if edge.condition is not None and edge.condition(state):
                return True

        return False

    @staticmethod
    def _resolve_input(
        node_id: str,
        incoming_edges: dict[str, list[Edge]],
        state: SharedState,
        initial_input: Any,
        order: list[str],
    ) -> Any:
        """Resolve the input for a node.

        Strategy:
        - Look at this node's incoming edges. If any upstream node has
          produced output (stored in state), use the output from the
          most recent predecessor (by topological order).
        - If no predecessor has output (first node), use the initial input.
        """
        edges = incoming_edges[node_id]
        if not edges:
            return initial_input

        # Find the latest predecessor (by topo order) that has a result
        predecessors = [e.source for e in edges]
        # Sort by their position in the topological order (latest first)
        predecessors_ordered = sorted(
            predecessors, key=lambda nid: order.index(nid), reverse=True
        )

        for pred_id in predecessors_ordered:
            value = state.get(pred_id)
            if value is not None:
                return value

        # No predecessor produced output yet — use initial input
        return initial_input
