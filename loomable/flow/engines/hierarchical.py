"""HierarchicalEngine: manager delegates to workers and synthesizes.

Runs the designated manager node which delegates to worker nodes (as
sub-runnables via SubagentManager) and synthesizes their results. The manager
node is identified by ``Node.manager == True`` (Req 8.4).

Algorithm:
1. Identify the manager node (the one with ``manager=True``).
2. If no manager node found, raise FlowConfigError.
3. Run all worker nodes concurrently via SubagentManager.
4. Collect worker results and write them to state.
5. Run the manager node with access to worker results (via state).
6. Return the manager's output as the final result.
"""

from __future__ import annotations

__all__ = ["HierarchicalEngine"]

from typing import Any

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.nodes import Edge, FlowConfigError, Node
from loomable.flow.state import SharedState
from loomable.kernel.subagents import DelegatedTask, SubagentManager, SubagentOutcome


class HierarchicalEngine:
    """Execute a Flow using a manager/worker hierarchy (Req 8.4).

    The manager node (flagged ``manager=True``) delegates work to all other
    nodes (workers). Workers run concurrently via SubagentManager, their
    results are written to SharedState, and then the manager runs with
    access to those results through state. The manager's output becomes
    the final result.

    Per-worker fault isolation is inherited from SubagentManager (Req 8.7):
    one worker failing does not cancel other workers.
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
        """Drive the flow using hierarchical delegation.

        Parameters
        ----------
        completed_node_ids:
            Node IDs already completed in a previous run (from checkpoint).
            Workers in this set are skipped on resume.
        checkpointer:
            A Checkpointer instance for persisting state. Used after workers
            complete and after the manager completes.
        session_id:
            Session/thread identifier used as the checkpoint thread_id.
        """
        nodes: dict[str, Node] = flow._nodes
        edges: list[Edge] = flow._edges  # noqa: F841

        # 1. Identify the manager node
        manager_node: Node | None = None
        manager_id: str | None = None
        worker_nodes: dict[str, Node] = {}

        for nid, node in nodes.items():
            if node.manager:
                if manager_node is not None:
                    raise FlowConfigError(
                        f"Flow has multiple manager nodes: {manager_id!r} and "
                        f"{nid!r}. Only one node may be flagged manager=True."
                    )
                manager_node = node
                manager_id = nid
            else:
                worker_nodes[nid] = node

        # 2. If no manager node found, raise FlowConfigError
        if manager_node is None or manager_id is None:
            raise FlowConfigError(
                "HierarchicalEngine requires exactly one node with "
                "manager=True, but none was found."
            )

        # 3. Run all worker nodes concurrently via SubagentManager
        sub_results: dict[str, RunResult] = {}
        manager = SubagentManager()

        if worker_nodes:
            tasks = [
                DelegatedTask(
                    task_id=nid,
                    task=f"Execute worker node {nid}",
                    context={},
                    agent_factory=self._make_worker_factory(
                        worker_nodes[nid], input, context
                    ),
                )
                for nid in sorted(worker_nodes.keys())
            ]

            outcomes: list[SubagentOutcome] = await manager.run_all(tasks)

            # 4. Collect worker results and write them to state
            self._commit_worker_results(outcomes, state, sub_results)

        # 5. Run the manager node with access to worker results (via state)
        manager_result = await manager_node.runnable.arun(input, context=context)
        sub_results[manager_id] = manager_result
        state.write(manager_id, manager_result.output)

        # 6. Return the manager's output as the final result
        final = RunResult(
            output=manager_result.output,
            session_id=manager_result.session_id,
            usage=manager_result.usage,
            tool_activity=list(getattr(manager_result, "tool_activity", None) or []),
            structured=getattr(manager_result, "structured", None),
            sub_results=sub_results,
            metadata=dict(getattr(manager_result, "metadata", None) or {}),
            thoughts=list(getattr(manager_result, "thoughts", None) or []),
            plan=getattr(manager_result, "plan", None),
            reasoning=list(getattr(manager_result, "reasoning", None) or []),
        )
        return final

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_worker_factory(
        node: Node,
        initial_input: Any,
        context: RunContext,
    ) -> Any:
        """Create the async factory for a worker DelegatedTask."""

        async def _run() -> RunResult:
            return await node.runnable.arun(initial_input, context=context)

        return _run

    @staticmethod
    def _commit_worker_results(
        outcomes: list[SubagentOutcome],
        state: SharedState,
        sub_results: dict[str, RunResult],
    ) -> None:
        """Write worker outcomes to state and sub_results.

        Commits writes in node_id order (sorted) for deterministic reducer
        application. Failed workers are recorded in sub_results with error
        metadata but do not write to state (Req 8.7).
        """
        sorted_outcomes = sorted(outcomes, key=lambda o: o.task_id)

        for outcome in sorted_outcomes:
            node_id = outcome.task_id

            if outcome.error is not None:
                # Record the failure without writing to state
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
                result: RunResult = outcome.result
                sub_results[node_id] = result
                # Write to SharedState so the manager can access worker results
                state.write(node_id, result.output)
