"""HierarchicalEngine: manager delegates to workers and synthesizes.

Runs the designated manager node which delegates to worker nodes (as
sub-runnables via SubagentManager) and synthesizes their results. The manager
node is identified by ``Node.manager == True`` (Req 8.4).

Supports checkpoint skip/write so kill/resume works for hierarchical flows.
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
from loomable.persist.checkpoint import Checkpoint


class HierarchicalEngine:
    """Execute a Flow using a manager/worker hierarchy (Req 8.4)."""

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
        pending_decisions: dict[str, str] | None = None,
    ) -> RunResult:
        """Drive the flow using hierarchical delegation.

        ``pending_decisions`` is accepted for ExecutionEngine protocol parity with
        :class:`~loomable.flow.engines.sequential.SequentialEngine` (HITL resume).
        Hierarchical graphs do not pause for confirmation today.
        """
        del pending_decisions  # protocol parity; HITL resume is SequentialEngine-only
        nodes: dict[str, Node] = flow._nodes
        completed: set[str] = set(completed_node_ids or [])

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

        if manager_node is None or manager_id is None:
            raise FlowConfigError(
                "HierarchicalEngine requires exactly one node with "
                "manager=True, but none was found."
            )

        sub_results: dict[str, RunResult] = {}

        # Restore already-completed worker outputs from state when resuming
        for nid in list(worker_nodes):
            if nid in completed and state.get(nid) is not None:
                # Reconstruct a minimal RunResult for sub_results
                out = state.get(nid)
                if isinstance(out, AgentOutput):
                    sub_results[nid] = RunResult(output=out, session_id=session_id or "")

        pending_workers = {
            nid: node for nid, node in worker_nodes.items() if nid not in completed
        }

        if pending_workers and not context.cancelled:
            manager = SubagentManager()
            tasks = [
                DelegatedTask(
                    task_id=nid,
                    task=f"Execute worker node {nid}",
                    context={},
                    agent_factory=self._make_worker_factory(
                        pending_workers[nid], nid, input, context
                    ),
                )
                for nid in sorted(pending_workers.keys())
            ]
            outcomes: list[SubagentOutcome] = await manager.run_all(tasks)
            self._commit_worker_results(outcomes, state, sub_results)
            for nid in pending_workers:
                completed.add(nid)
            if checkpointer is not None:
                await self._write_checkpoint(checkpointer, state, completed, session_id)

        # Manager: skip if already completed on resume or if cancelled
        from loomable.flow.observability import emit_node_start, emit_node_end

        if context.cancelled:
            from loomable.content import AgentOutput, MediaPart, Modality

            manager_result = sub_results.get(manager_id) or RunResult(
                output=AgentOutput(
                    parts=[
                        MediaPart(
                            modality=Modality.TEXT,
                            media_type="text/plain",
                            data=b"",
                        )
                    ]
                ),
                session_id=session_id or "",
            )
        elif manager_id in completed and state.get(manager_id) is not None:
            out = state.get(manager_id)
            if isinstance(out, AgentOutput):
                manager_result = RunResult(output=out, session_id=session_id or "")
            else:
                start_t = emit_node_start(context.events, manager_id)
                try:
                    manager_result = await manager_node.runnable.arun(input, context=context)
                finally:
                    emit_node_end(context.events, manager_id, start_t)
        else:
            start_t = emit_node_start(context.events, manager_id)
            try:
                manager_result = await manager_node.runnable.arun(input, context=context)
            finally:
                emit_node_end(context.events, manager_id, start_t)
            sub_results[manager_id] = manager_result
            state.write(manager_id, manager_result.output)
            self._apply_state_updates(manager_result, state)
            completed.add(manager_id)
            if checkpointer is not None:
                await self._write_checkpoint(checkpointer, state, completed, session_id)

        sub_results[manager_id] = manager_result
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
        if context.cancelled:
            from loomable.agent.context import StopReason

            final.metadata["stop_reason"] = StopReason.CANCELLED
        final.metadata["completed_node_ids"] = sorted(completed)
        # Worker state_updates were already applied during the hierarchy run;
        # don't leak them for a parent engine to re-apply.
        final.metadata.pop("state_updates", None)
        return final

    @staticmethod
    async def _write_checkpoint(
        checkpointer: Any,
        state: SharedState,
        completed: set[str],
        session_id: str | None,
    ) -> None:
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
    def _make_worker_factory(
        node: Node,
        node_id: str,
        initial_input: Any,
        context: RunContext,
    ) -> Any:
        async def _run() -> RunResult:
            from loomable.flow.observability import emit_node_end, emit_node_start

            start_t = emit_node_start(context.events, node_id)
            try:
                return await node.runnable.arun(initial_input, context=context)
            finally:
                emit_node_end(context.events, node_id, start_t)

        return _run

    @staticmethod
    def _apply_state_updates(result: RunResult, state: SharedState) -> None:
        updates = (result.metadata or {}).get("state_updates")
        if isinstance(updates, dict):
            for key, value in updates.items():
                state.write(key, value)
        elif isinstance(getattr(result, "structured", None), dict):
            for key, value in result.structured.items():
                state.write(key, value)

    @staticmethod
    def _commit_worker_results(
        outcomes: list[SubagentOutcome],
        state: SharedState,
        sub_results: dict[str, RunResult],
    ) -> None:
        """Write worker outcomes to state and sub_results (sorted for determinism)."""
        for outcome in sorted(outcomes, key=lambda o: o.task_id):
            nid = outcome.task_id
            if outcome.error is not None:
                err_text = str(outcome.error)
                output = AgentOutput(
                    parts=[
                        MediaPart(
                            modality=Modality.TEXT,
                            media_type="text/plain",
                            data=err_text.encode(),
                        )
                    ]
                )
                sub_results[nid] = RunResult(
                    output=output,
                    session_id="",
                    metadata={"error": err_text},
                )
                continue
            result = outcome.result
            if isinstance(result, RunResult):
                state.write(nid, result.output)
                HierarchicalEngine._apply_state_updates(result, state)
                sub_results[nid] = result
            else:
                output = AgentOutput(
                    parts=[
                        MediaPart(
                            modality=Modality.TEXT,
                            media_type="text/plain",
                            data=str(result).encode(),
                        )
                    ]
                )
                state.write(nid, output)
                sub_results[nid] = RunResult(output=output, session_id="")
