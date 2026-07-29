"""loomable.agent.orchestration - Multi-agent orchestration (Req 11).

This module implements :class:`Orchestrator`, the high-level driver that runs a set
of child :class:`~loomable.agent.builder.BuiltAgent` instances according to an
:class:`~loomable.agent.builder.OrchestrationMode`:

- ``PARALLEL`` broadcasts the same :class:`~loomable.content.AgentInput` to every
  sub-agent and runs them **concurrently**, delegating to the kernel
  :class:`~loomable.kernel.subagents.SubagentManager` (``run_all``), which provides
  concurrency, per-child fault isolation, and results keyed to the originating task
  (Req 11.2–11.5, 11.8). One failing child yields a
  :class:`~loomable.kernel.errors.SubagentError` for that child while its siblings
  still return.
- ``ROUTE`` selects exactly one sub-agent (the first by default, or via an injected
  router callable) and runs only that one (Req 11.6).
- ``COORDINATE`` delegates to the sub-agents (reusing the parallel path) and then
  synthesizes their results into a single :class:`~loomable.content.AgentOutput`
  (Req 11.7); when a ``leader`` agent is supplied it performs the synthesis,
  otherwise the child outputs are concatenated.

The kernel is reused unchanged (Req 11.8): this module only composes existing
primitives (``SubagentManager``, ``DelegatedTask``, ``SubagentOutcome``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from loomable.content import AgentInput, AgentOutput, Text
from loomable.kernel.errors import SubagentError
from loomable.kernel.subagents import DelegatedTask, SubagentManager

from .builder import BuiltAgent, OrchestrationMode
from .run import RunResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from loomable.kernel.subagents import SubagentOutcome


#: A router selects the index of the single sub-agent to run in ROUTE mode.
Router = Callable[[list[BuiltAgent], AgentInput], int]


def _subagent_id(agent: BuiltAgent, index: int) -> str:
    """Return a stable id for a sub-agent.

    Prefers an explicit ``name`` attribute when present; otherwise falls back to a
    positional ``subagent-{i}`` id so results remain keyed deterministically.
    """
    name = getattr(agent, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"subagent-{index}"


class Orchestrator:
    """Runs a set of sub-agents according to an :class:`OrchestrationMode` (Req 11)."""

    def __init__(
        self,
        sub_agents: list[BuiltAgent],
        mode: OrchestrationMode,
        leader: BuiltAgent | None = None,
        *,
        router: Router | None = None,
    ) -> None:
        self.sub_agents = sub_agents
        self.mode = mode
        self.leader = leader
        self.router = router

    async def run(self, input: AgentInput) -> RunResult:  # noqa: A002
        """Run the sub-agents for ``input`` per the configured mode (Req 11.2–11.7)."""
        if not self.sub_agents:
            raise ValueError("Orchestrator requires at least one sub-agent.")

        if self.mode is OrchestrationMode.ROUTE:
            return await self._run_route(input)
        if self.mode is OrchestrationMode.COORDINATE:
            return await self._run_coordinate(input)
        # PARALLEL (and any non-SINGLE default) broadcasts to all sub-agents.
        return await self._run_parallel(input)

    # ------------------------------------------------------------------
    # PARALLEL
    # ------------------------------------------------------------------

    async def _run_parallel(self, input: AgentInput) -> RunResult:  # noqa: A002
        """Broadcast ``input`` to every sub-agent concurrently and aggregate (Req 11.2–11.5)."""
        ids = [_subagent_id(sa, i) for i, sa in enumerate(self.sub_agents)]
        tasks = [
            DelegatedTask(
                task_id=ids[i],
                task=f"parallel sub-agent {ids[i]}",
                context={},
                # Bind sa per-iteration so each factory targets its own sub-agent.
                agent_factory=(lambda sa=sa: sa.arun(input)),
            )
            for i, sa in enumerate(self.sub_agents)
        ]

        outcomes = await SubagentManager().run_all(tasks)
        return self._aggregate(outcomes)

    def _aggregate(self, outcomes: list["SubagentOutcome"]) -> RunResult:
        """Build a :class:`RunResult` from ordered sub-agent outcomes.

        - ``sub_results`` maps each sub-agent id to its :class:`RunResult` (success)
          or :class:`SubagentError` (failure) (Req 11.4/11.5).
        - The aggregated :class:`AgentOutput` concatenates the text of successful
          children in sub-agent order.
        - ``usage`` sums the per-child usage; ``session_id`` is the leader's (when
          present) else the first successful child's (else the first child's id).
        """
        sub_results: dict[str, RunResult | SubagentError] = {}
        texts: list[str] = []
        usage: dict[str, int] = {}
        first_session_id: str | None = None

        for outcome in outcomes:
            if outcome.error is not None:
                sub_results[outcome.task_id] = outcome.error
                continue
            result = outcome.result
            sub_results[outcome.task_id] = result
            if first_session_id is None:
                first_session_id = result.session_id
            texts.append(result.output.text())
            for key, value in (result.usage or {}).items():
                usage[key] = usage.get(key, 0) + value

        session_id = self._session_id(first_session_id, outcomes)
        aggregated = AgentOutput(parts=[Text("".join(texts))])
        return RunResult(
            output=aggregated,
            session_id=session_id,
            usage=usage,
            tool_activity=[],
            sub_results=sub_results,
        )

    def _session_id(
        self, first_child_session: str | None, outcomes: list["SubagentOutcome"]
    ) -> str:
        """Resolve the aggregate session id: leader's, else a child's, else a task id."""
        if self.leader is not None:
            return self.leader.session.session_id
        if first_child_session is not None:
            return first_child_session
        return outcomes[0].task_id if outcomes else "orchestrator"

    # ------------------------------------------------------------------
    # ROUTE
    # ------------------------------------------------------------------

    async def _run_route(self, input: AgentInput) -> RunResult:  # noqa: A002
        """Select exactly one sub-agent and run only that one (Req 11.6)."""
        index = 0
        if self.router is not None:
            index = self.router(self.sub_agents, input)
        if not 0 <= index < len(self.sub_agents):
            raise ValueError(
                f"Router selected out-of-range sub-agent index {index} "
                f"(have {len(self.sub_agents)} sub-agents)."
            )

        chosen = self.sub_agents[index]
        result = await chosen.arun(input)
        # Record which sub-agent handled the input for traceability.
        result.sub_results = {_subagent_id(chosen, index): result}
        return result

    # ------------------------------------------------------------------
    # COORDINATE
    # ------------------------------------------------------------------

    async def _run_coordinate(self, input: AgentInput) -> RunResult:  # noqa: A002
        """Delegate to sub-agents then synthesize a single output (Req 11.7)."""
        gathered = await self._run_parallel(input)

        if self.leader is not None:
            # The leader synthesizes over the concatenated child outputs.
            synthesis_input = AgentInput.from_text(gathered.output.text())
            synthesized = await self.leader.arun(synthesis_input)
            synthesized.sub_results = gathered.sub_results
            return synthesized

        # No leader: the concatenated child outputs are the synthesized result.
        return gathered
