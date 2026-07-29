"""loomable.kernel.agent_loop - The perceive→plan→act→observe agent loop.

Implements the AgentLoop which wires together the Context Manager, Planner,
Guardrail Harness, Verification Gate, Tool Runtime, Memory Manager, Summarizer,
and Session Store into a single repeating cycle.

The key invariant is that the loop phase ordering is always
[perceive, plan, act, observe] repeated, and that LoopState is persisted
after each step for resumability.

Requirements covered: 18.1, 18.5
"""

from __future__ import annotations

from loomable.kernel.context import ContextManager
from loomable.kernel.guardrails import GuardrailHarness, VerificationGate
from loomable.kernel.memory import MemoryManager
from loomable.kernel.models import (
    AgentConfig,
    ContextItem,
    LoopPhase,
    LoopState,
    Session,
    ToolCall,
    Turn,
)
from loomable.kernel.planner import ExecutionPlan, Planner, TaskContext
from loomable.kernel.stores import SessionStore
from loomable.kernel.summarizer import Summarizer
from loomable.kernel.tool_runtime import ToolRuntime


#: The fixed ordering of loop phases.
PHASE_ORDER: list[LoopPhase] = ["perceive", "plan", "act", "observe"]


class AgentLoop:
    """The perceive→plan→act→observe agent loop.

    Wires together all kernel subsystems into a single repeating cycle.
    Persists a LoopState snapshot after each step so an interrupted loop
    can resume from the last completed step and phase.

    Parameters
    ----------
    config:
        The agent configuration.
    context_manager:
        Manages the context window (assemble, admit, token accounting).
    planner:
        Produces execution plans from task context.
    harness:
        Evaluates guardrail rules against proposed tool actions.
    gate:
        Configurable per-step verification gates.
    tool_runtime:
        Dispatches tool calls concurrently.
    memory:
        Coordinates multi-tier memory (L1, L2, L3).
    summarizer:
        Produces checkpoint summaries at configured intervals.
    session_store:
        Persists session and loop state to storage.
    session:
        The active session for this loop.
    """

    def __init__(
        self,
        config: AgentConfig,
        context_manager: ContextManager,
        planner: Planner,
        harness: GuardrailHarness,
        gate: VerificationGate,
        tool_runtime: ToolRuntime,
        memory: MemoryManager,
        summarizer: Summarizer,
        session_store: SessionStore,
        session: Session,
    ) -> None:
        self._config = config
        self._context_manager = context_manager
        self._planner = planner
        self._harness = harness
        self._gate = gate
        self._tool_runtime = tool_runtime
        self._memory = memory
        self._summarizer = summarizer
        self._session_store = session_store
        self._session = session
        self._stopped = False

        # Initialize loop state at the beginning (perceive phase, step 0)
        self.loop_state = LoopState(
            session_id=session.session_id,
            step=session.step,
            phase="perceive",
        )

    @property
    def session(self) -> Session:
        """The active session."""
        return self._session

    async def step(self) -> None:
        """Execute one full perceive→plan→act→observe cycle.

        1. perceive: assemble context window via ContextManager
        2. plan: invoke Planner to get next action(s)
        3. act: evaluate actions through GuardrailHarness, dispatch via ToolRuntime
        4. observe: record turn in MemoryManager, check summarization, run gate

        After each phase transition, update loop_state.phase.
        After completing the step, persist LoopState via session store.
        """
        # --- PERCEIVE ---
        self.loop_state.phase = "perceive"
        context_window = self._context_manager.assemble()

        # --- PLAN ---
        self.loop_state.phase = "plan"
        task_context = TaskContext(
            task=f"Step {self.loop_state.step}",
            context={"window_items": len(context_window.items)},
        )
        plan: ExecutionPlan = await self._planner.plan(task_context)

        # Extract tool calls from the plan metadata if available,
        # otherwise build from plan steps
        tool_calls = self._extract_tool_calls(plan)
        self.loop_state.pending = tool_calls

        # --- ACT ---
        self.loop_state.phase = "act"
        if tool_calls:
            allowed, violations = self._harness.evaluate(tool_calls)
            outcomes = await self._tool_runtime.dispatch(allowed)
        else:
            allowed = []
            violations = []
            outcomes = []

        # --- OBSERVE ---
        self.loop_state.phase = "observe"

        # Record the turn in memory
        turn_content = self._build_turn_content(plan, outcomes, violations)
        turn = Turn(
            role="assistant",
            content=turn_content,
            tokens=max(1, len(turn_content) // 4),  # rough estimate
            step=self.loop_state.step,
        )
        self._memory.record_turn(turn)

        # Also add to the session L1
        self._session.l1.append(turn)

        # Check if summarization is needed
        if self._summarizer.should_summarize(self.loop_state.step):
            summary = self._summarizer.summarize(self._memory.l1)
            self._memory.add_summary(summary)
            self._session.l2.append(summary)

        # Run verification gate
        gate_passed = self._gate.check(self.loop_state.step, {})
        if not gate_passed:
            self._stopped = True

        # Advance step counter
        self.loop_state.step += 1
        self._session.step = self.loop_state.step
        self.loop_state.pending = []

        # Persist state
        self.persist_state()

    async def run(self, max_steps: int | None = None) -> None:
        """Run the loop for up to max_steps (or until a stop condition).

        Parameters
        ----------
        max_steps:
            Maximum number of steps to execute. If None, runs until
            stopped by a verification gate or other stop condition.
        """
        steps_executed = 0
        while not self._stopped:
            if max_steps is not None and steps_executed >= max_steps:
                break
            await self.step()
            steps_executed += 1

    def persist_state(self) -> None:
        """Persist current loop state to session store.

        Saves the full session (including updated L1/L2 and step counter)
        via the SessionStore.
        """
        self._session_store.save(self._session)

    def resume_from(self, loop_state: LoopState) -> None:
        """Resume from a previously persisted loop state.

        Restores the loop state so that execution continues from
        the last completed step and phase.

        Parameters
        ----------
        loop_state:
            The LoopState snapshot to resume from.
        """
        self.loop_state = loop_state
        self._session.step = loop_state.step
        self._stopped = False

    def _extract_tool_calls(self, plan: ExecutionPlan) -> list[ToolCall]:
        """Extract tool calls from an execution plan.

        If the plan metadata contains 'tool_calls', use those directly.
        Otherwise, return an empty list (the plan is purely informational).
        """
        raw_calls = plan.metadata.get("tool_calls", [])
        if isinstance(raw_calls, list):
            calls: list[ToolCall] = []
            for item in raw_calls:
                if isinstance(item, ToolCall):
                    calls.append(item)
            return calls
        return []

    def _build_turn_content(
        self,
        plan: ExecutionPlan,
        outcomes: list,
        violations: list,
    ) -> str:
        """Build a textual summary of the step for memory recording."""
        parts: list[str] = []
        if plan.steps:
            parts.append(f"Plan: {'; '.join(plan.steps)}")
        if outcomes:
            parts.append(f"Outcomes: {len(outcomes)} tool calls completed")
        if violations:
            parts.append(f"Violations: {len(violations)} actions blocked")
        return " | ".join(parts) if parts else "No-op step"
