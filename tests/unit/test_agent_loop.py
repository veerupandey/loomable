"""Unit tests for the AgentLoop.

Tests verify:
- The loop phase ordering is always [perceive, plan, act, observe]
- LoopState is persisted after each step
- Resumability from a previously persisted LoopState
- The run() method respects max_steps
- Verification gate can stop the loop
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from loomable.kernel.agent_loop import AgentLoop, PHASE_ORDER
from loomable.kernel.context import ContextManager
from loomable.kernel.guardrails import GuardrailHarness, VerificationGate
from loomable.kernel.memory import MemoryManager
from loomable.kernel.models import (
    AgentConfig,
    ContextItem,
    ContextWindow,
    LoopState,
    Session,
    ToolCall,
    ToolOutcome,
    ToolResult,
)
from loomable.kernel.planner import ExecutionPlan, Planner, TaskContext
from loomable.kernel.stores import SessionStore
from loomable.kernel.summarizer import Summarizer
from loomable.kernel.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_config() -> AgentConfig:
    """Create a minimal AgentConfig for testing."""
    return AgentConfig(
        model={"provider": "test"},
        planning_model=None,
        tiers={},
        tier_policy=None,
        fallback_tiers={},
        token_budget=4096,
        checkpoint_interval=5,
    )


def _make_session(session_id: str = "test-session") -> Session:
    """Create a minimal Session for testing."""
    return Session(
        session_id=session_id,
        agent_config_ref="test-config",
    )


def _make_agent_loop(
    *,
    gate_passes: bool = True,
    plan_steps: list[str] | None = None,
    plan_tool_calls: list[ToolCall] | None = None,
) -> tuple[AgentLoop, dict]:
    """Create an AgentLoop with mock subsystems and return both.

    Returns the loop and a dict of the mock subsystems for assertions.
    """
    config = _make_config()
    session = _make_session()

    # Context Manager mock
    context_manager = MagicMock(spec=ContextManager)
    context_manager.assemble.return_value = ContextWindow(
        items=[ContextItem(kind="system", tokens=100, priority=100, pinned=True)]
    )

    # Planner mock
    planner = MagicMock(spec=Planner)
    plan = ExecutionPlan(
        steps=plan_steps or ["step1", "step2"],
        metadata={"tool_calls": plan_tool_calls or []},
    )
    planner.plan = AsyncMock(return_value=plan)

    # Guardrail Harness mock
    harness = MagicMock(spec=GuardrailHarness)
    harness.evaluate.return_value = (plan_tool_calls or [], [])

    # Verification Gate mock
    gate = MagicMock(spec=VerificationGate)
    gate.check.return_value = gate_passes

    # Tool Runtime mock
    tool_runtime = MagicMock(spec=ToolRuntime)
    tool_runtime.dispatch = AsyncMock(return_value=[])

    # Memory Manager mock
    memory = MagicMock(spec=MemoryManager)
    memory.l1 = []
    memory.l2 = []

    # Summarizer mock
    summarizer = MagicMock(spec=Summarizer)
    summarizer.should_summarize.return_value = False

    # Session Store mock
    session_store = MagicMock(spec=SessionStore)

    loop = AgentLoop(
        config=config,
        context_manager=context_manager,
        planner=planner,
        harness=harness,
        gate=gate,
        tool_runtime=tool_runtime,
        memory=memory,
        summarizer=summarizer,
        session_store=session_store,
        session=session,
    )

    mocks = {
        "context_manager": context_manager,
        "planner": planner,
        "harness": harness,
        "gate": gate,
        "tool_runtime": tool_runtime,
        "memory": memory,
        "summarizer": summarizer,
        "session_store": session_store,
    }

    return loop, mocks


# ---------------------------------------------------------------------------
# Tests: Phase ordering
# ---------------------------------------------------------------------------


class TestPhaseOrdering:
    """Tests that verify the loop phase ordering invariant."""

    async def test_phase_order_constant(self):
        """PHASE_ORDER should be perceive, plan, act, observe."""
        assert PHASE_ORDER == ["perceive", "plan", "act", "observe"]

    async def test_step_visits_all_phases_in_order(self):
        """A single step() should visit all four phases in order."""
        loop, mocks = _make_agent_loop()

        phases_visited: list[str] = []
        original_phase_setter = LoopState.__setattr__

        # Track phase transitions by patching the loop_state's phase attribute
        class PhaseTracker:
            def __init__(self, loop_state: LoopState):
                self._loop_state = loop_state

            def __setattr__(self, name, value):
                if name == "phase":
                    phases_visited.append(value)
                super().__setattr__(name, value)

        # Instead of complex patching, observe via a side effect on assemble
        phase_log: list[str] = []

        def log_phase_on_assemble():
            phase_log.append(loop.loop_state.phase)
            return ContextWindow(items=[])

        mocks["context_manager"].assemble.side_effect = log_phase_on_assemble

        async def log_phase_on_plan(task_context):
            phase_log.append(loop.loop_state.phase)
            return ExecutionPlan(steps=["s1"], metadata={"tool_calls": []})

        mocks["planner"].plan = log_phase_on_plan

        def log_phase_on_evaluate(actions):
            phase_log.append(loop.loop_state.phase)
            return ([], [])

        mocks["harness"].evaluate.side_effect = log_phase_on_evaluate

        def log_phase_on_record(turn):
            phase_log.append(loop.loop_state.phase)

        mocks["memory"].record_turn.side_effect = log_phase_on_record

        await loop.step()

        # Each subsystem call should have seen its corresponding phase
        assert phase_log[0] == "perceive"  # assemble called during perceive
        assert phase_log[1] == "plan"  # planner.plan called during plan
        # harness.evaluate not called when no tool calls
        # memory.record_turn called during observe
        assert phase_log[-1] == "observe"

    async def test_step_ends_in_observe_phase(self):
        """After a step completes, the loop_state phase should be observe."""
        loop, _ = _make_agent_loop()
        # Before step, phase is perceive (initial)
        assert loop.loop_state.phase == "perceive"

        await loop.step()

        # After step completes, phase should be observe (last phase in cycle)
        assert loop.loop_state.phase == "observe"

    async def test_multiple_steps_maintain_phase_ordering(self):
        """Multiple steps should each follow the same phase ordering."""
        loop, mocks = _make_agent_loop()

        # Track phases across multiple steps
        all_phases: list[str] = []

        original_assemble = mocks["context_manager"].assemble

        def track_on_assemble():
            all_phases.append(f"step{loop.loop_state.step}:perceive")
            return ContextWindow(items=[])

        mocks["context_manager"].assemble.side_effect = track_on_assemble

        def track_on_record(turn):
            all_phases.append(f"step{loop.loop_state.step}:observe")

        mocks["memory"].record_turn.side_effect = track_on_record

        await loop.step()
        await loop.step()
        await loop.step()

        # We should have 3 perceive entries and 3 observe entries
        perceive_entries = [p for p in all_phases if "perceive" in p]
        observe_entries = [p for p in all_phases if "observe" in p]
        assert len(perceive_entries) == 3
        assert len(observe_entries) == 3


# ---------------------------------------------------------------------------
# Tests: State persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    """Tests that verify LoopState is persisted after each step."""

    async def test_persist_called_after_step(self):
        """session_store.save should be called after each step."""
        loop, mocks = _make_agent_loop()

        await loop.step()

        mocks["session_store"].save.assert_called_once()

    async def test_persist_called_after_every_step(self):
        """session_store.save should be called after every step in run()."""
        loop, mocks = _make_agent_loop()

        await loop.run(max_steps=3)

        assert mocks["session_store"].save.call_count == 3

    async def test_step_counter_increments(self):
        """loop_state.step should increment after each step."""
        loop, _ = _make_agent_loop()

        assert loop.loop_state.step == 0
        await loop.step()
        assert loop.loop_state.step == 1
        await loop.step()
        assert loop.loop_state.step == 2

    async def test_session_step_synced_with_loop_state(self):
        """session.step should be synced with loop_state.step after each step."""
        loop, _ = _make_agent_loop()

        await loop.step()
        assert loop.session.step == loop.loop_state.step

        await loop.step()
        assert loop.session.step == loop.loop_state.step

    async def test_persist_state_saves_session(self):
        """persist_state() should call session_store.save with current session."""
        loop, mocks = _make_agent_loop()

        loop.persist_state()

        mocks["session_store"].save.assert_called_once_with(loop.session)


# ---------------------------------------------------------------------------
# Tests: Resumability
# ---------------------------------------------------------------------------


class TestResumability:
    """Tests that verify resume_from restores loop state correctly."""

    async def test_resume_from_restores_step(self):
        """resume_from should restore the step counter."""
        loop, _ = _make_agent_loop()

        saved_state = LoopState(
            session_id="test-session",
            step=5,
            phase="act",
        )

        loop.resume_from(saved_state)

        assert loop.loop_state.step == 5
        assert loop.loop_state.phase == "act"
        assert loop.session.step == 5

    async def test_resume_from_clears_stopped_flag(self):
        """resume_from should clear the stopped flag so the loop can continue."""
        loop, _ = _make_agent_loop(gate_passes=False)

        # Run one step — gate fails, loop stops
        await loop.step()
        assert loop._stopped is True

        # Resume
        saved_state = LoopState(
            session_id="test-session",
            step=1,
            phase="perceive",
        )
        loop.resume_from(saved_state)
        assert loop._stopped is False

    async def test_resume_from_preserves_pending_calls(self):
        """resume_from should preserve any pending tool calls in the state."""
        loop, _ = _make_agent_loop()

        pending = [ToolCall(id="tc-1", tool_name="test_tool", args={"x": 1})]
        saved_state = LoopState(
            session_id="test-session",
            step=3,
            phase="act",
            pending=pending,
        )

        loop.resume_from(saved_state)

        assert loop.loop_state.pending == pending


# ---------------------------------------------------------------------------
# Tests: Run behavior
# ---------------------------------------------------------------------------


class TestRunBehavior:
    """Tests for the run() method."""

    async def test_run_respects_max_steps(self):
        """run(max_steps=N) should execute exactly N steps."""
        loop, mocks = _make_agent_loop()

        await loop.run(max_steps=5)

        assert loop.loop_state.step == 5
        assert mocks["session_store"].save.call_count == 5

    async def test_run_stops_on_gate_failure(self):
        """run() should stop when a verification gate fails."""
        loop, mocks = _make_agent_loop(gate_passes=False)

        await loop.run(max_steps=10)

        # Should have stopped after first step because gate failed
        assert loop.loop_state.step == 1
        assert mocks["session_store"].save.call_count == 1

    async def test_run_without_max_steps_runs_until_stopped(self):
        """run() without max_steps should run until stopped."""
        loop, mocks = _make_agent_loop()

        # Make gate fail after 3 steps
        call_count = {"n": 0}

        def gate_check(step, context):
            call_count["n"] += 1
            return call_count["n"] < 3

        mocks["gate"].check.side_effect = gate_check

        await loop.run()

        # Ran 3 steps: gates returned True, True, False
        assert loop.loop_state.step == 3


# ---------------------------------------------------------------------------
# Tests: Subsystem wiring
# ---------------------------------------------------------------------------


class TestSubsystemWiring:
    """Tests that verify correct subsystem interactions."""

    async def test_context_manager_assemble_called(self):
        """step() should call context_manager.assemble() in perceive phase."""
        loop, mocks = _make_agent_loop()

        await loop.step()

        mocks["context_manager"].assemble.assert_called_once()

    async def test_planner_plan_called(self):
        """step() should call planner.plan() in plan phase."""
        loop, mocks = _make_agent_loop()

        await loop.step()

        mocks["planner"].plan.assert_called_once()

    async def test_tool_runtime_dispatch_called_with_allowed_actions(self):
        """step() should dispatch only allowed actions from guardrail harness."""
        tool_calls = [
            ToolCall(id="tc-1", tool_name="safe_tool", args={}),
            ToolCall(id="tc-2", tool_name="blocked_tool", args={}),
        ]
        allowed = [tool_calls[0]]

        loop, mocks = _make_agent_loop(plan_tool_calls=tool_calls)
        mocks["harness"].evaluate.return_value = (allowed, [])
        mocks["tool_runtime"].dispatch = AsyncMock(
            return_value=[ToolOutcome(call_id="tc-1", result=ToolResult(content="ok"))]
        )

        await loop.step()

        mocks["tool_runtime"].dispatch.assert_called_once_with(allowed)

    async def test_memory_record_turn_called(self):
        """step() should record a turn in memory during observe phase."""
        loop, mocks = _make_agent_loop()

        await loop.step()

        mocks["memory"].record_turn.assert_called_once()

    async def test_summarizer_checked_each_step(self):
        """step() should check if summarization is needed each step."""
        loop, mocks = _make_agent_loop()

        await loop.step()

        mocks["summarizer"].should_summarize.assert_called_once()

    async def test_verification_gate_checked_each_step(self):
        """step() should check the verification gate each step."""
        loop, mocks = _make_agent_loop()

        await loop.step()

        mocks["gate"].check.assert_called_once()
