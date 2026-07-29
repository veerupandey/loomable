"""Unit tests for loomable.kernel.subagents — SubagentManager."""

from __future__ import annotations

import asyncio

import pytest

from loomable.kernel.errors import SubagentError
from loomable.kernel.subagents import DelegatedTask, SubagentManager, SubagentOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_task(task_id: str, result: object = "ok", delay: float = 0.0) -> DelegatedTask:
    """Create a DelegatedTask whose factory returns `result` after `delay`."""

    async def _work():
        if delay > 0:
            await asyncio.sleep(delay)
        return result

    return DelegatedTask(
        task_id=task_id,
        task=f"task-{task_id}",
        context={},
        agent_factory=_work,
    )


def make_failing_task(task_id: str, exc: Exception | None = None) -> DelegatedTask:
    """Create a DelegatedTask whose factory raises an exception."""

    async def _work():
        raise (exc or RuntimeError(f"subagent {task_id} exploded"))

    return DelegatedTask(
        task_id=task_id,
        task=f"failing-task-{task_id}",
        context={},
        agent_factory=_work,
    )


# ---------------------------------------------------------------------------
# SubagentOutcome invariant tests
# ---------------------------------------------------------------------------


class TestSubagentOutcome:
    def test_requires_exactly_one_of_result_or_error(self):
        """Constructing with neither result nor error raises ValueError."""
        with pytest.raises(ValueError, match="exactly one"):
            SubagentOutcome(task_id="t1")

    def test_rejects_both_result_and_error(self):
        """Constructing with both result and error raises ValueError."""
        with pytest.raises(ValueError, match="exactly one"):
            SubagentOutcome(
                task_id="t1",
                result="data",
                error=SubagentError(subagent_id="t1"),
            )

    def test_accepts_result_only(self):
        outcome = SubagentOutcome(task_id="t1", result="hello")
        assert outcome.result == "hello"
        assert outcome.error is None

    def test_accepts_error_only(self):
        err = SubagentError(subagent_id="t1")
        outcome = SubagentOutcome(task_id="t1", error=err)
        assert outcome.error is err
        assert outcome.result is None


# ---------------------------------------------------------------------------
# SubagentManager.spawn tests
# ---------------------------------------------------------------------------


class TestSubagentManagerSpawn:
    async def test_spawn_returns_result_on_success(self):
        """spawn() returns SubagentOutcome with result on success."""
        manager = SubagentManager()
        task = make_task("s1", result={"answer": 42})
        outcome = await manager.spawn(task)

        assert outcome.task_id == "s1"
        assert outcome.result == {"answer": 42}
        assert outcome.error is None

    async def test_spawn_returns_error_on_failure(self):
        """spawn() returns SubagentOutcome with SubagentError on failure."""
        manager = SubagentManager()
        task = make_failing_task("s2")
        outcome = await manager.spawn(task)

        assert outcome.task_id == "s2"
        assert outcome.result is None
        assert isinstance(outcome.error, SubagentError)
        assert outcome.error.subagent_id == "s2"

    async def test_spawn_preserves_cause(self):
        """spawn() chains the original exception as __cause__."""
        manager = SubagentManager()
        original = ValueError("bad input")
        task = make_failing_task("s3", exc=original)
        outcome = await manager.spawn(task)

        assert outcome.error.__cause__ is original


# ---------------------------------------------------------------------------
# SubagentManager.run_all tests
# ---------------------------------------------------------------------------


class TestSubagentManagerRunAll:
    async def test_run_all_empty_list(self):
        """run_all() with no tasks returns an empty list."""
        manager = SubagentManager()
        outcomes = await manager.run_all([])
        assert outcomes == []

    async def test_run_all_single_success(self):
        """run_all() with one successful task returns one outcome."""
        manager = SubagentManager()
        task = make_task("t1", result="done")
        outcomes = await manager.run_all([task])

        assert len(outcomes) == 1
        assert outcomes[0].task_id == "t1"
        assert outcomes[0].result == "done"

    async def test_run_all_multiple_successes(self):
        """run_all() returns outcomes for all successful tasks in order."""
        manager = SubagentManager()
        tasks = [make_task(f"t{i}", result=i) for i in range(5)]
        outcomes = await manager.run_all(tasks)

        assert len(outcomes) == 5
        for i, outcome in enumerate(outcomes):
            assert outcome.task_id == f"t{i}"
            assert outcome.result == i

    async def test_run_all_one_failure_does_not_cancel_siblings(self):
        """One subagent failure returns error for that task; siblings succeed."""
        manager = SubagentManager()
        tasks = [
            make_task("ok1", result="a"),
            make_failing_task("bad"),
            make_task("ok2", result="b"),
        ]
        outcomes = await manager.run_all(tasks)

        assert len(outcomes) == 3

        # First task succeeded
        assert outcomes[0].task_id == "ok1"
        assert outcomes[0].result == "a"
        assert outcomes[0].error is None

        # Second task failed
        assert outcomes[1].task_id == "bad"
        assert outcomes[1].result is None
        assert isinstance(outcomes[1].error, SubagentError)
        assert outcomes[1].error.subagent_id == "bad"

        # Third task succeeded
        assert outcomes[2].task_id == "ok2"
        assert outcomes[2].result == "b"
        assert outcomes[2].error is None

    async def test_run_all_concurrent_execution(self):
        """run_all() runs tasks concurrently (total time < sum of delays)."""
        manager = SubagentManager()
        delay = 0.1
        num_tasks = 3
        tasks = [make_task(f"c{i}", result=i, delay=delay) for i in range(num_tasks)]

        start = asyncio.get_event_loop().time()
        outcomes = await manager.run_all(tasks)
        elapsed = asyncio.get_event_loop().time() - start

        # All tasks completed
        assert len(outcomes) == num_tasks
        for i, outcome in enumerate(outcomes):
            assert outcome.result == i

        # If serial, would take num_tasks * delay = 0.3s
        # Concurrent should be close to delay = 0.1s
        assert elapsed < delay * num_tasks * 0.8  # generous margin

    async def test_run_all_all_failures(self):
        """run_all() with all failing tasks returns errors for each."""
        manager = SubagentManager()
        tasks = [make_failing_task(f"f{i}") for i in range(3)]
        outcomes = await manager.run_all(tasks)

        assert len(outcomes) == 3
        for i, outcome in enumerate(outcomes):
            assert outcome.task_id == f"f{i}"
            assert isinstance(outcome.error, SubagentError)
            assert outcome.error.subagent_id == f"f{i}"

    async def test_run_all_outcomes_keyed_to_originating_task(self):
        """Each outcome's task_id matches the originating DelegatedTask's task_id."""
        manager = SubagentManager()
        tasks = [
            make_task("alpha", result="A"),
            make_failing_task("beta"),
            make_task("gamma", result="G"),
        ]
        outcomes = await manager.run_all(tasks)

        assert outcomes[0].task_id == "alpha"
        assert outcomes[1].task_id == "beta"
        assert outcomes[2].task_id == "gamma"
