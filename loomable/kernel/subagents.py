"""loomable.kernel.subagents - Subagent Manager for delegated task execution.

The SubagentManager spawns and runs delegated tasks as concurrent agent loops.
Each DelegatedTask carries a factory that creates the async function representing
the subagent's work. Results are keyed back to the originating task.

One subagent failure does NOT cancel siblings — the same isolation pattern used
by ToolRuntime (asyncio.gather with return_exceptions=True).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from loomable.kernel.errors import SubagentError


@dataclass
class DelegatedTask:
    """A unit of work to be performed by a subagent.

    Attributes:
        task_id: Unique identifier for this delegated task.
        task: Human-readable description of the work.
        context: Arbitrary context data passed to the subagent.
        agent_factory: A callable that creates an async function representing
            the subagent's work. The factory receives no arguments and returns
            an awaitable that produces the subagent's result.
    """

    task_id: str
    task: str
    context: dict[str, Any]
    agent_factory: Callable[[], Awaitable[Any]]


@dataclass
class SubagentOutcome:
    """The outcome of a delegated task — carries exactly one of result or error.

    Attributes:
        task_id: The identifier of the originating DelegatedTask.
        result: The subagent's result if it succeeded, None otherwise.
        error: A SubagentError if the subagent failed, None otherwise.

    Invariant: exactly one of result/error is set (not None).
    """

    task_id: str
    result: Any | None = None
    error: SubagentError | None = None

    def __post_init__(self) -> None:
        has_result = self.result is not None
        has_error = self.error is not None
        if has_result == has_error:
            raise ValueError(
                "SubagentOutcome must carry exactly one of 'result' or 'error', "
                f"got result={has_result}, error={has_error}"
            )


class SubagentManager:
    """Spawns and manages delegated tasks as concurrent subagent loops.

    Uses asyncio.gather(..., return_exceptions=True) with per-task isolation
    so that one subagent failure does not cancel siblings.
    """

    async def spawn(self, task: DelegatedTask) -> SubagentOutcome:
        """Run a single delegated task and return its outcome.

        Args:
            task: The DelegatedTask to execute.

        Returns:
            A SubagentOutcome keyed to the originating task, carrying either
            the subagent's result or a SubagentError naming the failed subagent.
        """
        try:
            result = await task.agent_factory()
            return SubagentOutcome(task_id=task.task_id, result=result)
        except Exception as exc:
            error = SubagentError(subagent_id=task.task_id)
            error.__cause__ = exc
            return SubagentOutcome(task_id=task.task_id, error=error)

    async def run_all(self, tasks: list[DelegatedTask]) -> list[SubagentOutcome]:
        """Run all delegated tasks concurrently and return their outcomes.

        Each outcome is keyed to its originating task. One subagent failure
        does not cancel siblings — all tasks run to completion independently.

        Args:
            tasks: List of DelegatedTask objects to execute concurrently.

        Returns:
            A list of SubagentOutcome objects, one per input task, in the
            same order as the input tasks.
        """
        if not tasks:
            return []

        coroutines = [self._run_one(task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        outcomes: list[SubagentOutcome] = []
        for task, result in zip(tasks, results):
            if isinstance(result, SubagentOutcome):
                outcomes.append(result)
            elif isinstance(result, BaseException):
                # An unexpected exception escaped _run_one; wrap it
                error = SubagentError(subagent_id=task.task_id)
                error.__cause__ = result
                outcomes.append(SubagentOutcome(task_id=task.task_id, error=error))
            else:
                # Should not happen, but handle defensively
                error = SubagentError(subagent_id=task.task_id)
                outcomes.append(SubagentOutcome(task_id=task.task_id, error=error))

        return outcomes

    async def _run_one(self, task: DelegatedTask) -> SubagentOutcome:
        """Run a single delegated task with exception isolation.

        Catches exceptions from the subagent factory and wraps them into
        a SubagentOutcome with a SubagentError, ensuring isolation.
        """
        try:
            result = await task.agent_factory()
            return SubagentOutcome(task_id=task.task_id, result=result)
        except Exception as exc:
            error = SubagentError(subagent_id=task.task_id)
            error.__cause__ = exc
            return SubagentOutcome(task_id=task.task_id, error=error)
