"""Workflow — Top-level orchestrator that compiles a steps list into a Flow.

A Workflow accepts a declarative list of composable elements (Step, Condition,
Parallel_Group, Loop, nested Workflows) and compiles them into a Flow graph
at construction time via WorkflowCompiler. This gives early error detection,
explain() support before any execution, and zero runtime overhead beyond the
existing engine.

The Workflow implements the Runnable protocol so it can be nested inside other
Flows, used as a Loop body, or passed to existing helpers.
"""

from __future__ import annotations

__all__ = ["Workflow"]

import asyncio
from typing import Any, TYPE_CHECKING

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.compiler import WorkflowCompiler
from loomable.flow.flow import Flow, FlowPlan
from loomable.flow.nodes import FlowConfigError
from loomable.flow.state import SharedState

if TYPE_CHECKING:
    from loomable.flow.memory import MemoryStore


class Workflow:
    """Top-level orchestrator that compiles a steps list into a Flow.

    Parameters
    ----------
    name:
        A human-readable name for the workflow.
    steps:
        A non-empty list of composable elements (Step, Condition,
        Parallel_Group, Loop, or nested Workflow) to execute in order.
    deps:
        Optional dependency injection object shared across all steps.
    memory:
        If ``True``, auto-creates a TieredMemoryStore scoped to
        ``session_id``. Can also be a MemoryStore instance directly.
    session_id:
        Optional session identifier for memory/checkpoint scoping.
    """

    def __init__(
        self,
        name: str,
        steps: list[Any],
        *,
        deps: Any = None,
        memory: bool | Any = False,
        session_id: str | None = None,
    ) -> None:
        # Validate steps is non-empty
        if not steps:
            raise ValueError("At least one step is required")

        # Validate no duplicate step names
        self._validate_no_duplicate_names(steps)

        self._name = name
        self._steps = list(steps)
        self._deps = deps
        self._session_id = session_id

        # Resolve memory configuration
        memory_store: Any = None
        if memory is True:
            from loomable.flow.memory import TieredMemoryStore

            memory_store = TieredMemoryStore(session_id=session_id)
        elif memory and memory is not False:
            # Assume it's a MemoryStore instance
            memory_store = memory
        self._memory = memory_store

        # Compile the steps list into a Flow at construction time
        self._compiled_flow: Flow = WorkflowCompiler.compile(
            steps,
            name=name,
            deps=deps,
            memory=memory_store,
            session_id=session_id,
        )

        # SharedState from the most recent run (None until first execution)
        self._last_state: SharedState | None = None

    @staticmethod
    def _validate_no_duplicate_names(steps: list[Any]) -> None:
        """Walk the steps list and detect duplicate names.

        Checks Steps and Parallel_Groups (which have name attributes).
        Raises FlowConfigError if any name appears more than once.
        """
        from loomable.flow.step import Step
        from loomable.flow.parallel_group import Parallel_Group

        seen: set[str] = set()
        for element in steps:
            name: str | None = None

            if isinstance(element, Step):
                name = element.name
            elif isinstance(element, Parallel_Group):
                name = element.name
            elif hasattr(element, "name"):
                # Nested Workflows and other named elements
                attr = element.name
                if callable(attr) and not isinstance(attr, property):
                    attr = attr()
                if attr:
                    name = str(attr)

            if name is not None:
                if name in seen:
                    raise FlowConfigError(f"Duplicate step name: '{name}'")
                seen.add(name)

    async def arun(
        self, input: Any, *, context: RunContext | None = None  # noqa: A002
    ) -> RunResult:
        """Execute the workflow by delegating to the compiled Flow.

        Parameters
        ----------
        input:
            The initial input to pass to the first step.
        context:
            Optional RunContext for dependency injection and observability.

        Returns
        -------
        RunResult
            The result from the last step in the workflow.
        """
        result = await self._compiled_flow.arun(input, context=context)

        # Capture SharedState from the context for post-execution inspection
        if context is not None and context.shared_state is not None:
            self._last_state = context.shared_state
        else:
            # The Flow creates its own SharedState internally; extract from result
            # metadata if available. Otherwise, create a fresh one.
            self._last_state = SharedState()

        return result

    def run(self, input: Any) -> RunResult:  # noqa: A002
        """Synchronous convenience wrapper around ``arun``.

        Wraps ``arun`` in ``asyncio.run()`` for use in non-async contexts.
        """
        return asyncio.run(self.arun(input))

    def explain(self) -> FlowPlan:
        """Return the FlowPlan describing the compiled graph topology.

        Available before any execution — the graph is compiled at
        construction time.
        """
        return self._compiled_flow.explain()

    @property
    def state(self) -> SharedState:
        """Access the SharedState from the most recent run.

        Returns an empty SharedState if the workflow has not been executed yet.
        """
        if self._last_state is None:
            return SharedState()
        return self._last_state

    @property
    def name(self) -> str:
        """The workflow's name."""
        return self._name

    def __repr__(self) -> str:
        n = len(self._steps)
        return f"Workflow(name={self._name!r}, steps={n})"
