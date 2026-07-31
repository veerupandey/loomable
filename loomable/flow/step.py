"""Step — Named wrapper around a Runnable or callable.

A Step is the atomic building block of Workflows. It carries a name
(used as node_id when compiled into a Flow), an optional description,
and optional per-step dependency injection.

The Step satisfies the :class:`Runnable` protocol by delegating ``arun``
to the wrapped agent (or a :class:`FunctionRunnable` adapter for plain
callables).
"""

from __future__ import annotations

__all__ = ["Step"]

from typing import Any, Callable

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.runnable import FunctionRunnable, Runnable


class Step:
    """Named wrapper around a Runnable or callable.

    Parameters
    ----------
    name:
        A non-empty string that uniquely identifies this step within a
        Workflow. Used as ``node_id`` in the compiled Flow graph.
    agent:
        The execution unit — either an object satisfying the :class:`Runnable`
        protocol or a plain sync/async callable (which is adapted via
        :class:`FunctionRunnable`).
    description:
        Optional human-readable description of what this step does.
    deps:
        Optional dependency object injected into :class:`RunContext` when
        this step executes, overriding any flow-level deps for this step only.
    """

    def __init__(
        self,
        name: str,
        agent: Runnable | Callable[..., Any],
        *,
        description: str = "",
        deps: Any = None,
    ) -> None:
        if not name:
            raise ValueError("Step name is required")

        self._name = name
        self._description = description
        self._deps = deps

        # Wrap plain callables in FunctionRunnable to satisfy the Runnable protocol.
        if isinstance(agent, Runnable):
            self._agent: Runnable = agent
        elif callable(agent):
            self._agent = FunctionRunnable(agent)
        else:
            raise TypeError(
                f"agent must be a Runnable or callable, got {type(agent).__name__}"
            )

    async def arun(
        self, input: Any, *, context: RunContext | None = None  # noqa: A002
    ) -> RunResult:
        """Execute the wrapped agent, injecting step-level deps if configured.

        If this Step has ``deps`` set, it overrides the ``deps`` on the
        RunContext for this execution only. A new or cloned RunContext is
        used to avoid mutating a shared context object.
        """
        if self._deps is not None:
            # Inject step-level deps by creating/updating the context.
            if context is None:
                context = RunContext(deps=self._deps)
            else:
                # Clone-ish: create a new RunContext preserving other fields
                # but overriding deps for this step only.
                context = RunContext(
                    events=context.events,
                    max_steps=context.max_steps,
                    token_budget=context.token_budget,
                    loop_repeat_threshold=context.loop_repeat_threshold,
                    deps=self._deps,
                    shared_state=context.shared_state,
                    memory=context.memory,
                )

        return await self._agent.arun(input, context=context)

    @property
    def name(self) -> str:
        """The step's unique identifier (used as node_id in compiled Flows)."""
        return self._name

    @property
    def description(self) -> str:
        """Optional human-readable description of this step."""
        return self._description
