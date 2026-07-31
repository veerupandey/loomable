"""FlowClass — Decorator-driven event-based workflow definition.

This module provides the decorator infrastructure for the FlowClass pattern:
methods annotated with ``@start()``, ``@listen(source)``, or ``@router(source)``
are compiled into a Flow graph at instantiation time by the FlowClassCompiler.

The decorators are simple metadata attachers — they mark methods with a
``_flow_meta`` attribute containing a dataclass describing the role. The actual
graph compilation is handled by ``FlowClassCompiler`` (see compiler.py).

The ``FlowClass`` base class is also defined here. Subclasses define decorated
methods; at instantiation time (``__init__``), the methods are compiled into a
Flow via ``FlowClassCompiler``. Execution is via ``kickoff()`` or ``arun()``.
"""

from __future__ import annotations

__all__ = [
    "FlowClass",
    "start",
    "listen",
    "router",
    "_StartMeta",
    "_ListenMeta",
    "_RouterMeta",
]

from dataclasses import dataclass
from typing import Any, Callable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from loomable.agent.context import RunContext
    from loomable.agent.run import RunResult
    from loomable.flow.flow import FlowPlan
    from loomable.flow.state import SharedState

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _StartMeta:
    """Metadata attached to @start()-decorated methods."""

    pass


@dataclass
class _ListenMeta:
    """Metadata attached to @listen(source)-decorated methods."""

    source: str


@dataclass
class _RouterMeta:
    """Metadata attached to @router(source)-decorated methods."""

    source: str


# ---------------------------------------------------------------------------
# Decorator functions
# ---------------------------------------------------------------------------


def start() -> Callable[[F], F]:
    """Mark a method as a flow entry point.

    The decorated method will become a start node in the compiled Flow graph.
    It receives the initial input when ``kickoff()`` is called.

    Usage::

        class MyFlow(FlowClass):
            @start()
            async def begin(self, input):
                return processed_input
    """

    def decorator(fn: F) -> F:
        fn._flow_meta = _StartMeta()  # type: ignore[attr-defined]
        return fn

    return decorator


def listen(source: str) -> Callable[[F], F]:
    """Mark a method as listening to another method's output.

    The decorated method will receive the output of the ``source`` method
    as its input. An edge is created from the source node to this node
    in the compiled Flow graph.

    Parameters
    ----------
    source:
        The name of the method whose output this method consumes.

    Usage::

        class MyFlow(FlowClass):
            @listen("begin")
            async def process(self, input):
                return transformed_input
    """

    def decorator(fn: F) -> F:
        fn._flow_meta = _ListenMeta(source=source)  # type: ignore[attr-defined]
        return fn

    return decorator


def router(source: str) -> Callable[[F], F]:
    """Mark a method as a router that routes based on return value.

    The decorated method receives the output of the ``source`` method and
    returns a string naming the next node to execute. This creates a
    RouterNode in the compiled Flow graph.

    Parameters
    ----------
    source:
        The name of the method whose output this router receives.

    Usage::

        class MyFlow(FlowClass):
            @router("analyze")
            async def route_decision(self, input):
                if input.needs_review:
                    return "review"
                return "publish"
    """

    def decorator(fn: F) -> F:
        fn._flow_meta = _RouterMeta(source=source)  # type: ignore[attr-defined]
        return fn

    return decorator


# ---------------------------------------------------------------------------
# FlowClass base class
# ---------------------------------------------------------------------------


class FlowClass:
    """Base class for decorator-driven event-based workflows.

    Subclasses define methods decorated with ``@start()``, ``@listen(source)``,
    and ``@router(source)``. At instantiation time, the ``FlowClassCompiler``
    introspects these decorated methods and compiles them into an internal
    ``Flow`` graph. Execution is driven by ``kickoff()`` or ``arun()``.

    The class satisfies the ``Runnable`` protocol (``arun(input, *, context=None)
    -> RunResult``) so it can compose inside other Flows, Loops, and helpers.

    Subclasses may define an ``agents`` attribute (dict or namespace) that
    decorated methods can access via ``self.agents``.

    Usage::

        class MyFlow(FlowClass):
            agents = {"researcher": some_agent}

            @start()
            async def begin(self, input):
                return f"started: {input}"

            @listen("begin")
            async def process(self, input):
                result = await self.agents["researcher"].arun(input)
                return result.output.text

        flow = MyFlow()
        result = await flow.kickoff("hello")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Compile decorated methods into a Flow graph at instantiation time.

        Raises
        ------
        FlowConfigError
            If no ``@start`` method exists, if a source reference is invalid,
            or if cycles are detected in the listener graph.
        """
        from loomable.flow.compiler import FlowClassCompiler

        self._compiled_flow = FlowClassCompiler.compile(self)
        self._last_state: "SharedState | None" = None

    async def kickoff(self, input: Any) -> "RunResult":  # noqa: A002
        """Execute the compiled Flow with the given input.

        Parameters
        ----------
        input:
            The initial input to feed to the ``@start`` method(s).

        Returns
        -------
        RunResult
            The result of executing the compiled Flow graph.
        """
        from loomable.agent.context import RunContext
        from loomable.flow.state import SharedState

        ctx = RunContext()
        result = await self._compiled_flow.arun(input, context=ctx)
        # Capture the SharedState from the context after execution
        if ctx.shared_state is not None:
            self._last_state = ctx.shared_state
        return result

    async def arun(
        self, input: Any, *, context: "RunContext | None" = None  # noqa: A002
    ) -> "RunResult":
        """Execute the compiled Flow (Runnable protocol).

        Delegates to ``kickoff(input)``. The ``context`` parameter is accepted
        for protocol conformance but the FlowClass manages its own context
        internally.

        Parameters
        ----------
        input:
            The initial input to feed to the ``@start`` method(s).
        context:
            Optional RunContext (accepted for Runnable protocol conformance).

        Returns
        -------
        RunResult
            The result of executing the compiled Flow graph.
        """
        return await self.kickoff(input)

    def explain(self) -> "FlowPlan":
        """Return a FlowPlan describing the compiled graph topology.

        Available before any execution — shows the compiled topology
        derived from decorated methods without requiring a run.

        Returns
        -------
        FlowPlan
            Inspectable representation of the flow's execution plan with
            method names as node_ids.
        """
        return self._compiled_flow.explain()

    @property
    def state(self) -> "SharedState | None":
        """Access the SharedState from the most recent execution.

        Returns ``None`` if the flow has not been executed yet.
        """
        return self._last_state
