"""Runnable protocol and function adapter.

The ``Runnable`` protocol is the single execution contract shared by every
composable unit in loomable: agents, plain functions, loops, and flows.

``FunctionRunnable`` adapts sync and async plain Python functions to satisfy
the protocol so they can be used as graph nodes without ceremony.
"""

from __future__ import annotations

__all__ = ["Runnable", "FunctionRunnable"]

import asyncio
import inspect
import json
from typing import Any, Protocol, runtime_checkable

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality


@runtime_checkable
class Runnable(Protocol):
    """The universal execution contract.

    Any object with an ``arun`` method matching this signature composes as a
    node in a Flow, a body in a Loop, or a standalone executable.
    """

    async def arun(
        self, input: Any, *, context: "RunContext | None" = None  # noqa: A002
    ) -> "RunResult":
        """Execute asynchronously and return a RunResult."""
        ...


class FunctionRunnable:
    """Adapt a plain sync or async function to the Runnable protocol.

    The wrapped function receives the ``input`` positionally. If the function
    signature declares a ``context`` keyword parameter, the RunContext is
    passed through; otherwise it is omitted (preserving simple function
    signatures).

    The function's return value is wrapped into a RunResult:
    - If it already returns a RunResult, it is used as-is.
    - If it returns a ``dict``, each key is written into
      ``context.shared_state`` (so planners can publish ``plan_steps`` for
      ``MapNode``) and preserved in ``metadata["return_value"]``.
    - Otherwise, the return value is converted to a text AgentOutput.
    """

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)
        # Inspect signature once to determine which special parameters to inject.
        sig = inspect.signature(fn)
        self._accepts_context = "context" in sig.parameters
        self._accepts_deps = "deps" in sig.parameters
        self._accepts_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

    async def arun(
        self, input: Any, *, context: RunContext | None = None  # noqa: A002
    ) -> RunResult:
        """Run the wrapped function and produce a RunResult."""
        kwargs: dict[str, Any] = {}
        if self._accepts_context or self._accepts_var_keyword:
            kwargs["context"] = context
        if (self._accepts_deps or self._accepts_var_keyword) and context is not None:
            kwargs["deps"] = context.deps

        if self._is_async:
            raw = await self._fn(input, **kwargs)
        else:
            raw = self._fn(input, **kwargs)

        if isinstance(raw, RunResult):
            return raw

        metadata: dict[str, Any] = {}
        if isinstance(raw, dict):
            metadata["return_value"] = raw
            # Publish structured keys into SharedState for downstream nodes
            # (e.g. planner → MapNode via plan_steps).
            if context is not None and context.shared_state is not None:
                for key, value in raw.items():
                    context.shared_state.write(key, value)
            try:
                text = json.dumps(raw)
            except (TypeError, ValueError):
                text = str(raw)
        else:
            text = str(raw) if raw is not None else ""

        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=text.encode("utf-8"),
                )
            ]
        )
        return RunResult(output=output, session_id="", metadata=metadata)
