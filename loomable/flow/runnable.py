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
    - Otherwise, the return value is converted to a text AgentOutput.
    """

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)
        # Inspect signature once to determine which special parameters to inject.
        sig = inspect.signature(fn)
        self._accepts_context = "context" in sig.parameters
        self._accepts_deps = "deps" in sig.parameters

    async def arun(
        self, input: Any, *, context: RunContext | None = None  # noqa: A002
    ) -> RunResult:
        """Run the wrapped function and produce a RunResult."""
        kwargs: dict[str, Any] = {}
        if self._accepts_context:
            kwargs["context"] = context
        if self._accepts_deps and context is not None:
            kwargs["deps"] = context.deps

        if self._is_async:
            raw = await self._fn(input, **kwargs)
        else:
            raw = self._fn(input, **kwargs)

        if isinstance(raw, RunResult):
            return raw

        # Dict returns are treated as SharedState updates (plan_steps, etc.)
        # so plan→map→synthesize can pass lists without stringifying.
        if isinstance(raw, dict):
            import json

            text = json.dumps(raw, ensure_ascii=False, default=str)
            output = AgentOutput(
                parts=[
                    MediaPart(
                        modality=Modality.TEXT,
                        media_type="text/plain",
                        data=text.encode("utf-8"),
                    )
                ]
            )
            return RunResult(
                output=output,
                session_id="",
                structured=raw,
                metadata={"state_updates": raw},
            )

        # Wrap a plain return value into a RunResult with a text AgentOutput.
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
        return RunResult(output=output, session_id="")
