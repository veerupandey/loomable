"""Route — N-way declarative routing construct (Agno Router / LangGraph multi-edge).

Unlike :class:`~loomable.flow.condition.Condition` (bool if/else), Route selects
among named choices via a chooser Runnable or callable. Compiles to
:class:`~loomable.flow.nodes.RouterNode` + gated edges + join.
"""

from __future__ import annotations

__all__ = ["Route"]

from typing import Any

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.command import Command
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.state import SharedState


class Route:
    """N-way branch: chooser picks one (or more) named choice paths.

    Parameters
    ----------
    chooser:
        Runnable/callable that returns a choice name, list of names, or a
        :class:`~loomable.flow.command.Command` with ``goto=``.
    choices:
        Mapping of choice name → composable element (Step, agent, callable,
        list of steps).
    handoff:
        When True, the selected branch owns the final output for this path.
    """

    def __init__(
        self,
        chooser: Any,
        choices: dict[str, Any],
        *,
        handoff: bool = False,
    ) -> None:
        if not choices:
            raise ValueError("Route requires at least one choice")
        self._chooser = chooser
        self._choices = dict(choices)
        self.handoff = bool(handoff)

    @property
    def chooser(self) -> Any:
        return self._chooser

    @property
    def choices(self) -> dict[str, Any]:
        return dict(self._choices)

    async def arun(
        self, input: Any, *, context: RunContext | None = None  # noqa: A002
    ) -> RunResult:
        """Standalone execution: choose a branch and run it sequentially."""
        from loomable.flow.workflow import Workflow

        ctx = context or RunContext()
        if ctx.shared_state is None:
            ctx.shared_state = SharedState()

        selection = await self._resolve_selection(input, ctx)
        if isinstance(selection, list):
            if not selection:
                raise ValueError("Route chooser returned an empty selection")
            selected = selection[0]
        else:
            selected = selection

        if selected not in self._choices:
            raise ValueError(
                f"Route chooser selected {selected!r} not in "
                f"{list(self._choices)!r}"
            )

        branch = self._choices[selected]
        steps = branch if isinstance(branch, list) else [branch]
        wf = Workflow(name=f"_route_{selected}", steps=None)
        for el in steps:
            wf.add(el)
        result = await wf.arun(input, context=ctx)
        result.metadata.setdefault("router_selected", selected)
        result.metadata.setdefault(
            "route_decision",
            {
                "selected": selected,
                "choices": list(self._choices),
                "reason": "standalone_route",
                "handoff": self.handoff,
            },
        )
        return result

    async def _resolve_selection(
        self, input: Any, context: RunContext  # noqa: A002
    ) -> str | list[str]:
        chooser = self._chooser
        if isinstance(chooser, Runnable):
            result = await chooser.arun(input, context=context)
        elif callable(chooser):
            result = await FunctionRunnable(chooser).arun(input, context=context)
        else:
            raise TypeError(
                f"Route chooser must be Runnable or callable, got {type(chooser).__name__}"
            )

        cmd = Command.from_metadata(result.metadata)
        if cmd and cmd.goto is not None:
            if cmd.update and context.shared_state is not None:
                for key, value in cmd.update.items():
                    context.shared_state.write(key, value)
            return cmd.goto

        if result.metadata and "selection" in result.metadata:
            return result.metadata["selection"]

        text = ""
        if result.output and result.output.parts:
            part = result.output.parts[0]
            data = getattr(part, "data", b"")
            if isinstance(data, bytes):
                text = data.decode("utf-8").strip()
            else:
                text = str(data).strip()
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) == 1:
            return parts[0]
        if len(parts) > 1:
            return parts
        return text
