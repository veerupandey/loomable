"""Parallel_Group — Container for concurrent step execution.

A Parallel_Group groups one or more Steps (or composable elements) for
concurrent execution. It compiles to a Flow with ``engine="parallel"`` and
no inter-node edges, so all contained steps execute concurrently in a single
BSP superstep via the ParallelEngine.

Each step's output is merged into SharedState keyed by the step's name.
The Parallel_Group implements the Runnable protocol so it can be used
standalone, nested inside Workflows, or composed with existing helpers.
"""

from __future__ import annotations

__all__ = ["Parallel_Group"]

from typing import Any, TYPE_CHECKING

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.flow import Flow
from loomable.flow.nodes import Node
from loomable.flow.runnable import Runnable

if TYPE_CHECKING:
    pass


class Parallel_Group:
    """Container for concurrent step execution.

    Accepts one or more composable elements (Steps, Conditions, Loops,
    Workflows) as positional arguments and executes them concurrently
    using the ParallelEngine.

    Parameters
    ----------
    *steps:
        One or more composable elements to execute concurrently. Each must
        have a ``name`` attribute (used as node_id and SharedState key).
    name:
        Optional name for this parallel group. If not provided, auto-generates
        from the contained step names (e.g., ``"parallel_research_analysis"``).
    """

    def __init__(
        self,
        *steps: Any,
        name: str | None = None,
    ) -> None:
        if not steps:
            raise ValueError("At least one step is required")

        self._steps = list(steps)

        # Auto-generate name from step names if not provided.
        if name is not None:
            self._name = name
        else:
            self._name = self._auto_generate_name(self._steps)

        # Compile to a parallel Flow at construction time.
        self._compiled_flow = self._compile()

    @property
    def name(self) -> str:
        """The parallel group's name (auto-generated or user-provided)."""
        return self._name

    @property
    def steps(self) -> list[Any]:
        """The contained steps."""
        return list(self._steps)

    async def arun(
        self, input: Any, *, context: RunContext | None = None  # noqa: A002
    ) -> RunResult:
        """Execute all steps concurrently and merge outputs by step name.

        Delegates to the internal parallel Flow. Each step's output is
        written to SharedState keyed by the step's name. The merged
        state is accessible via the RunResult's sub_results.
        """
        result = await self._compiled_flow.arun(input, context=context)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compile(self) -> Flow:
        """Compile the steps into a parallel Flow with no inter-node edges.

        Each step becomes a node identified by its ``name`` attribute.
        The Flow uses ``engine="parallel"`` so all nodes execute concurrently
        in a single BSP superstep.
        """
        nodes: dict[str, Runnable] = {}

        for step in self._steps:
            # Each composable element must have a name attribute for keying.
            step_name = self._get_step_name(step)

            # The step itself is the Runnable for the node (it has arun).
            nodes[step_name] = step

        # Create a Flow with parallel engine and no edges (all nodes run concurrently).
        return Flow(nodes=nodes, edges=[], engine="parallel")

    @staticmethod
    def _get_step_name(step: Any) -> str:
        """Extract the name from a composable element.

        Steps, Conditions, Parallel_Groups, Workflows, and Loops may each
        expose their name differently. Falls back to the class name + id
        if no name attribute is found.
        """
        if hasattr(step, "name"):
            name = step.name
            if callable(name) and not isinstance(name, property):
                name = name()
            if name:
                return str(name)

        # Fallback for elements without a name attribute
        return f"{type(step).__name__}_{id(step)}"

    @staticmethod
    def _auto_generate_name(steps: list[Any]) -> str:
        """Auto-generate a group name from contained step names.

        Combines step names into a single identifier like
        ``"parallel_research_analysis"`` by joining with underscores
        and prefixing with "parallel_".
        """
        names: list[str] = []
        for step in steps:
            if hasattr(step, "name"):
                name = step.name
                if callable(name) and not isinstance(name, property):
                    name = name()
                if name:
                    names.append(str(name))

        if names:
            return "parallel_" + "_".join(names)

        return "parallel_group"

    def __repr__(self) -> str:
        n = len(self._steps)
        return f"Parallel_Group(name={self._name!r}, steps={n})"
