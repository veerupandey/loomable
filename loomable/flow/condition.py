"""Condition — Declarative if/else branching construct.

A Condition evaluates a predicate against SharedState and routes execution
to either ``then_steps`` or ``else_steps``. It is the building block for
branching logic inside Workflows.

The Condition satisfies the :class:`Runnable` protocol so it can be used
standalone or embedded within a Workflow, Loop, or other composable
structure.

When used standalone (via ``arun``), the Condition creates an internal
execution path: evaluates the predicate, then runs the appropriate branch
sequentially. When compiled into a Workflow, it translates to a RouterNode
with branching edges (handled by the WorkflowCompiler).
"""

from __future__ import annotations

__all__ = ["Condition", "ComposableElement"]

from typing import Any, Callable, Union, TYPE_CHECKING

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.loop import Loop
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.state import SharedState

if TYPE_CHECKING:
    from loomable.flow.step import Step
    from loomable.flow.workflow import Workflow


# ---------------------------------------------------------------------------
# ComposableElement type alias
# ---------------------------------------------------------------------------

# Forward references are used for Workflow to avoid circular imports.
# At runtime, validation uses _VALID_COMPOSABLE_TYPES tuple for isinstance checks.
ComposableElement = Union["Step", "Condition", "Parallel_Group", Loop, "Workflow"]


def _is_valid_composable(element: Any) -> bool:
    """Check if an element is a valid composable type at runtime.

    We import lazily to avoid circular imports between Condition,
    Parallel_Group, and Workflow.
    """
    from loomable.flow.step import Step

    # Check for types that are already available
    valid_types = (Step, Condition, Loop)

    if isinstance(element, valid_types):
        return True

    # Check Parallel_Group (may not exist yet during early development)
    try:
        from loomable.flow.parallel_group import Parallel_Group

        if isinstance(element, Parallel_Group):
            return True
    except ImportError:
        pass

    # Check Workflow (may not exist yet during early development)
    try:
        from loomable.flow.workflow import Workflow

        if isinstance(element, Workflow):
            return True
    except ImportError:
        pass

    # Also accept anything that satisfies the Runnable protocol AND has a
    # name attribute (duck-typing for Step-like objects)
    return False


def _get_type_name(element: Any) -> str:
    """Get a readable type name for error messages."""
    return type(element).__name__


# ---------------------------------------------------------------------------
# Parallel_Group forward reference (for type alias only)
# ---------------------------------------------------------------------------

# This is a string forward reference used in the type alias above.
# The actual class will be imported lazily in _is_valid_composable.
if TYPE_CHECKING:
    from loomable.flow.parallel_group import Parallel_Group


# ---------------------------------------------------------------------------
# Condition class
# ---------------------------------------------------------------------------


class Condition:
    """Declarative if/else branching construct.

    Evaluates a predicate callable against SharedState and routes execution
    to ``then_steps`` (when True) or ``else_steps`` (when False).

    Parameters
    ----------
    condition:
        A callable that receives a :class:`SharedState` and returns a bool.
        Determines which branch to execute.
    then_steps:
        List of composable elements to execute when the condition is True.
        Must contain at least one element.
    else_steps:
        Optional list of composable elements to execute when the condition
        is False. If not provided and condition is False, input passes through
        unchanged.
    """

    def __init__(
        self,
        condition: Callable[[SharedState], bool],
        then_steps: list[Any],
        else_steps: list[Any] | None = None,
    ) -> None:
        # Validate then_steps is non-empty
        if not then_steps:
            raise ValueError("At least one then_step is required")

        # Validate all elements in then_steps are valid composable types
        for element in then_steps:
            if not _is_valid_composable(element):
                raise TypeError(
                    f"Invalid element type: {_get_type_name(element)}. "
                    "Expected Step, Condition, Parallel_Group, Loop, or Workflow"
                )

        # Validate all elements in else_steps (if provided)
        if else_steps is not None:
            for element in else_steps:
                if not _is_valid_composable(element):
                    raise TypeError(
                        f"Invalid element type: {_get_type_name(element)}. "
                        "Expected Step, Condition, Parallel_Group, Loop, or Workflow"
                    )

        self._condition = condition
        self._then_steps = list(then_steps)
        self._else_steps = list(else_steps) if else_steps is not None else None

    @property
    def condition(self) -> Callable[[SharedState], bool]:
        """The predicate callable."""
        return self._condition

    @property
    def then_steps(self) -> list[Any]:
        """Steps to execute when condition is True."""
        return list(self._then_steps)

    @property
    def else_steps(self) -> list[Any] | None:
        """Steps to execute when condition is False (or None)."""
        return list(self._else_steps) if self._else_steps is not None else None

    async def arun(
        self, input: Any, *, context: RunContext | None = None  # noqa: A002
    ) -> RunResult:
        """Execute the Condition standalone.

        Evaluates the condition predicate against SharedState from the context,
        then runs the appropriate branch sequentially:
        - If True: runs then_steps in sequence
        - If False + else_steps: runs else_steps in sequence
        - If False + no else_steps: passes input through unchanged

        Each step in the branch receives the output of the previous step
        (pipeline style).
        """
        ctx = context or RunContext()

        # Get or create SharedState for predicate evaluation
        state = ctx.shared_state
        if state is None:
            state = SharedState()
            ctx.shared_state = state

        # Evaluate the condition predicate
        predicate_result = self._condition(state)

        if predicate_result:
            # Execute then_steps sequentially
            return await self._run_steps(self._then_steps, input, ctx)
        elif self._else_steps is not None:
            # Execute else_steps sequentially
            return await self._run_steps(self._else_steps, input, ctx)
        else:
            # No else_steps — pass input through unchanged
            return self._passthrough_result(input)

    async def _run_steps(
        self, steps: list[Any], input: Any, context: RunContext  # noqa: A002
    ) -> RunResult:
        """Run a list of steps sequentially, piping output to next input.

        Each step must satisfy the Runnable protocol (has ``arun``).
        """
        current_input = input
        last_result: RunResult | None = None

        for step in steps:
            # Each composable element satisfies the Runnable protocol
            last_result = await step.arun(current_input, context=context)
            # Extract text output for next step's input
            current_input = last_result.output.text() if last_result.output else ""

        assert last_result is not None  # then_steps guaranteed non-empty
        return last_result

    @staticmethod
    def _passthrough_result(input: Any) -> RunResult:  # noqa: A002
        """Create a RunResult that passes the input through unchanged."""
        from loomable.content import AgentOutput, MediaPart, Modality

        text = str(input) if input is not None else ""
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

    def __repr__(self) -> str:
        then_count = len(self._then_steps)
        else_count = len(self._else_steps) if self._else_steps else 0
        return f"Condition(then_steps={then_count}, else_steps={else_count})"
