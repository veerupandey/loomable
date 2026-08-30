"""loomable.kernel.planner - Execution planning capability.

The Planner produces an execution plan for a given task. It uses a separately
configured Planning_Model when one is present, otherwise falls back to the
agent's primary model. An unavailable planning model raises PlanningModelError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loomable.kernel.errors import PlanningModelError
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.models import ModelRequest, ModelResponse


@dataclass
class TaskContext:
    """Context describing a task to be planned.

    Attributes
    ----------
    task:
        A natural-language description of the task to plan.
    context:
        Arbitrary context data relevant to the task (e.g. tool schemas,
        prior turns, constraints).
    """

    task: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """An execution plan produced by the Planner.

    Attributes
    ----------
    steps:
        Ordered list of plan steps (natural language or structured).
    metadata:
        Arbitrary metadata about the plan (e.g. model used, token usage).
    """

    steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Planner:
    """Produces execution plans by invoking a configured model.

    When a ``planning_model_id`` is provided, the Planner uses that specific
    model tier for plan generation. Otherwise, it falls back to the primary
    (default) model. If the configured planning model is unavailable (not
    registered in the ModelInterface), a ``PlanningModelError`` is raised
    identifying the model.

    Parameters
    ----------
    model_interface:
        The model routing interface used to invoke providers.
    planning_model_id:
        Optional identifier for a dedicated planning model/tier.
        When ``None``, the default model is used.
    """

    def __init__(
        self,
        model_interface: ModelInterface,
        planning_model_id: str | None = None,
    ) -> None:
        self._model_interface = model_interface
        self._planning_model_id = planning_model_id

    @property
    def planning_model_id(self) -> str | None:
        """Return the configured planning model id, or None."""
        return self._planning_model_id

    async def plan(self, task: TaskContext) -> ExecutionPlan:
        """Produce an execution plan for the given task.

        Invokes the configured planning model if set, otherwise the primary
        model. Parses the response into an ``ExecutionPlan``.

        Parameters
        ----------
        task:
            The task context describing what needs to be planned.

        Returns
        -------
        ExecutionPlan
            The generated execution plan.

        Raises
        ------
        PlanningModelError
            If the configured planning model is unavailable.
        """
        request = self._build_request(task)

        try:
            response = await self._model_interface.invoke(
                request, tier=self._planning_model_id
            )
        except Exception as exc:
            # If a planning model was explicitly configured and is unavailable,
            # wrap as PlanningModelError identifying the model.
            if self._planning_model_id is not None:
                raise PlanningModelError(self._planning_model_id) from exc
            raise

        return self._parse_response(response)

    def _build_request(self, task: TaskContext) -> ModelRequest:
        """Build a ModelRequest from the task context."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a planning agent. Produce a step-by-step "
                    "execution plan for the given task."
                ),
            },
            {
                "role": "user",
                "content": task.task,
            },
        ]

        return ModelRequest(
            messages=messages,
            metadata={"task_context": task.context},
        )

    def _parse_response(self, response: ModelResponse) -> ExecutionPlan:
        """Parse a model response into an ExecutionPlan."""
        from loomable.plan_parse import parse_plan_steps

        content = response.content or ""
        steps = parse_plan_steps(content)

        return ExecutionPlan(
            steps=steps,
            metadata={
                "usage": response.usage,
                "model_metadata": response.metadata,
            },
        )
