"""loomable.agent.team - Team convenience wrapper for explicit orchestration modes.

:class:`Team` builds a parent :class:`~loomable.agent.builder.Agent` with
auto-generated coordination instructions and ``subagents=members``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .builder import Agent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from loomable.agent.run import RunResult
    from loomable.content import AgentInput
    from loomable.kernel.contracts import ModelProvider

    from .builder import ModelSpec

TeamMode = Literal["coordinate", "route", "broadcast", "sequential"]

__all__ = ["Team", "TeamMode"]

_MODE_INSTRUCTIONS: dict[str, str] = {
    "coordinate": (
        "You are a team coordinator. Delegate the task to ALL team members below, "
        "then synthesize their responses into one unified answer.\n\n"
        "Process:\n"
        "1. Call each delegate_to_* tool with the full task (or a focused portion).\n"
        "2. Wait for all members to respond.\n"
        "3. Synthesize the combined insights into a single coherent response."
    ),
    "route": (
        "You are a team router. Select the SINGLE best team member for this task "
        "and delegate only to them.\n\n"
        "Process:\n"
        "1. Analyze the task and pick the best-matched member.\n"
        "2. Call only that member's delegate_to_* tool.\n"
        "3. Return their output (lightly edited if needed)."
    ),
    "broadcast": (
        "You are a team broadcast coordinator. Send the SAME input to ALL team "
        "members and merge their labeled results.\n\n"
        "Process:\n"
        "1. Call every delegate_to_* tool with the same task.\n"
        "2. Present each member's response clearly labeled by role."
    ),
    "sequential": (
        "You are a team pipeline coordinator. Process the task through team members "
        "IN ORDER. Each member builds on the previous member's output.\n\n"
        "Process:\n"
        "1. Call the first member's delegate_to_* tool with the original task.\n"
        "2. Pass each subsequent member the prior member's output as context.\n"
        "3. Return the final member's output (optionally summarize the chain)."
    ),
}


def _assemble_team_instructions(
    mode: str,
    members: list[Agent],
    extra: str | None,
) -> str:
    """Build the parent agent instructions for a team mode."""
    if mode not in _MODE_INSTRUCTIONS:
        valid = ", ".join(sorted(_MODE_INSTRUCTIONS))
        raise ValueError(f"Unknown team mode {mode!r}. Valid modes: {valid}")

    from .delegation import format_member_roster

    parts = [_MODE_INSTRUCTIONS[mode], "", "Team members:", format_member_roster(members)]
    if extra:
        parts.extend(["", "Additional instructions:", extra.strip()])
    return "\n".join(parts)


class Team:
    """Explicit multi-agent orchestration via auto-generated parent instructions.

    Under the hood this creates a parent :class:`Agent` with ``subagents=members``
    and mode-specific coordination instructions.

    Parameters
    ----------
    members:
        Specialist agents that the coordinator can delegate to.
    model:
        Model for the coordinating parent agent.
    mode:
        Orchestration pattern: ``coordinate``, ``route``, ``broadcast``, or
        ``sequential``.
    instructions:
        Optional extra instructions appended to the auto-generated prompt.
    """

    def __init__(
        self,
        members: list[Agent],
        model: "ModelProvider | ModelSpec | str",
        *,
        mode: TeamMode = "coordinate",
        instructions: str | None = None,
    ) -> None:
        if not members:
            raise ValueError("Team requires at least one member")

        self._members = members
        self._model = model
        self._mode = mode
        self._instructions = instructions
        self._agent = Agent(
            model=model,
            role="Team Coordinator",
            goal=f"Coordinate the team in {mode} mode",
            instructions=_assemble_team_instructions(mode, members, instructions),
            subagents=members,
        )

    @property
    def agent(self) -> Agent:
        """The underlying parent :class:`Agent` builder."""
        return self._agent

    @property
    def mode(self) -> str:
        """The team's orchestration mode."""
        return self._mode

    @property
    def members(self) -> list[Agent]:
        """The team member agents."""
        return list(self._members)

    async def arun(
        self,
        input: "AgentInput | str",  # noqa: A002
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
        context: dict[str, Any] | None = None,
    ) -> "RunResult":
        """Run the team coordinator and return a :class:`~loomable.agent.run.RunResult`."""
        return await self._agent.arun(
            input,
            images=images,
            videos=videos,
            audio=audio,
            output_schema=output_schema,
            context=context,
        )

    def run(
        self,
        input: "AgentInput | str",  # noqa: A002
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
        context: dict[str, Any] | None = None,
    ) -> "RunResult":
        """Synchronous wrapper around :meth:`arun`."""
        return self._agent.run(
            input,
            images=images,
            videos=videos,
            audio=audio,
            output_schema=output_schema,
            context=context,
        )
