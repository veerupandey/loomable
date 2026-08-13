"""loomable.agent.team - Named multi-agent orchestration modes.

:class:`Team` is a thin convenience wrapper over the subagent-delegation feature
(``Agent(subagents=[...])``). It lets callers pick a named orchestration mode
(``coordinate``, ``route``, ``broadcast``, ``sequential``) instead of writing
custom parent instructions. Under the hood a ``Team`` builds a single parent
:class:`~loomable.agent.builder.Agent` with auto-generated, mode-specific
instructions and ``subagents=members`` (docs/API.md "Team: Explicit
Orchestration Modes").
"""

from __future__ import annotations

__all__ = ["Team", "TeamMode"]

from typing import TYPE_CHECKING, Any, Iterable, Literal

from .builder import Agent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .run import RunResult

TeamMode = Literal["coordinate", "route", "broadcast", "sequential"]

_MODE_INSTRUCTIONS: dict[str, str] = {
    "coordinate": (
        "You coordinate a team of specialist subagents. Delegate the task to "
        "every relevant member, then synthesize their responses into one "
        "cohesive final answer."
    ),
    "route": (
        "You route work to a team of specialist subagents. Pick the single best "
        "member for the task, delegate to that member only, and return its "
        "answer."
    ),
    "broadcast": (
        "You broadcast the task to a team of specialist subagents. Send the same "
        "input to all members and merge their responses into a single labeled "
        "summary."
    ),
    "sequential": (
        "You run a team of specialist subagents in sequence. Delegate to each "
        "member in order, feeding earlier results into later steps, then return "
        "the final combined output."
    ),
}


class Team:
    """Named multi-agent orchestration over a set of member agents.

    Parameters mirror docs/API.md: ``members`` are the specialist agents,
    ``model`` backs the parent orchestrator, ``mode`` selects the orchestration
    pattern, and ``instructions`` (optional) are appended to the auto-generated
    mode instructions. Extra keyword arguments are forwarded to the parent
    :class:`Agent`.
    """

    def __init__(
        self,
        *,
        members: "Iterable[Agent]",
        model: Any,
        mode: TeamMode = "coordinate",
        instructions: str = "",
        name: str = "Team",
        **agent_kwargs: Any,
    ) -> None:
        self.members = list(members)
        if not self.members:
            raise ValueError("Team requires at least one member agent.")
        if mode not in _MODE_INSTRUCTIONS:
            raise ValueError(
                f"Unknown team mode '{mode}'. Expected one of "
                f"{sorted(_MODE_INSTRUCTIONS)}."
            )
        self.mode: TeamMode = mode

        combined = _MODE_INSTRUCTIONS[mode]
        if instructions:
            combined = f"{combined}\n\n{instructions}"

        self._agent = Agent(
            model=model,
            name=name,
            instructions=combined,
            subagents=self.members,
            **agent_kwargs,
        )

    @property
    def agent(self) -> Agent:
        """The underlying parent :class:`Agent` orchestrating the members."""
        return self._agent

    async def arun(self, input: Any, **kwargs: Any) -> "RunResult":  # noqa: A002
        """Run the team asynchronously (delegates to the parent agent)."""
        return await self._agent.arun(input, **kwargs)

    def run(self, input: Any, **kwargs: Any) -> "RunResult":  # noqa: A002
        """Run the team synchronously (delegates to the parent agent)."""
        return self._agent.run(input, **kwargs)

    def astream(self, input: Any, **kwargs: Any) -> Any:  # noqa: A002
        """Stream the team's output (delegates to the parent agent)."""
        return self._agent.astream(input, **kwargs)
