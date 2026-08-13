"""loomable.agent.delegation - Subagent delegation tools.

Turns each subagent (an :class:`~loomable.agent.builder.Agent`) into a
``delegate_to_<name>`` :class:`~loomable.agent.tools.FunctionTool` so the parent
agent's LLM can delegate tasks at runtime (Proposal §2). The parent decides who
to call, when, and in what order; subagent failures stay isolated because
:meth:`FunctionTool.invoke` captures exceptions and returns an error result
rather than crashing the parent run.
"""

from __future__ import annotations

__all__ = ["make_delegation_tools", "delegation_tool_name"]

import re
from typing import TYPE_CHECKING, Sequence

from .tools import FunctionTool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .builder import Agent


def _slug(text: str) -> str:
    """Turn arbitrary label text into a valid tool-name fragment.

    Tool names exposed to an LLM must match ``^[a-zA-Z0-9_-]+$``. Any run of
    non-alphanumeric characters collapses to a single underscore.
    """
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")


def _agent_label(agent: "Agent", index: int) -> str:
    """Best label for a subagent: explicit ``name`` first, then ``role``."""
    return (
        getattr(agent, "_name", "") or getattr(agent, "_role", "") or f"agent_{index + 1}"
    )


def delegation_tool_name(agent: "Agent", index: int = 0) -> str:
    """Return the ``delegate_to_<slug>`` tool name for a subagent."""
    slug = _slug(_agent_label(agent, index)) or f"agent_{index + 1}"
    return f"delegate_to_{slug}"


def make_delegation_tools(subagents: "Sequence[Agent]") -> list[FunctionTool]:
    """Build one delegation :class:`FunctionTool` per subagent.

    Each tool is named ``delegate_to_<name>`` (derived from the subagent's
    ``name`` or ``role``), takes a single ``task`` string, runs the subagent via
    :meth:`Agent.arun`, and returns the subagent's text output. Duplicate names
    are disambiguated with a numeric suffix so every tool name stays unique.
    """
    tools: list[FunctionTool] = []
    used: dict[str, int] = {}

    for index, sub in enumerate(subagents):
        name = delegation_tool_name(sub, index)
        count = used.get(name, 0)
        used[name] = count + 1
        if count:
            name = f"{name}_{count + 1}"

        role = getattr(sub, "_role", "") or getattr(sub, "_name", "") or "specialist"
        goal = getattr(sub, "_goal", "") or ""
        description = f"Delegate a self-contained task to the {role} subagent."
        if goal:
            description += f" Its goal: {goal}."

        def _build(sub_agent: "Agent"):
            async def delegate(task: str) -> str:
                """Delegate a task to this subagent and return its text response."""
                result = await sub_agent.arun(task)
                return result.output.text()

            return delegate

        tools.append(
            FunctionTool(_build(sub), name=name, description=description, idempotent=True)
        )

    return tools
