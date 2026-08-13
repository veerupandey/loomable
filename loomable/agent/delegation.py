"""loomable.agent.delegation - Runtime subagent delegation tools.

Each configured subagent becomes a :class:`~loomable.agent.tools.FunctionTool`
named ``delegate_to_<role_slug>`` so the parent LLM can delegate work at runtime.
Failures are isolated: exceptions become error strings instead of crashing the parent.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .tools import FunctionTool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .builder import Agent

__all__ = ["make_delegation_tools"]


def _subagent_label(agent: "Agent", index: int) -> str:
    """Human-readable label for a subagent (role, name, or fallback)."""
    role = getattr(agent, "_role", "") or ""
    if role:
        return role
    name = getattr(agent, "_name", "") or ""
    if name:
        return name
    return f"subagent_{index}"


def _role_slug(label: str) -> str:
    """Convert a role label to a stable tool-name slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "subagent"


def _unique_slug(label: str, used: set[str]) -> str:
    """Return a unique slug, suffixing with _2, _3, ... on collision."""
    base = _role_slug(label)
    slug = base
    counter = 2
    while slug in used:
        slug = f"{base}_{counter}"
        counter += 1
    used.add(slug)
    return slug


def format_member_roster(subagents: list["Agent"]) -> str:
    """Format subagents as a bullet list of delegate tool names and roles."""
    lines: list[str] = []
    used_slugs: set[str] = set()
    for index, subagent in enumerate(subagents):
        label = _subagent_label(subagent, index)
        slug = _unique_slug(label, used_slugs)
        role = getattr(subagent, "_role", "") or label
        goal = getattr(subagent, "_goal", "") or ""
        line = f"- delegate_to_{slug}: {role}"
        if goal:
            line += f" — {goal}"
        lines.append(line)
    return "\n".join(lines)


def _build_description(agent: "Agent", role: str) -> str:
    """Build a tool description from the subagent's role and goal."""
    goal = getattr(agent, "_goal", "") or ""
    parts = [f"Delegate a task to {role}."]
    if goal:
        parts.append(f"Their specialty: {goal}")
    parts.append("Provide a clear task description; returns the subagent's text response.")
    return " ".join(parts)


def make_delegation_tools(subagents: list["Agent"]) -> list[FunctionTool]:
    """Create delegation tools for each subagent.

    Each subagent becomes a :class:`FunctionTool` named ``delegate_to_<role_slug>``
    (for example ``delegate_to_researcher``). The tool runs ``subagent.arun(task)``
    and returns the subagent's text output. Nested subagents are supported because
    each subagent retains its own ``subagents`` configuration.

    Parameters
    ----------
    subagents:
        Builder :class:`~loomable.agent.builder.Agent` instances to expose as tools.

    Returns
    -------
    list[FunctionTool]
        One delegation tool per subagent.
    """
    tools: list[FunctionTool] = []
    used_slugs: set[str] = set()

    for index, subagent in enumerate(subagents):
        role = _subagent_label(subagent, index)
        slug = _unique_slug(role, used_slugs)
        tool_name = f"delegate_to_{slug}"
        description = _build_description(subagent, role)

        async def _delegate(
            task: str,
            *,
            _subagent: "Agent" = subagent,
            _role: str = role,
        ) -> str:
            """Delegate a task to this subagent and return its text response."""
            try:
                result = await _subagent.arun(task)
                return result.output.text()
            except Exception as exc:  # noqa: BLE001 - isolate subagent failures
                return f"Subagent '{_role}' failed: {exc}"

        _delegate.__name__ = tool_name
        _delegate.__doc__ = f"Delegate a task to {role}. Returns their text response."

        tools.append(
            FunctionTool(
                _delegate,
                name=tool_name,
                description=description,
                idempotent=False,
            )
        )

    return tools
