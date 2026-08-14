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

__all__ = ["make_delegation_tools", "format_member_roster", "spawn_specialist"]


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


def make_delegation_tools(
    subagents: list["Agent"],
    *,
    max_delegations: int | None = None,
    max_depth: int = 4,
    depth: int = 0,
) -> list[FunctionTool]:
    """Create delegation tools for each subagent.

    Parameters
    ----------
    subagents:
        Builder :class:`~loomable.agent.builder.Agent` instances to expose as tools.
    max_delegations:
        Soft budget: after this many successful delegate calls in one parent run,
        further calls return a budget error string (does not crash the parent).
    max_depth:
        Maximum nesting depth for nested subagents (default 4).
    depth:
        Current nesting depth (internal).
    """
    tools: list[FunctionTool] = []
    used_slugs: set[str] = set()
    call_count = {"n": 0}

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
            if depth >= max_depth:
                return (
                    f"Subagent '{_role}' skipped: max delegation depth "
                    f"{max_depth} reached."
                )
            if max_delegations is not None and call_count["n"] >= max_delegations:
                return (
                    f"Subagent '{_role}' skipped: max_delegations="
                    f"{max_delegations} budget exhausted."
                )
            try:
                result = await _subagent.arun(task)
                call_count["n"] += 1
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


async def spawn_specialist(
    *,
    model: Any,
    role: str,
    task: str,
    goal: str = "",
    instructions: str | None = None,
    tools: list[Any] | None = None,
    modalities: str | None = None,
    note_store: Any | None = None,
    memory_tool: bool = False,
    knowledge: list[str] | None = None,
    embedder: Any = None,
) -> str:
    """Create an ephemeral specialist Agent, run ``task``, and discard it.

    Enterprise spawn pattern — no long-lived registration required::

        text = await spawn_specialist(
            model=provider,
            role="Cert Auditor",
            task="Review CHG-55219 for pool saturation risk",
        )

    Optional L3 kwargs (``note_store`` / knowledge) match Agent so Case spawn
    dispatch shares long-term memory with the parent Case.
    """
    from .builder import Agent

    kwargs: dict[str, Any] = {
        "model": model,
        "role": role,
        "goal": goal or f"Complete tasks as {role}",
        "instructions": instructions or f"You are {role}. Be concise and factual.",
        "tools": tools or [],
    }
    if modalities is not None:
        kwargs["modalities"] = modalities
    if note_store is not None:
        kwargs["note_store"] = note_store
        kwargs["memory_tool"] = memory_tool
    if knowledge is not None:
        kwargs["knowledge"] = knowledge
    if embedder is not None:
        kwargs["embedder"] = embedder
    agent = Agent(**kwargs)
    result = await agent.arun(task)
    return result.output.text()
