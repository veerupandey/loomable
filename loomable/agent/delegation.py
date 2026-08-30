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

__all__ = [
    "make_delegation_tools",
    "format_member_roster",
    "delegation_tool_names",
    "spawn_specialist",
]


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


def delegation_tool_names(subagents: list["Agent"]) -> list[tuple["Agent", str]]:
    """Return ``(member, delegate_to_<slug>)`` pairs matching :func:`make_delegation_tools`."""
    pairs: list[tuple["Agent", str]] = []
    used_slugs: set[str] = set()
    for index, subagent in enumerate(subagents):
        label = _subagent_label(subagent, index)
        slug = _unique_slug(label, used_slugs)
        pairs.append((subagent, f"delegate_to_{slug}"))
    return pairs


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
        Maximum nesting depth for nested subagents (default 4). Depth ``0`` is
        the top-level parent; a child invoked via ``delegate_to_*`` runs at
        ``depth + 1`` and rebuilds its own nested tools with that depth.
    depth:
        Current nesting depth (internal / propagated into children).
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
            prev_depth = getattr(_subagent, "_delegation_depth", 0)
            prev_chain_max = getattr(_subagent, "_delegation_max_depth", None)
            target_depth = depth + 1
            if target_depth != prev_depth:
                _subagent._delegation_depth = target_depth
                _subagent._delegation_max_depth = max_depth
                _subagent._built = None
            try:
                result = await _subagent.arun(task)
                call_count["n"] += 1
                return result.output.text()
            except Exception as exc:  # noqa: BLE001 - isolate subagent failures
                return f"Subagent '{_role}' failed: {exc}"
            finally:
                if target_depth != prev_depth:
                    _subagent._delegation_depth = prev_depth
                    if prev_chain_max is None:
                        if hasattr(_subagent, "_delegation_max_depth"):
                            delattr(_subagent, "_delegation_max_depth")
                    else:
                        _subagent._delegation_max_depth = prev_chain_max
                    _subagent._built = None

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
    memory: Any | None = None,
    knowledge: list[str] | None = None,
    knowledge_base: Any = None,
    retrievers: list[Any] | None = None,
    embedder: Any = None,
    skills: list[Any] | None = None,
    tool_hooks: list[Any] | None = None,
    think_tool: bool = False,
    require_tools: list[str] | None = None,
    strict_require_tools: bool | None = None,
    require_confirmation: list[str] | None = None,
    approver: Any | None = None,
    resilience: Any | None = None,
    tool_timeout: float | None = None,
    tool_concurrency: int | None = None,
    max_tool_iterations: int | None = None,
    token_budget: int | None = None,
    max_run_tokens: int | None = None,
    discovery: bool | None = None,
    discovery_core_tools: list[str] | None = None,
    defer_local_tools: bool | None = None,
    lazy_mcp: bool | None = None,
    activation_allowlist: list[str] | None = None,
    activation_denylist: list[str] | None = None,
) -> str:
    """Create an ephemeral specialist Agent, run ``task``, and discard it.

    Enterprise spawn pattern — no long-lived registration required::

        text = await spawn_specialist(
            model=provider,
            role="Cert Auditor",
            task="Review CHG-55219 for pool saturation risk",
        )

    Optional kwargs (skills, tool_hooks/offload, memory, budgets) let deep agents
    share research infrastructure with specialists. Passing ``discovery=True``
    (and optionally ``discovery_core_tools`` / ``defer_local_tools`` /
    ``lazy_mcp`` / ``activation_allowlist`` / ``activation_denylist``) wires
    progressive capability discovery into the specialist too, so a large
    shared toolset (research kit, images, MCP) doesn't blow its schema budget.
    """
    from .builder import Agent

    kwargs: dict[str, Any] = {
        "model": model,
        "role": role,
        "goal": goal or f"Complete tasks as {role}",
        "instructions": instructions or f"You are {role}. Be concise and factual.",
        "tools": tools or [],
        "think_tool": think_tool,
    }
    if modalities is not None:
        kwargs["modalities"] = modalities
    if note_store is not None:
        kwargs["note_store"] = note_store
        kwargs["memory_tool"] = memory_tool
    if memory is not None:
        kwargs["memory"] = memory
    if knowledge is not None:
        kwargs["knowledge"] = knowledge
    if knowledge_base is not None:
        kwargs["knowledge_base"] = knowledge_base
    if retrievers is not None:
        kwargs["retrievers"] = retrievers
    if embedder is not None:
        kwargs["embedder"] = embedder
    if skills is not None:
        kwargs["skills"] = skills
    if tool_hooks is not None:
        kwargs["tool_hooks"] = tool_hooks
    if require_tools is not None:
        kwargs["require_tools"] = require_tools
    if strict_require_tools:
        kwargs["strict_require_tools"] = True
    if require_confirmation is not None:
        kwargs["require_confirmation"] = require_confirmation
    if approver is not None:
        kwargs["approver"] = approver
    if resilience is not None:
        kwargs["resilience"] = resilience
    if tool_timeout is not None:
        kwargs["tool_timeout"] = tool_timeout
    if tool_concurrency is not None:
        kwargs["tool_concurrency"] = tool_concurrency
    if max_tool_iterations is not None:
        kwargs["max_tool_iterations"] = max_tool_iterations
    if token_budget is not None:
        kwargs["token_budget"] = token_budget
    if max_run_tokens is not None:
        kwargs["max_run_tokens"] = max_run_tokens
    if discovery is not None:
        kwargs["discovery"] = discovery
    if discovery_core_tools is not None:
        kwargs["discovery_core_tools"] = discovery_core_tools
    if defer_local_tools is not None:
        kwargs["defer_local_tools"] = defer_local_tools
    if lazy_mcp is not None:
        kwargs["lazy_mcp"] = lazy_mcp
    if activation_allowlist is not None:
        kwargs["activation_allowlist"] = activation_allowlist
    if activation_denylist is not None:
        kwargs["activation_denylist"] = activation_denylist
    agent = Agent(**kwargs)
    result = await agent.arun(task)
    return result.output.text()
