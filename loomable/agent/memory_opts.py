"""Shared Agent memory kwargs — Team / Case / Flow must accept the same set.

Conversation (L1/L2): ``session_id``, ``resume``, ``session_store`` / ``memory_backend``,
window / compaction knobs.

Long-term (L3): ``note_store``, ``memory_tool``, ``knowledge`` / ``embedder``.

Searchable knowledge: ``knowledge_base`` (vector DB) and ``retrievers``.

Flow/Workflow ``memory=`` (TieredMemoryStore) and ``checkpointer`` are separate
concerns and are intentionally not in this set.
"""

from __future__ import annotations

from typing import Any

# Keys forwarded identically on Agent, Team (parent), and Case (role agents).
MEMORY_KEYS: tuple[str, ...] = (
    "memory",
    "session_id",
    "user_id",
    "scopes",
    "resume",
    "use_memory",
    "memory_window",
    "compaction_threshold",
    "use_llm_summarizer",
    "session_store",
    "memory_backend",
    "note_store",
    "memory_tool",
    "knowledge",
    "knowledge_base",
    "embedder",
    "knowledge_top_k",
    "retrievers",
)

# Attribute names on Agent for the corresponding kwargs.
_AGENT_ATTR: dict[str, str] = {
    "memory": "_memory_bundle",
    "session_id": "_session_id",
    "user_id": "_user_id",
    "scopes": "_scopes",
    "resume": "_resume",
    "use_memory": "_use_memory",
    "memory_window": "_memory_window",
    "compaction_threshold": "_compaction_threshold",
    "use_llm_summarizer": "_use_llm_summarizer",
    "session_store": "_session_store",
    "memory_backend": "_memory_backend",
    "note_store": "_note_store",
    "memory_tool": "_memory_tool",
    "knowledge": "_knowledge",
    "knowledge_base": "_knowledge_base",
    "embedder": "_embedder",
    "knowledge_top_k": "_knowledge_top_k",
    "retrievers": "_retrievers",
}

__all__ = [
    "MEMORY_KEYS",
    "filter_memory_kwargs",
    "memory_kwargs_from_agent",
    "role_scoped_memory",
    "inherit_agent_knowledge",
    "apply_knowledge_base",
    "inherit_agent_require_tools",
    "apply_require_tools",
]


def filter_memory_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only Agent memory keys that were explicitly provided (not None)."""
    return {k: kwargs[k] for k in MEMORY_KEYS if k in kwargs and kwargs[k] is not None}


def memory_kwargs_from_agent(agent: Any) -> dict[str, Any]:
    """Extract memory configuration from an :class:`~loomable.agent.builder.Agent`."""
    out: dict[str, Any] = {}
    for key, attr in _AGENT_ATTR.items():
        if not hasattr(agent, attr):
            continue
        val = getattr(agent, attr)
        if val is None:
            continue
        out[key] = val
    return out


def role_scoped_memory(
    memory: dict[str, Any],
    *,
    role: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Copy memory kwargs for a Case/Team role without colliding L1/L2 threads.

    Shared ``note_store`` / knowledge / knowledge_base stay the same object.
    Conversation ``session_id`` becomes ``{base}:{role}`` so planner/worker/synth
    do not overwrite each other's chat turns on one store key.
    """
    out = dict(memory)
    base = session_id or out.get("session_id")
    if base and role:
        out["session_id"] = f"{base}:{role}"
        out.pop("resume", None)
    return out


def inherit_agent_knowledge(
    agent: Any,
    *,
    knowledge_base: Any = None,
    retrievers: Any = None,
    embedder: Any = None,
) -> None:
    """Fill missing knowledge_base / retrievers / embedder on an Agent builder."""
    if agent is None:
        return
    changed = False
    if knowledge_base is not None and getattr(agent, "_knowledge_base", None) is None:
        if hasattr(agent, "_knowledge_base"):
            agent._knowledge_base = knowledge_base
            changed = True
    if retrievers is not None and getattr(agent, "_retrievers", None) is None:
        if hasattr(agent, "_retrievers"):
            agent._retrievers = list(retrievers)
            changed = True
    if embedder is not None and getattr(agent, "_embedder", None) is None:
        if hasattr(agent, "_embedder"):
            agent._embedder = embedder
            changed = True
    if changed and hasattr(agent, "_built"):
        agent._built = None
        if hasattr(agent, "_knowledge_retrievers"):
            agent._knowledge_retrievers = None


def apply_knowledge_base(
    obj: Any,
    *,
    knowledge_base: Any = None,
    retrievers: Any = None,
    embedder: Any = None,
    _seen: set[int] | None = None,
) -> None:
    """Walk Agent / Team / Step / Workflow / Flow graphs and inherit a shared KB."""
    if obj is None or (
        knowledge_base is None and retrievers is None and embedder is None
    ):
        return
    seen = _seen if _seen is not None else set()
    if isinstance(obj, (list, tuple)):
        for item in obj:
            apply_knowledge_base(
                item,
                knowledge_base=knowledge_base,
                retrievers=retrievers,
                embedder=embedder,
                _seen=seen,
            )
        return
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    kwargs = dict(
        knowledge_base=knowledge_base, retrievers=retrievers, embedder=embedder
    )
    if hasattr(obj, "_knowledge_base") and callable(getattr(obj, "build", None)):
        inherit_agent_knowledge(obj, **kwargs)
        for member in getattr(obj, "_subagents", None) or []:
            apply_knowledge_base(member, _seen=seen, **kwargs)
        for member in getattr(obj, "_members", None) or []:
            apply_knowledge_base(member, _seen=seen, **kwargs)
        return
    inner = getattr(obj, "_agent", None)
    if inner is not None and inner is not obj:
        apply_knowledge_base(inner, _seen=seen, **kwargs)
    runnable = getattr(obj, "runnable", None)
    if runnable is not None and runnable is not obj:
        apply_knowledge_base(runnable, _seen=seen, **kwargs)
    for attr in ("_members", "_steps", "_then_steps", "_else_steps"):
        child = getattr(obj, attr, None)
        if child:
            apply_knowledge_base(child, _seen=seen, **kwargs)
    nodes = getattr(obj, "_nodes", None)
    if isinstance(nodes, dict):
        apply_knowledge_base(list(nodes.values()), _seen=seen, **kwargs)
    for attr in ("_body", "_true", "_false"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            apply_knowledge_base(child, _seen=seen, **kwargs)


def inherit_agent_require_tools(
    agent: Any,
    *,
    require_tools: list[str] | None = None,
    strict_require_tools: bool | None = None,
    overwrite: bool = False,
) -> None:
    """Fill missing ``require_tools`` / ``strict_require_tools`` on an Agent."""
    if agent is None or not hasattr(agent, "_require_tools"):
        return
    changed = False
    if require_tools:
        existing = list(getattr(agent, "_require_tools", None) or [])
        if overwrite or not existing:
            agent._require_tools = list(require_tools)
            changed = True
    if strict_require_tools is True and not getattr(agent, "_strict_require_tools", False):
        agent._strict_require_tools = True
        changed = True
    if changed and hasattr(agent, "_built"):
        agent._built = None


def apply_require_tools(
    obj: Any,
    *,
    require_tools: list[str] | None = None,
    strict_require_tools: bool | None = None,
    _seen: set[int] | None = None,
) -> None:
    """Walk Agent / Team / Step / Workflow graphs and inherit require_tools."""
    if obj is None or (not require_tools and not strict_require_tools):
        return
    seen = _seen if _seen is not None else set()
    if isinstance(obj, (list, tuple)):
        for item in obj:
            apply_require_tools(
                item,
                require_tools=require_tools,
                strict_require_tools=strict_require_tools,
                _seen=seen,
            )
        return
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    kwargs = dict(
        require_tools=require_tools, strict_require_tools=strict_require_tools
    )
    if hasattr(obj, "_require_tools") and callable(getattr(obj, "build", None)):
        inherit_agent_require_tools(obj, **kwargs)
        for member in getattr(obj, "_subagents", None) or []:
            apply_require_tools(member, _seen=seen, **kwargs)
        for member in getattr(obj, "_members", None) or []:
            apply_require_tools(member, _seen=seen, **kwargs)
        return
    inner = getattr(obj, "_agent", None)
    if inner is not None and inner is not obj:
        apply_require_tools(inner, _seen=seen, **kwargs)
    runnable = getattr(obj, "runnable", None)
    if runnable is not None and runnable is not obj:
        apply_require_tools(runnable, _seen=seen, **kwargs)
    for attr in ("_members", "_steps", "_then_steps", "_else_steps"):
        child = getattr(obj, attr, None)
        if child:
            apply_require_tools(child, _seen=seen, **kwargs)
    nodes = getattr(obj, "_nodes", None)
    if isinstance(nodes, dict):
        apply_require_tools(list(nodes.values()), _seen=seen, **kwargs)
    for attr in ("_body", "_true", "_false"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            apply_require_tools(child, _seen=seen, **kwargs)
