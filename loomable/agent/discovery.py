"""Runtime capability discovery — skills, local tools, and MCP.

Industry pattern: advertise a small always-on surface, then let the model
``search_*`` / ``load_*`` / ``activate_*`` to pull more into context.

Usage (wired by :class:`~loomable.agent.builder.Agent`)::

    search_skills(query="research")
    load_skill(name="research")
    search_tools(query="pdf")
    search_mcp(query="github")
    activate_tool(name="create_issue")  # MCP or catalogued local tool
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loomable.agent.tools import FunctionTool
from loomable.kernel.skills import SkillLoader, SkillManifest

__all__ = [
    "CapabilityCatalog",
    "DISCOVERY_META_TOOL_NAMES",
    "DISCOVERY_SYSTEM_NOTE",
    "DiscoveryRuntime",
    "SkillStub",
    "ToolStub",
    "format_skill_catalog_for_prompt",
    "make_discovery_tools",
    "rank_match",
    "tool_schema_payload",
]


@dataclass
class SkillStub:
    name: str
    description: str
    path: Path
    loaded: bool = False


@dataclass
class ToolStub:
    name: str
    description: str
    source: str  # "local" | "mcp" | "skill"
    server_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    activated: bool = False
    # Deferred MCP activation payload
    mcp_client: Any | None = None
    mcp_session: Any | None = None


@dataclass
class CapabilityCatalog:
    """Searchable index of skills / tools / MCP tools."""

    skills: list[SkillStub] = field(default_factory=list)
    tools: list[ToolStub] = field(default_factory=list)

    def skill_by_name(self, name: str) -> SkillStub | None:
        key = (name or "").strip().lower()
        for s in self.skills:
            if s.name.lower() == key:
                return s
        return None

    def tool_by_name(self, name: str) -> ToolStub | None:
        key = (name or "").strip().lower()
        for t in self.tools:
            if t.name.lower() == key:
                return t
        return None


def tool_schema_payload(tool: Any) -> dict[str, Any]:
    """OpenAI-style function schema for an activated tool (or best-effort stub)."""
    if hasattr(tool, "schema") and callable(tool.schema):
        try:
            sch = tool.schema()
            if isinstance(sch, dict):
                return sch
        except Exception:  # noqa: BLE001
            pass
    return {
        "type": "function",
        "function": {
            "name": getattr(tool, "name", "") or "",
            "description": getattr(tool, "description", "") or "",
            "parameters": getattr(tool, "parameters", None)
            or {"type": "object", "properties": {}},
        },
    }


DISCOVERY_META_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "search_skills",
        "load_skill",
        "search_tools",
        "search_mcp",
        "activate_tool",
    }
)


def format_skill_catalog_for_prompt(skills: list[SkillStub]) -> str:
    """Level-1 skill disclosure: names + descriptions only."""
    if not skills:
        return ""
    lines = ["Available skills (call load_skill(name) to load full instructions):"]
    for s in skills:
        status = "loaded" if s.loaded else "not loaded"
        desc = (s.description or "").strip() or "(no description)"
        lines.append(f"- {s.name} [{status}]: {desc}")
    return "\n".join(lines)


def rank_match(query: str, name: str, description: str = "") -> float:
    """Simple relevance score for discovery ranking."""
    q = (query or "").strip().lower()
    if not q:
        return 0.1
    blob = f"{name} {description}".lower()
    score = 0.0
    if q == name.lower():
        score += 5.0
    if name.lower().startswith(q):
        score += 2.0
    if q in name.lower():
        score += 1.5
    for token in re.split(r"\W+", q):
        if not token:
            continue
        if token in name.lower():
            score += 1.0
        if token in description.lower():
            score += 0.5
        if token in blob:
            score += 0.25
    return score


class DiscoveryRuntime:
    """Mutable runtime used by discovery tools to activate capabilities."""

    def __init__(
        self,
        catalog: CapabilityCatalog,
        *,
        skill_loader: SkillLoader | None = None,
        tool_runtime: Any | None = None,
        on_skill_body: Callable[[str, str], None] | None = None,
    ) -> None:
        self.catalog = catalog
        self.skill_loader = skill_loader or SkillLoader()
        self.tool_runtime = tool_runtime
        self.on_skill_body = on_skill_body
        self.loaded_skill_bodies: list[tuple[str, str]] = []
        self._pending_local: dict[str, Any] = {}  # name -> Tool
        # Mid-run skill bodies to inject as system messages after tool results.
        self._pending_prompt_injections: list[tuple[str, str]] = []

    def bind_runtime(self, tool_runtime: Any) -> None:
        self.tool_runtime = tool_runtime

    def register_pending_local(self, name: str, tool: Any) -> None:
        self._pending_local[name] = tool

    def drain_prompt_injections(self) -> list[tuple[str, str]]:
        items = list(self._pending_prompt_injections)
        self._pending_prompt_injections.clear()
        return items

    def _register_tool(self, tool: Any) -> None:
        if self.tool_runtime is None:
            return
        tools = getattr(self.tool_runtime, "_tools", None)
        if isinstance(tools, dict):
            tools[tool.name] = tool

    def search_skills(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 8), 20))
        ranked = sorted(
            self.catalog.skills,
            key=lambda s: rank_match(query, s.name, s.description),
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for s in ranked:
            score = rank_match(query, s.name, s.description)
            if (query or "").strip() and score <= 0:
                continue
            out.append(
                {
                    "name": s.name,
                    "description": s.description,
                    "loaded": s.loaded,
                    "path": str(s.path),
                    "score": round(score, 3),
                }
            )
            if len(out) >= limit:
                break
        if not out and not (query or "").strip():
            out = [
                {
                    "name": s.name,
                    "description": s.description,
                    "loaded": s.loaded,
                    "path": str(s.path),
                    "score": 0.0,
                }
                for s in self.catalog.skills[:limit]
            ]
        return out

    def search_tools(self, query: str, limit: int = 8, *, source: str = "") -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 8), 30))
        src = (source or "").strip().lower()
        candidates = [
            t
            for t in self.catalog.tools
            if not src or t.source == src
        ]
        ranked = sorted(
            candidates,
            key=lambda t: rank_match(query, t.name, t.description),
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for t in ranked:
            score = rank_match(query, t.name, t.description)
            if (query or "").strip() and score <= 0:
                continue
            out.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "source": t.source,
                    "server_id": t.server_id,
                    "activated": t.activated,
                    "score": round(score, 3),
                }
            )
            if len(out) >= limit:
                break
        if not out and not (query or "").strip():
            out = [
                {
                    "name": t.name,
                    "description": t.description,
                    "source": t.source,
                    "server_id": t.server_id,
                    "activated": t.activated,
                    "score": 0.0,
                }
                for t in candidates[:limit]
            ]
        return out

    def search_mcp(self, query: str, limit: int = 8, server_id: str = "") -> list[dict[str, Any]]:
        sid = (server_id or "").strip()
        results = self.search_tools(query, limit=limit, source="mcp")
        if sid:
            results = [r for r in results if (r.get("server_id") or "") == sid]
        return results

    def load_skill(self, name: str) -> dict[str, Any]:
        stub = self.catalog.skill_by_name(name)
        if stub is None:
            return {
                "ok": False,
                "error": f"unknown skill: {name}",
                "available": [s.name for s in self.catalog.skills],
            }
        if stub.loaded:
            return {"ok": True, "skill": stub.name, "already_loaded": True}

        # Discover the skill dir (direct) then load
        manifests = self.skill_loader.discover([stub.path])
        if not manifests:
            # parent catalog containing this skill name
            parent = stub.path.parent
            manifests = [
                m for m in self.skill_loader.discover([parent]) if m.name == stub.name
            ]
        if not manifests:
            return {"ok": False, "error": f"could not discover skill: {name}"}
        try:
            loaded = self.skill_loader.load(manifests[0])
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

        body = (getattr(loaded, "body", None) or "").strip()
        for script_tool in loaded.get_tools():
            self._register_tool(script_tool)
            # mark/update catalog
            existing = self.catalog.tool_by_name(script_tool.name)
            if existing:
                existing.activated = True
                existing.source = "skill"
            else:
                self.catalog.tools.append(
                    ToolStub(
                        name=script_tool.name,
                        description=getattr(script_tool, "description", "") or "",
                        source="skill",
                        activated=True,
                    )
                )

        stub.loaded = True
        if body:
            self.loaded_skill_bodies.append((stub.name, body))
            self._pending_prompt_injections.append((stub.name, body))
            if self.on_skill_body is not None:
                self.on_skill_body(stub.name, body)
        return {
            "ok": True,
            "skill": stub.name,
            "description": stub.description,
            "body": body[:12_000] if body else "",
            "body_chars": len(body),
            "tools_registered": [t.name for t in loaded.get_tools()],
            "hint": "Skill instructions are now active; use newly registered tools if any.",
        }

    def activate_tool(self, name: str) -> dict[str, Any]:
        stub = self.catalog.tool_by_name(name)
        if stub is None:
            return {
                "ok": False,
                "error": f"unknown tool: {name}",
                "hint": "Call search_tools / search_mcp first",
            }
        if stub.activated:
            return {"ok": True, "name": stub.name, "already_activated": True}

        if stub.source == "mcp":
            if stub.mcp_client is None or stub.mcp_session is None:
                return {"ok": False, "error": f"MCP tool {name} has no live session"}
            from loomable.agent.tools import MCPTool

            mcp_tool = MCPTool(
                name=stub.name,
                description=stub.description,
                parameters=stub.parameters
                or {"type": "object", "properties": {}},
                mcp_client=stub.mcp_client,
                session=stub.mcp_session,
            )
            self._register_tool(mcp_tool)
            stub.activated = True
            return {
                "ok": True,
                "name": stub.name,
                "source": "mcp",
                "server_id": stub.server_id,
                "schema": tool_schema_payload(mcp_tool),
                "hint": "Tool is now callable; schema included below.",
            }

        pending = self._pending_local.get(stub.name)
        if pending is not None:
            self._register_tool(pending)
            stub.activated = True
            return {
                "ok": True,
                "name": stub.name,
                "source": stub.source,
                "schema": tool_schema_payload(pending),
                "hint": "Tool is now callable; schema included below.",
            }

        # Already in runtime (eager mode)
        if self.tool_runtime is not None:
            tools = getattr(self.tool_runtime, "_tools", {}) or {}
            if stub.name in tools:
                stub.activated = True
                return {
                    "ok": True,
                    "name": stub.name,
                    "already_activated": True,
                    "schema": tool_schema_payload(tools[stub.name]),
                }

        return {
            "ok": False,
            "error": f"tool {name} cannot be activated (no deferred handle)",
        }


def catalog_from_skill_manifests(manifests: list[SkillManifest]) -> list[SkillStub]:
    return [
        SkillStub(
            name=m.name,
            description=m.description or "",
            path=m.body_path.parent,
        )
        for m in manifests
    ]


def make_discovery_tools(runtime: DiscoveryRuntime) -> list[FunctionTool]:
    """Build the always-on discovery meta-tools."""

    async def search_skills(query: str = "", limit: int = 8) -> str:
        """Search available skills by name/description (progressive disclosure)."""
        hits = runtime.search_skills(query, limit=limit)
        return json.dumps({"query": query, "skills": hits}, ensure_ascii=False)

    async def load_skill(name: str) -> str:
        """Load a skill by name: inject instructions and register its script tools."""
        return json.dumps(runtime.load_skill(name), ensure_ascii=False)

    async def search_tools(query: str = "", limit: int = 8, source: str = "") -> str:
        """Search local / skill / catalogued tools without advertising full schemas."""
        hits = runtime.search_tools(query, limit=limit, source=source)
        return json.dumps({"query": query, "tools": hits}, ensure_ascii=False)

    async def search_mcp(query: str = "", limit: int = 8, server_id: str = "") -> str:
        """Search MCP tools discovered from connected servers."""
        hits = runtime.search_mcp(query, limit=limit, server_id=server_id)
        return json.dumps(
            {"query": query, "server_id": server_id or None, "tools": hits},
            ensure_ascii=False,
        )

    async def activate_tool(name: str) -> str:
        """Activate a catalogued tool (especially deferred MCP tools) for calling."""
        return json.dumps(runtime.activate_tool(name), ensure_ascii=False)

    return [
        FunctionTool(
            search_skills,
            name="search_skills",
            description="Search skill catalog by query; returns name/description stubs.",
        ),
        FunctionTool(
            load_skill,
            name="load_skill",
            description="Load a skill by name into this run (instructions + script tools).",
            idempotent=False,
        ),
        FunctionTool(
            search_tools,
            name="search_tools",
            description="Search available tools (local/skill/mcp catalog) by query.",
        ),
        FunctionTool(
            search_mcp,
            name="search_mcp",
            description="Search MCP tools from connected servers by query.",
        ),
        FunctionTool(
            activate_tool,
            name="activate_tool",
            description="Activate a deferred/catalogued tool (e.g. MCP) so it can be called.",
            idempotent=False,
        ),
    ]


DISCOVERY_SYSTEM_NOTE = """\
Capability discovery: use search_skills / load_skill for workflows, search_tools
for deferred local tools, search_mcp + activate_tool for MCP tools. Prefer
searching before assuming a tool exists. After load/activate, call the tool
(schemas for activated tools are returned by activate_tool and also appear on
the next turn).
"""
