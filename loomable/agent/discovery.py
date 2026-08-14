"""Runtime capability discovery — skills, local tools, and MCP.

Industry pattern: advertise a small always-on surface, then let the model
``search_*`` / ``load_*`` / ``activate_*`` to pull more into context.

Usage (wired by :class:`~loomable.agent.builder.Agent`)::

    search_skills(query="research")
    load_skill(name="research")
    search_tools(query="pdf")
    search_mcp(query="github")
    activate_tool(name="create_issue")  # MCP or catalogued local tool

P1/P2 additions (see ``docs/COMPETITIVE.md``):

    search_namespaces(query="mcp")           # browse tool groups
    activate_mcp_server(server_id="github")  # lazy connect on demand
    refresh_capabilities()                   # re-scan skills / MCP mid-run
    list_skill_resources(name="research")    # scripts/references/assets
    read_skill_resource(name="research", path="references/checklist.md")
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from loomable.agent.tools import FunctionTool
from loomable.kernel.skills import SkillLoader, SkillManifest

__all__ = [
    "CapabilityCatalog",
    "DISCOVERY_META_TOOL_NAMES",
    "DISCOVERY_SYSTEM_NOTE",
    "DiscoveryRuntime",
    "NamespaceStub",
    "ServerStub",
    "SkillStub",
    "ToolStub",
    "catalog_from_skill_manifests",
    "format_skill_catalog_for_prompt",
    "make_discovery_tools",
    "rank_bm25",
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
    # Tool namespace / group (e.g. "mcp:github", "images") for search_namespaces.
    namespace: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ServerStub:
    """A configured MCP server — connected or catalogued for lazy activation."""

    server_id: str
    description: str = ""
    connected: bool = False
    # Raw MCPServerSpec (dict) retained so ``activate_mcp_server`` can connect
    # on demand when the server was catalogued lazily (not connected at build).
    spec: Any | None = None
    tool_count: int = 0


@dataclass
class NamespaceStub:
    """A named group of tools (custom group or an auto ``mcp:<server_id>``)."""

    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)


@dataclass
class CapabilityCatalog:
    """Searchable index of skills / tools / MCP servers / namespaces."""

    skills: list[SkillStub] = field(default_factory=list)
    tools: list[ToolStub] = field(default_factory=list)
    servers: list[ServerStub] = field(default_factory=list)
    namespaces: list[NamespaceStub] = field(default_factory=list)

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

    def server_by_id(self, server_id: str) -> ServerStub | None:
        key = (server_id or "").strip().lower()
        for s in self.servers:
            if s.server_id.lower() == key:
                return s
        return None

    def namespace_by_name(self, name: str) -> NamespaceStub | None:
        key = (name or "").strip().lower()
        for n in self.namespaces:
            if n.name.lower() == key:
                return n
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
        "search_namespaces",
        "activate_tool",
        "activate_mcp_server",
        "refresh_capabilities",
        "list_skill_resources",
        "read_skill_resource",
    }
)

# Level-3 skill resources that can be listed/read on demand (deepagents-style
# progressive disclosure beyond the SKILL.md body).
_SKILL_RESOURCE_DIRS: tuple[str, ...] = ("scripts", "references", "assets")


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
    """Simple relevance score for discovery ranking (kept for back-compat)."""
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


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", (text or "").lower()) if t]


def _build_bm25_df(docs: list[tuple[str, str]]) -> tuple[dict[str, int], float, int]:
    """Document-frequency stats over a catalog slice, for BM25 idf terms."""
    df: dict[str, int] = {}
    lengths: list[int] = []
    for name, description in docs:
        tokens = _tokenize(f"{name} {description}")
        lengths.append(len(tokens) or 1)
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    n_docs = max(1, len(docs))
    avg_len = (sum(lengths) / n_docs) if lengths else 1.0
    return df, avg_len, n_docs


def rank_bm25(
    query: str,
    name: str,
    description: str = "",
    *,
    df: dict[str, int] | None = None,
    n_docs: int = 1,
    avg_len: float = 1.0,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25-lite relevance score using catalog document-frequency stats.

    Ranks a single ``(name, description)`` document against ``query`` using a
    standard Okapi BM25 term-weighting formula. ``df`` / ``n_docs`` / ``avg_len``
    are typically computed once per search over the whole candidate slice (see
    :meth:`DiscoveryRuntime._bm25_rank`) so common terms across the catalog
    (e.g. "tool", "search") are weighted down relative to distinctive ones.

    Falls back to a single-document approximation when ``df`` is omitted so it
    stays usable standalone, mirroring :func:`rank_match`.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.1
    doc_tokens = _tokenize(f"{name} {description}")
    if not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    tf: dict[str, int] = {}
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1

    df = df or {}
    n_docs = max(1, n_docs)
    avg_len = avg_len or 1.0

    score = 0.0
    matched = 0
    for term in q_tokens:
        freq = tf.get(term, 0)
        if freq == 0:
            continue
        matched += 1
        n_t = df.get(term, 1)
        idf = math.log(1.0 + (n_docs - n_t + 0.5) / (n_t + 0.5))
        idf = max(idf, 1e-6)
        denom = freq + k1 * (1 - b + b * (doc_len / avg_len))
        score += idf * (freq * (k1 + 1)) / denom

    if matched == 0:
        return 0.0

    # Stable tie-break for exact/prefix name matches (BM25 alone can rank a
    # long description-heavy doc above an exact-name match on short queries).
    name_l = name.lower()
    q_l = (query or "").strip().lower()
    if name_l == q_l:
        score += 3.0
    elif name_l.startswith(q_l):
        score += 1.0
    elif q_l and q_l in name_l:
        score += 0.5

    return round(score, 6)


def _pattern_match(name: str, pattern: str) -> bool:
    pattern = (pattern or "").strip()
    if not pattern:
        return False
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def _matches_any(name: str, patterns: Sequence[str] | None) -> bool:
    if not patterns:
        return False
    return any(_pattern_match(name, p) for p in patterns)


class DiscoveryRuntime:
    """Mutable runtime used by discovery tools to activate capabilities."""

    def __init__(
        self,
        catalog: CapabilityCatalog,
        *,
        skill_loader: SkillLoader | None = None,
        tool_runtime: Any | None = None,
        on_skill_body: Callable[[str, str], None] | None = None,
        skill_roots: list[Path] | None = None,
        activation_allowlist: Sequence[str] | None = None,
        activation_denylist: Sequence[str] | None = None,
        lazy_mcp: bool = False,
        on_activate_check: Callable[[str], bool | str] | None = None,
    ) -> None:
        self.catalog = catalog
        self.skill_loader = skill_loader or SkillLoader()
        self.tool_runtime = tool_runtime
        self.on_skill_body = on_skill_body
        # Roots re-scanned by refresh_capabilities() to pick up new skills
        # added to disk after the agent was built.
        self.skill_roots: list[Path] = list(skill_roots) if skill_roots else []
        self.activation_allowlist: list[str] = (
            [str(p) for p in activation_allowlist] if activation_allowlist else []
        )
        self.activation_denylist: list[str] = (
            [str(p) for p in activation_denylist] if activation_denylist else []
        )
        self.lazy_mcp = bool(lazy_mcp)
        self.on_activate_check = on_activate_check
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

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _bm25_rank(self, query: str, items: list[Any]) -> list[tuple[Any, float]]:
        """Rank catalog items (skills/tools/namespaces) by BM25-lite score."""
        if not items:
            return []
        docs = [
            (getattr(it, "name", "") or "", getattr(it, "description", "") or "")
            for it in items
        ]
        df, avg_len, n_docs = _build_bm25_df(docs)
        scored = [
            (it, rank_bm25(query, name, description, df=df, n_docs=n_docs, avg_len=avg_len))
            for it, (name, description) in zip(items, docs)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Activation policy
    # ------------------------------------------------------------------

    def _activation_allowed(self, name: str) -> tuple[bool, str | None]:
        """Check ``activation_allowlist`` / ``activation_denylist`` / ``on_activate_check``.

        Denylist wins over allowlist. Patterns support a trailing ``*``
        wildcard (``"fs_*"`` matches any name starting with ``fs_``).
        """
        if _matches_any(name, self.activation_denylist):
            return False, f"tool '{name}' is denylisted for activation"
        if self.activation_allowlist and not _matches_any(name, self.activation_allowlist):
            return False, f"tool '{name}' is not in the activation allowlist"
        if self.on_activate_check is not None:
            try:
                verdict = self.on_activate_check(name)
            except Exception as exc:  # noqa: BLE001
                return False, f"activation check raised: {exc}"
            if verdict is False:
                return False, f"tool '{name}' activation rejected by policy"
            if isinstance(verdict, str) and verdict:
                return False, verdict
        return True, None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_skills(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 8), 20))
        scored = self._bm25_rank(query, self.catalog.skills)
        out: list[dict[str, Any]] = []
        for s, score in scored:
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

    def search_tools(
        self,
        query: str,
        limit: int = 8,
        *,
        source: str = "",
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 8), 30))
        src = (source or "").strip().lower()
        ns = (namespace or "").strip()
        candidates = [
            t
            for t in self.catalog.tools
            if (not src or t.source == src) and (not ns or (t.namespace or "") == ns)
        ]

        def _payload(t: ToolStub, score: float) -> dict[str, Any]:
            return {
                "name": t.name,
                "description": t.description,
                "source": t.source,
                "server_id": t.server_id,
                "namespace": t.namespace,
                "tags": list(t.tags),
                "activated": t.activated,
                "score": round(score, 3),
            }

        scored = self._bm25_rank(query, candidates)
        out: list[dict[str, Any]] = []
        for t, score in scored:
            if (query or "").strip() and score <= 0:
                continue
            out.append(_payload(t, score))
            if len(out) >= limit:
                break
        if not out and not (query or "").strip():
            out = [_payload(t, 0.0) for t in candidates[:limit]]
        return out

    def search_mcp(self, query: str, limit: int = 8, server_id: str = "") -> list[dict[str, Any]]:
        sid = (server_id or "").strip()
        results = self.search_tools(query, limit=limit, source="mcp")
        if sid:
            results = [r for r in results if (r.get("server_id") or "") == sid]

        # Also surface unconnected servers so the model knows to call
        # activate_mcp_server before a tool shows up here (lazy MCP).
        server_hits: list[dict[str, Any]] = []
        q = (query or "").strip()
        for server in self.catalog.servers:
            if server.connected:
                continue
            if sid and server.server_id != sid:
                continue
            score = rank_bm25(query, server.server_id, server.description) if q else 0.0
            if q and score <= 0:
                continue
            server_hits.append(
                {
                    "server_id": server.server_id,
                    "description": server.description,
                    "connected": False,
                    "score": round(score, 3),
                    "hint": (
                        f"Call activate_mcp_server(server_id={server.server_id!r}) "
                        "to connect and catalog its tools."
                    ),
                }
            )
        server_hits.sort(key=lambda r: r["score"], reverse=True)
        return results + server_hits[: max(1, min(int(limit or 8), 30))]

    def search_namespaces(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 8), 30))
        scored = self._bm25_rank(query, self.catalog.namespaces)
        out: list[dict[str, Any]] = []
        for ns, score in scored:
            if (query or "").strip() and score <= 0:
                continue
            out.append(
                {
                    "name": ns.name,
                    "description": ns.description,
                    "tools": list(ns.tools),
                    "score": round(score, 3),
                }
            )
            if len(out) >= limit:
                break
        if not out and not (query or "").strip():
            out = [
                {
                    "name": ns.name,
                    "description": ns.description,
                    "tools": list(ns.tools),
                    "score": 0.0,
                }
                for ns in self.catalog.namespaces[:limit]
            ]
        return out

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

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

    def list_skill_resources(self, name: str) -> dict[str, Any]:
        """List level-3 skill resources: ``SKILL.md`` + scripts/references/assets."""
        stub = self.catalog.skill_by_name(name)
        if stub is None:
            return {
                "ok": False,
                "error": f"unknown skill: {name}",
                "available": [s.name for s in self.catalog.skills],
            }
        try:
            root = stub.path.resolve()
        except OSError:
            root = stub.path
        resources: list[dict[str, Any]] = []
        if (root / "SKILL.md").is_file():
            resources.append({"path": "SKILL.md", "kind": "file"})
        for sub in _SKILL_RESOURCE_DIRS:
            sub_dir = root / sub
            if not sub_dir.is_dir():
                continue
            for path_obj in sorted(sub_dir.rglob("*")):
                if path_obj.is_file():
                    resources.append(
                        {"path": path_obj.relative_to(root).as_posix(), "kind": "file"}
                    )
        return {"ok": True, "skill": stub.name, "resources": resources}

    def read_skill_resource(
        self, name: str, path: str, max_chars: int = 12_000
    ) -> dict[str, Any]:
        """Read one skill resource file, restricted to safe sub-paths.

        Only ``SKILL.md`` or files under ``scripts/``, ``references/``, or
        ``assets/`` may be read, and the resolved path must stay inside the
        skill directory (path-traversal safe).
        """
        stub = self.catalog.skill_by_name(name)
        if stub is None:
            return {
                "ok": False,
                "error": f"unknown skill: {name}",
                "available": [s.name for s in self.catalog.skills],
            }
        rel = (path or "").strip().lstrip("/")
        if not rel:
            return {"ok": False, "error": "path is required"}

        try:
            root = stub.path.resolve()
        except OSError:
            root = stub.path
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return {
                "ok": False,
                "error": "path escapes the skill directory; not allowed",
            }

        top = rel.split("/", 1)[0]
        allowed = rel == "SKILL.md" or top in _SKILL_RESOURCE_DIRS
        if not allowed:
            return {
                "ok": False,
                "error": (
                    "path must be SKILL.md or under scripts/, references/, "
                    "or assets/"
                ),
            }
        if not candidate.is_file():
            return {"ok": False, "error": f"resource not found: {rel}"}

        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return {"ok": False, "error": f"could not read resource: {exc}"}

        max_chars = max(1, min(int(max_chars or 12_000), 100_000))
        return {
            "ok": True,
            "skill": stub.name,
            "path": rel,
            "content": text[:max_chars],
            "chars": len(text),
            "truncated": len(text) > max_chars,
        }

    # ------------------------------------------------------------------
    # Tool / MCP activation
    # ------------------------------------------------------------------

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

        allowed, reason = self._activation_allowed(stub.name)
        if not allowed:
            return {
                "ok": False,
                "name": stub.name,
                "error": reason or f"activation of {name} is not permitted",
                "denied": True,
            }

        if stub.source == "mcp":
            if stub.mcp_client is None or stub.mcp_session is None:
                if stub.server_id:
                    connected = self.activate_mcp_server(stub.server_id)
                    if not connected.get("ok"):
                        return {
                            "ok": False,
                            "error": connected.get("error")
                            or f"could not connect to MCP server {stub.server_id}",
                        }
                    # Re-fetch: activate_mcp_server refreshed/created the stub.
                    stub = self.catalog.tool_by_name(name) or stub
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

    def ensure_tools_activated(self, names: Sequence[str]) -> list[str]:
        """Best-effort activate each deferred tool; return names still missing.

        For MCP tools without a live session, ``activate_mcp_server`` is tried
        first (lazy connect on demand). Used by the tool loop when
        ``require_tools`` names a tool that discovery deferred, so the model
        doesn't have to manually search+activate before the deliverable gate
        can be satisfied.
        """
        missing: list[str] = []
        for raw in names or []:
            tool_name = (raw or "").strip()
            if not tool_name:
                continue
            stub = self.catalog.tool_by_name(tool_name)
            if stub is None:
                missing.append(tool_name)
                continue
            if stub.activated:
                continue
            if (
                stub.source == "mcp"
                and (stub.mcp_client is None or stub.mcp_session is None)
                and stub.server_id
            ):
                self.activate_mcp_server(stub.server_id)
            result = self.activate_tool(tool_name)
            if not result.get("ok"):
                missing.append(tool_name)
        return missing

    def _run_async(self, coro: Any) -> Any:
        """Run an awaitable synchronously, tolerating an already-running loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return asyncio.run(coro)

    async def _activate_mcp_server_async(self, server_id: str) -> dict[str, Any]:
        server = self.catalog.server_by_id(server_id)
        if server is None:
            return {
                "ok": False,
                "error": f"unknown MCP server: {server_id}",
                "available": [s.server_id for s in self.catalog.servers],
            }
        if server.connected:
            return {"ok": True, "server_id": server.server_id, "already_connected": True}
        if server.spec is None:
            return {
                "ok": False,
                "error": f"MCP server {server_id} has no connection spec to lazily connect",
            }

        from loomable.kernel.errors import MCPConnectionError
        from loomable.kernel.mcp_client import MCPClient

        client = MCPClient()
        try:
            session = await client.connect(server.spec)
            capabilities = await client.list_capabilities(session)
        except MCPConnectionError as exc:
            return {"ok": False, "error": f"failed to connect to {server_id}: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"failed to connect to {server_id}: {exc}"}

        ns_name = f"mcp:{server.server_id}"
        tool_names: list[str] = []
        for tool_info in capabilities.tools:
            tool_name = tool_info.get("name", "")
            if not tool_name:
                continue
            description = tool_info.get("description", "")
            parameters = tool_info.get("parameters") or {
                "type": "object",
                "properties": {},
            }
            existing = self.catalog.tool_by_name(tool_name)
            if existing is not None:
                existing.source = "mcp"
                existing.server_id = server.server_id
                existing.mcp_client = client
                existing.mcp_session = session
                existing.parameters = parameters if isinstance(parameters, dict) else {}
                existing.description = description or existing.description
                existing.namespace = existing.namespace or ns_name
            else:
                self.catalog.tools.append(
                    ToolStub(
                        name=tool_name,
                        description=description,
                        source="mcp",
                        server_id=server.server_id,
                        parameters=parameters if isinstance(parameters, dict) else {},
                        activated=False,
                        mcp_client=client,
                        mcp_session=session,
                        namespace=ns_name,
                    )
                )
            tool_names.append(tool_name)

        server.connected = True
        server.tool_count = len(tool_names)
        ns = self.catalog.namespace_by_name(ns_name)
        if ns is None:
            self.catalog.namespaces.append(
                NamespaceStub(
                    name=ns_name,
                    description=server.description or f"MCP server {server.server_id}",
                    tools=tool_names,
                )
            )
        else:
            for t in tool_names:
                if t not in ns.tools:
                    ns.tools.append(t)

        return {
            "ok": True,
            "server_id": server.server_id,
            "tools_catalogued": tool_names,
            "namespace": ns_name,
            "hint": "Call activate_tool(name) to make a specific tool callable.",
        }

    def activate_mcp_server(self, server_id: str) -> dict[str, Any]:
        """Connect a lazily-catalogued MCP server and catalog its tools.

        No-op (``already_connected``) if the server is already connected.
        Runs the async :class:`~loomable.kernel.mcp_client.MCPClient` connect
        flow synchronously so it can be called from sync tool handlers.
        """
        return self._run_async(self._activate_mcp_server_async(server_id))

    # ------------------------------------------------------------------
    # Mid-run catalog refresh
    # ------------------------------------------------------------------

    async def _refresh_server_async(
        self, server_id: str, client: Any, session: Any
    ) -> list[str]:
        capabilities = await client.list_capabilities(session)
        ns_name = f"mcp:{server_id}"
        ns = self.catalog.namespace_by_name(ns_name)
        added: list[str] = []
        for tool_info in capabilities.tools:
            tool_name = tool_info.get("name", "")
            if not tool_name or self.catalog.tool_by_name(tool_name) is not None:
                continue
            self.catalog.tools.append(
                ToolStub(
                    name=tool_name,
                    description=tool_info.get("description", ""),
                    source="mcp",
                    server_id=server_id,
                    parameters=tool_info.get("parameters")
                    or {"type": "object", "properties": {}},
                    activated=False,
                    mcp_client=client,
                    mcp_session=session,
                    namespace=ns_name,
                )
            )
            added.append(tool_name)
            if ns is not None and tool_name not in ns.tools:
                ns.tools.append(tool_name)
        return added

    def refresh_capabilities(self) -> dict[str, Any]:
        """Re-discover ``skill_roots`` and re-list tools from connected MCP sessions.

        Picks up skills added to disk after the agent was built, and any new
        tools a connected MCP server has started exposing. Safe to call
        repeatedly mid-run; already-catalogued names are left untouched.
        """
        added_skills: list[str] = []
        if self.skill_roots:
            seen = {s.name for s in self.catalog.skills}
            for root in self.skill_roots:
                try:
                    manifests = self.skill_loader.discover([Path(root)])
                except Exception:  # noqa: BLE001
                    continue
                for stub in catalog_from_skill_manifests(manifests):
                    if stub.name in seen:
                        continue
                    seen.add(stub.name)
                    self.catalog.skills.append(stub)
                    added_skills.append(stub.name)

        added_tools: list[str] = []
        for server in list(self.catalog.servers):
            if not server.connected:
                continue
            client = None
            session = None
            for t in self.catalog.tools:
                if t.server_id == server.server_id and t.mcp_session is not None:
                    client, session = t.mcp_client, t.mcp_session
                    break
            if client is None or session is None:
                continue
            try:
                added = self._run_async(
                    self._refresh_server_async(server.server_id, client, session)
                )
            except Exception:  # noqa: BLE001
                continue
            added_tools.extend(added)

        return {
            "ok": True,
            "skills_added": added_skills,
            "tools_added": added_tools,
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

    async def search_tools(
        query: str = "", limit: int = 8, source: str = "", namespace: str = ""
    ) -> str:
        """Search local / skill / catalogued tools without advertising full schemas."""
        hits = runtime.search_tools(query, limit=limit, source=source, namespace=namespace)
        return json.dumps({"query": query, "tools": hits}, ensure_ascii=False)

    async def search_mcp(query: str = "", limit: int = 8, server_id: str = "") -> str:
        """Search MCP tools from connected servers, plus unconnected server hints."""
        hits = runtime.search_mcp(query, limit=limit, server_id=server_id)
        return json.dumps(
            {"query": query, "server_id": server_id or None, "tools": hits},
            ensure_ascii=False,
        )

    async def search_namespaces(query: str = "", limit: int = 8) -> str:
        """Search tool/server namespaces (e.g. mcp:<server_id>, custom groups)."""
        hits = runtime.search_namespaces(query, limit=limit)
        return json.dumps({"query": query, "namespaces": hits}, ensure_ascii=False)

    async def activate_tool(name: str) -> str:
        """Activate a catalogued tool (especially deferred MCP tools) for calling."""
        return json.dumps(runtime.activate_tool(name), ensure_ascii=False)

    async def activate_mcp_server(server_id: str) -> str:
        """Connect an unconnected (lazy) MCP server and catalog its tools."""
        return json.dumps(runtime.activate_mcp_server(server_id), ensure_ascii=False)

    async def refresh_capabilities() -> str:
        """Re-scan skill roots and re-list tools from connected MCP servers."""
        return json.dumps(runtime.refresh_capabilities(), ensure_ascii=False)

    async def list_skill_resources(name: str) -> str:
        """List a skill's level-3 resources (SKILL.md, scripts/, references/, assets/)."""
        return json.dumps(runtime.list_skill_resources(name), ensure_ascii=False)

    async def read_skill_resource(name: str, path: str, max_chars: int = 12_000) -> str:
        """Read one skill resource file under scripts/, references/, or assets/."""
        return json.dumps(
            runtime.read_skill_resource(name, path, max_chars=max_chars),
            ensure_ascii=False,
        )

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
            search_namespaces,
            name="search_namespaces",
            description="Search tool namespaces/groups (e.g. mcp:<server_id>).",
        ),
        FunctionTool(
            activate_tool,
            name="activate_tool",
            description="Activate a deferred/catalogued tool (e.g. MCP) so it can be called.",
            idempotent=False,
        ),
        FunctionTool(
            activate_mcp_server,
            name="activate_mcp_server",
            description="Connect a lazy/unconnected MCP server and catalog its tools.",
            idempotent=False,
        ),
        FunctionTool(
            refresh_capabilities,
            name="refresh_capabilities",
            description="Re-scan skill roots and connected MCP servers for new capabilities.",
            idempotent=False,
        ),
        FunctionTool(
            list_skill_resources,
            name="list_skill_resources",
            description="List a skill's scripts/references/assets resources.",
        ),
        FunctionTool(
            read_skill_resource,
            name="read_skill_resource",
            description="Read one skill resource file (scripts/, references/, assets/, SKILL.md).",
        ),
    ]


DISCOVERY_SYSTEM_NOTE = """\
Capability discovery: use search_skills / load_skill for workflows, search_tools
for deferred local tools, search_mcp + activate_tool for MCP tools, and
search_namespaces to browse tool groups (e.g. mcp:<server_id>). Some MCP
servers may be listed but not yet connected — call
activate_mcp_server(server_id) to connect and catalog their tools, then
activate_tool(name) to make one callable. Use list_skill_resources(name) /
read_skill_resource(name, path) for a skill's scripts/references/assets beyond
its main body. Call refresh_capabilities() if the catalog may be stale (new
skills on disk, or a server's tools changed). Prefer searching before assuming
a tool exists. After load/activate, call the tool (schemas for activated tools
are returned by activate_tool and also appear on the next turn).
"""
