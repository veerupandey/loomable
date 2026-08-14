"""Unit tests for progressive capability discovery (skills / tools / MCP)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loomable.agent import Agent, FunctionTool
from loomable.agent.discovery import (
    CapabilityCatalog,
    DiscoveryRuntime,
    NamespaceStub,
    ServerStub,
    SkillStub,
    ToolStub,
    make_discovery_tools,
    rank_bm25,
    rank_match,
)
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.kernel.tool_runtime import ToolRuntime


class _FakeProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


def _content(result) -> str:
    if hasattr(result, "content"):
        return str(result.content or "")
    return str(result)


def test_rank_match_prefers_exact_name() -> None:
    assert rank_match("research", "research", "x") > rank_match(
        "research", "web_research", "research stuff"
    )
    assert rank_match("pdf", "read_pdf", "extract pdf text") > 0
    assert rank_match("zzzz", "alpha", "beta") == 0.0


def test_search_and_load_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill for tests\n---\n\n# Demo\n\nDo the demo.\n",
        encoding="utf-8",
    )
    catalog = CapabilityCatalog(
        skills=[
            SkillStub(
                name="demo",
                description="Demo skill for tests",
                path=skill_dir,
            )
        ]
    )
    injected: list[tuple[str, str]] = []
    runtime = DiscoveryRuntime(
        catalog,
        on_skill_body=lambda n, b: injected.append((n, b)),
    )
    hits = runtime.search_skills("demo")
    assert hits and hits[0]["name"] == "demo"
    assert hits[0]["loaded"] is False

    result = runtime.load_skill("demo")
    assert result["ok"] is True
    assert result["body_chars"] > 0
    assert "Do the demo" in result["body"]
    assert catalog.skill_by_name("demo").loaded is True
    assert injected and "Do the demo" in injected[0][1]
    pending = runtime.drain_prompt_injections()
    assert pending and pending[0][0] == "demo"

    again = runtime.load_skill("demo")
    assert again.get("already_loaded") is True


def test_activate_deferred_local_tool() -> None:
    async def deferred_echo(text: str = "") -> str:
        return f"echo:{text}"

    tool = FunctionTool(deferred_echo, name="deferred_echo", description="Echo text")
    catalog = CapabilityCatalog(
        tools=[
            ToolStub(
                name="deferred_echo",
                description="Echo text",
                source="local",
                activated=False,
            )
        ]
    )
    runtime = DiscoveryRuntime(catalog, tool_runtime=ToolRuntime({}))
    runtime.register_pending_local("deferred_echo", tool)

    assert "deferred_echo" not in runtime.tool_runtime._tools
    out = runtime.activate_tool("deferred_echo")
    assert out["ok"] is True
    assert "deferred_echo" in runtime.tool_runtime._tools
    assert catalog.tool_by_name("deferred_echo").activated is True


@pytest.mark.asyncio
async def test_agent_discovery_registers_meta_tools(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    skill = skill_root / "extra"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: extra\ndescription: Extra catalog skill\n---\nExtra body.\n",
        encoding="utf-8",
    )

    agent = Agent(
        model=_FakeProvider(),
        discovery=True,
        skill_catalog=[skill_root],
        tools=[],
    )
    built = agent.build()
    names = set(built.tool_runtime._tools)
    assert {
        "search_skills",
        "load_skill",
        "search_tools",
        "search_mcp",
        "activate_tool",
    } <= names
    assert built.discovery is not None
    assert "Capability discovery" in (built.instructions or "")
    hits = built.discovery.search_skills("extra")
    assert any(h["name"] == "extra" for h in hits)


@pytest.mark.asyncio
async def test_tool_schemas_include_activated_tool() -> None:
    async def deferred_echo(text: str = "") -> str:
        return f"echo:{text}"

    deferred = FunctionTool(deferred_echo, name="deferred_echo", description="Echo")
    agent = Agent(model=_FakeProvider(), discovery=True, tools=[])
    built = agent.build()
    built.discovery.register_pending_local("deferred_echo", deferred)
    built.discovery.catalog.tools.append(
        ToolStub(
            name="deferred_echo",
            description="Echo",
            source="local",
            activated=False,
        )
    )
    built.tool_runtime._tools.pop("deferred_echo", None)

    result = built.discovery.activate_tool("deferred_echo")
    assert result["ok"]
    assert "deferred_echo" in built.tool_runtime._tools

    from loomable.agent.tools import FunctionTool as FT
    from loomable.agent.tools import MCPTool

    schemas = []
    for tool_obj in built.tool_runtime._tools.values():
        if isinstance(tool_obj, (FT, MCPTool)):
            schemas.append(tool_obj.schema())
    names = {s["function"]["name"] for s in schemas}
    assert "deferred_echo" in names
    assert "activate_tool" in names


@pytest.mark.asyncio
async def test_make_discovery_tools_json() -> None:
    catalog = CapabilityCatalog(
        skills=[
            SkillStub(name="research", description="Research any topic", path=Path("."))
        ]
    )
    runtime = DiscoveryRuntime(catalog)
    tools = {t.name: t for t in make_discovery_tools(runtime)}
    raw = await tools["search_skills"].invoke({"query": "research"})
    payload = json.loads(_content(raw))
    assert payload["skills"]
    assert payload["skills"][0]["name"] == "research"


@pytest.mark.asyncio
async def test_discovery_defers_non_core_local_tools() -> None:
    async def core_ping() -> str:
        return "ping"

    async def heavy_pdf() -> str:
        return "pdf"

    agent = Agent(
        model=_FakeProvider(),
        discovery=True,
        defer_local_tools=True,
        discovery_core_tools=["core_ping"],
        tools=[
            FunctionTool(core_ping, name="core_ping", description="Core"),
            FunctionTool(heavy_pdf, name="heavy_pdf", description="Heavy PDF"),
        ],
    )
    built = agent.build()
    names = set(built.tool_runtime._tools)
    assert "core_ping" in names
    assert "heavy_pdf" not in names
    assert "search_tools" in names
    stub = built.discovery.catalog.tool_by_name("heavy_pdf")
    assert stub is not None and stub.activated is False
    out = built.discovery.activate_tool("heavy_pdf")
    assert out["ok"] and "schema" in out
    assert out["schema"]["function"]["name"] == "heavy_pdf"
    assert "heavy_pdf" in built.tool_runtime._tools


@pytest.mark.asyncio
async def test_discovery_progressive_skills_metadata_only(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo workflow\n---\n\n# Demo\n\nSECRET_BODY_TOKEN\n",
        encoding="utf-8",
    )
    agent = Agent(
        model=_FakeProvider(),
        discovery=True,
        skills=[tmp_path / "skills"],
        tools=[],
    )
    built = agent.build()
    prompt = built.instructions or ""
    assert "SECRET_BODY_TOKEN" not in prompt
    assert "demo" in prompt.lower()
    assert "Available skills" in prompt or "load_skill" in prompt.lower()
    loaded = built.discovery.load_skill("demo")
    assert loaded["ok"]
    assert "SECRET_BODY_TOKEN" in loaded["body"]
    assert "SECRET_BODY_TOKEN" in (built.instructions or "")


@pytest.mark.asyncio
async def test_eager_skills_true_injects_body(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo\n---\n\nEAGER_BODY\n",
        encoding="utf-8",
    )
    agent = Agent(
        model=_FakeProvider(),
        discovery=True,
        eager_skills=True,
        skills=[tmp_path / "skills"],
        tools=[],
    )
    built = agent.build()
    assert "EAGER_BODY" in (built.instructions or "")


# ---------------------------------------------------------------------------
# P1/P2: BM25 ranking, activation policy, skill resources, namespaces, lazy MCP
# ---------------------------------------------------------------------------


def test_bm25_prefers_better_match() -> None:
    assert rank_bm25("research", "research", "x") > rank_bm25(
        "research", "web_research", "research stuff"
    )
    assert rank_bm25(
        "read pdf", "read_pdf", "Extract text from a PDF file"
    ) > rank_bm25("read pdf", "unrelated_tool", "does something else entirely")
    assert rank_bm25("zzzz", "alpha", "beta") == 0.0


def test_search_tools_uses_bm25_ranking() -> None:
    catalog = CapabilityCatalog(
        tools=[
            ToolStub(
                name="read_pdf",
                description="Extract text from a PDF file",
                source="local",
                activated=True,
            ),
            ToolStub(
                name="unrelated_tool",
                description="Does something else entirely",
                source="local",
                activated=True,
            ),
        ]
    )
    runtime = DiscoveryRuntime(catalog)
    hits = runtime.search_tools("pdf")
    assert hits[0]["name"] == "read_pdf"
    assert hits[0]["score"] > 0


def test_activation_denylist_blocks() -> None:
    async def run_shell(cmd: str = "") -> str:
        return f"ran:{cmd}"

    tool = FunctionTool(run_shell, name="run_shell", description="Run a shell command")
    catalog = CapabilityCatalog(
        tools=[
            ToolStub(name="run_shell", description="Run a shell command", source="local")
        ]
    )
    runtime = DiscoveryRuntime(
        catalog, tool_runtime=ToolRuntime({}), activation_denylist=["run_*"]
    )
    runtime.register_pending_local("run_shell", tool)

    out = runtime.activate_tool("run_shell")
    assert out["ok"] is False
    assert out.get("denied") is True
    assert "run_shell" not in runtime.tool_runtime._tools
    assert catalog.tool_by_name("run_shell").activated is False


def test_activation_allowlist_restricts() -> None:
    async def noop() -> str:
        return "ok"

    catalog = CapabilityCatalog(
        tools=[
            ToolStub(name="safe_tool", description="Safe", source="local"),
            ToolStub(name="other_tool", description="Other", source="local"),
        ]
    )
    runtime = DiscoveryRuntime(
        catalog, tool_runtime=ToolRuntime({}), activation_allowlist=["safe_*"]
    )
    runtime.register_pending_local("safe_tool", FunctionTool(noop, name="safe_tool"))
    runtime.register_pending_local("other_tool", FunctionTool(noop, name="other_tool"))

    assert runtime.activate_tool("safe_tool")["ok"] is True
    blocked = runtime.activate_tool("other_tool")
    assert blocked["ok"] is False
    assert blocked.get("denied") is True


def test_on_activate_check_rejects_with_reason() -> None:
    async def noop() -> str:
        return "ok"

    catalog = CapabilityCatalog(tools=[ToolStub(name="risky", description="Risky", source="local")])
    runtime = DiscoveryRuntime(
        catalog,
        tool_runtime=ToolRuntime({}),
        on_activate_check=lambda name: "needs human approval",
    )
    runtime.register_pending_local("risky", FunctionTool(noop, name="risky"))

    out = runtime.activate_tool("risky")
    assert out["ok"] is False
    assert out["error"] == "needs human approval"


def test_list_and_read_skill_resources(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo\n---\nBody\n", encoding="utf-8"
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "checklist.md").write_text("# Checklist\n- item\n", encoding="utf-8")
    scripts = skill_dir / "scripts"
    scripts.mkdir()
    (scripts / "run.py").write_text("print('hi')\n", encoding="utf-8")

    catalog = CapabilityCatalog(
        skills=[SkillStub(name="demo", description="Demo", path=skill_dir)]
    )
    runtime = DiscoveryRuntime(catalog)

    listing = runtime.list_skill_resources("demo")
    assert listing["ok"] is True
    paths = {r["path"] for r in listing["resources"]}
    assert "SKILL.md" in paths
    assert "references/checklist.md" in paths
    assert "scripts/run.py" in paths

    read = runtime.read_skill_resource("demo", "references/checklist.md")
    assert read["ok"] is True
    assert "Checklist" in read["content"]

    # Path traversal outside the skill directory is rejected.
    traversal = runtime.read_skill_resource("demo", "../../etc/passwd")
    assert traversal["ok"] is False

    # Files outside scripts/references/assets/SKILL.md are rejected even
    # when they live inside the skill directory.
    (skill_dir / "secret.txt").write_text("nope", encoding="utf-8")
    disallowed = runtime.read_skill_resource("demo", "secret.txt")
    assert disallowed["ok"] is False

    unknown = runtime.list_skill_resources("does-not-exist")
    assert unknown["ok"] is False


def test_ensure_tools_activated_registers_deferred_tools() -> None:
    async def deferred_echo(text: str = "") -> str:
        return f"echo:{text}"

    tool = FunctionTool(deferred_echo, name="deferred_echo", description="Echo text")
    catalog = CapabilityCatalog(
        tools=[ToolStub(name="deferred_echo", description="Echo text", source="local")]
    )
    runtime = DiscoveryRuntime(catalog, tool_runtime=ToolRuntime({}))
    runtime.register_pending_local("deferred_echo", tool)

    missing = runtime.ensure_tools_activated(["deferred_echo", "totally_unknown"])
    assert missing == ["totally_unknown"]
    assert "deferred_echo" in runtime.tool_runtime._tools
    assert catalog.tool_by_name("deferred_echo").activated is True

    # Already-activated tools are a no-op, not re-reported as missing.
    again = runtime.ensure_tools_activated(["deferred_echo"])
    assert again == []


def test_search_namespaces() -> None:
    catalog = CapabilityCatalog(
        namespaces=[
            NamespaceStub(
                name="mcp:github", description="GitHub MCP tools", tools=["create_issue"]
            ),
            NamespaceStub(
                name="images",
                description="Image analysis tools",
                tools=["fetch_image", "analyze_image"],
            ),
        ]
    )
    runtime = DiscoveryRuntime(catalog)
    hits = runtime.search_namespaces("github")
    assert hits and hits[0]["name"] == "mcp:github"

    all_hits = runtime.search_namespaces("")
    assert len(all_hits) == 2


def test_search_mcp_hints_unconnected_server() -> None:
    catalog = CapabilityCatalog(
        servers=[
            ServerStub(
                server_id="gh",
                description="GitHub MCP server",
                connected=False,
                spec={"server_id": "gh"},
            )
        ]
    )
    runtime = DiscoveryRuntime(catalog, lazy_mcp=True)
    hits = runtime.search_mcp("github")
    assert hits and hits[0]["server_id"] == "gh"
    assert hits[0]["connected"] is False
    assert "activate_mcp_server" in hits[0]["hint"]


class _FakeMCPSession:
    def __init__(self, server_id: str) -> None:
        self.server_id = server_id


class _FakeMCPCapabilities:
    def __init__(self, tools: list[dict]) -> None:
        self.tools = tools


class _FakeMCPClientForActivation:
    """Stand-in for :class:`~loomable.kernel.mcp_client.MCPClient` in tests."""

    async def connect(self, spec: dict) -> _FakeMCPSession:
        return _FakeMCPSession(spec.get("server_id", "unknown"))

    async def list_capabilities(self, session: _FakeMCPSession) -> _FakeMCPCapabilities:
        return _FakeMCPCapabilities(
            tools=[
                {
                    "name": "remote_search",
                    "description": "Search the remote server",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        )


def test_activate_mcp_server_lazy_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lazy MCP server stub connects on demand via activate_mcp_server."""
    monkeypatch.setattr(
        "loomable.kernel.mcp_client.MCPClient", _FakeMCPClientForActivation
    )

    catalog = CapabilityCatalog(
        servers=[
            ServerStub(
                server_id="gh",
                description="GitHub MCP server",
                connected=False,
                spec={"server_id": "gh"},
            )
        ]
    )
    runtime = DiscoveryRuntime(catalog, tool_runtime=ToolRuntime({}), lazy_mcp=True)

    # Not yet connected: no tools catalogued for it.
    assert catalog.tool_by_name("remote_search") is None

    result = runtime.activate_mcp_server("gh")
    assert result["ok"] is True
    assert result["namespace"] == "mcp:gh"
    assert "remote_search" in result["tools_catalogued"]

    server = catalog.server_by_id("gh")
    assert server is not None and server.connected is True

    stub = catalog.tool_by_name("remote_search")
    assert stub is not None
    assert stub.activated is False
    assert stub.source == "mcp"
    assert stub.mcp_client is not None and stub.mcp_session is not None

    ns = catalog.namespace_by_name("mcp:gh")
    assert ns is not None and "remote_search" in ns.tools

    # Idempotent: activating again is a no-op success.
    again = runtime.activate_mcp_server("gh")
    assert again["ok"] is True
    assert again.get("already_connected") is True

    # The catalogued (deferred) tool can now be activated for calling.
    activated = runtime.activate_tool("remote_search")
    assert activated["ok"] is True
    assert activated["schema"]["function"]["name"] == "remote_search"
    assert "remote_search" in runtime.tool_runtime._tools


def test_activate_tool_auto_connects_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """activate_tool on an MCP stub with no session triggers activate_mcp_server."""
    monkeypatch.setattr(
        "loomable.kernel.mcp_client.MCPClient", _FakeMCPClientForActivation
    )

    catalog = CapabilityCatalog(
        servers=[
            ServerStub(
                server_id="gh", description="GitHub MCP", connected=False, spec={"server_id": "gh"}
            )
        ],
        tools=[
            ToolStub(
                name="remote_search",
                description="Search the remote server",
                source="mcp",
                server_id="gh",
                activated=False,
            )
        ],
    )
    runtime = DiscoveryRuntime(catalog, tool_runtime=ToolRuntime({}), lazy_mcp=True)

    out = runtime.activate_tool("remote_search")
    assert out["ok"] is True
    assert out["source"] == "mcp"
    assert "remote_search" in runtime.tool_runtime._tools
    assert catalog.server_by_id("gh").connected is True


def test_ensure_tools_activated_connects_lazy_mcp_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "loomable.kernel.mcp_client.MCPClient", _FakeMCPClientForActivation
    )
    catalog = CapabilityCatalog(
        servers=[
            ServerStub(
                server_id="gh", description="GitHub MCP", connected=False, spec={"server_id": "gh"}
            )
        ],
        tools=[
            ToolStub(
                name="remote_search",
                description="Search the remote server",
                source="mcp",
                server_id="gh",
                activated=False,
            )
        ],
    )
    runtime = DiscoveryRuntime(catalog, tool_runtime=ToolRuntime({}), lazy_mcp=True)

    missing = runtime.ensure_tools_activated(["remote_search"])
    assert missing == []
    assert "remote_search" in runtime.tool_runtime._tools
