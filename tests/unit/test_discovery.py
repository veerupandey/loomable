"""Unit tests for progressive capability discovery (skills / tools / MCP)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loomable.agent import Agent, FunctionTool
from loomable.agent.discovery import (
    CapabilityCatalog,
    DiscoveryRuntime,
    SkillStub,
    ToolStub,
    make_discovery_tools,
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
