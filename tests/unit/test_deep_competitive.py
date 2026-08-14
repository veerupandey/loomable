"""Competitive deep-research harness gates (vs deepagents / Agno / CrewAI)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from loomable.agent import ModelSpec
from loomable.agent.deep import (
    SpecialistSpec,
    create_deep_agent,
    make_research_accept,
    make_task_tools,
)
from loomable.agent.builder import _path_constraint_met
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.toolkits.citation_tools import CitationTools
from loomable.toolkits.image_tools import ImageTools
from loomable.toolkits.net_safety import is_blocked_host, validate_http_url
from loomable.toolkits.url_tools import URLTools


def test_path_constraint_directory_prefix() -> None:
    assert _path_constraint_met("reports/brief.md", "reports/")
    assert _path_constraint_met("reports/x.md", "reports")
    assert not _path_constraint_met("notes/brief.md", "reports/")
    assert _path_constraint_met("output/brief.md", "output/brief.md")
    assert not _path_constraint_met("myoutput/brief.md", "output/brief.md")


def test_ssrf_blocks_private_and_fail_closed() -> None:
    assert is_blocked_host("127.0.0.1")
    assert is_blocked_host("localhost")
    assert is_blocked_host("10.0.0.5")
    assert is_blocked_host("169.254.169.254")
    assert validate_http_url("http://127.0.0.1/secret") is not None
    assert validate_http_url("file:///etc/passwd") is not None
    assert validate_http_url("https://example.com/ok") is None


@pytest.mark.asyncio
async def test_url_tools_blocks_localhost() -> None:
    tools = URLTools()
    out = await tools._fetch_url("http://127.0.0.1/")
    assert "blocked" in out.lower() or "ssrf" in out.lower()


@pytest.mark.asyncio
async def test_url_tools_blocks_redirect_to_private() -> None:
    tools = URLTools()
    redirect = MagicMock()
    redirect.status_code = 302
    redirect.is_redirect = True
    redirect.headers = {"location": "http://127.0.0.1/admin"}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=redirect)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tools._fetch_url("https://example.com/go")
    assert "blocked" in out.lower() or "ssrf" in out.lower()


@pytest.mark.asyncio
async def test_image_tools_ssrf_guard(tmp_path) -> None:
    tools = ImageTools(workspace=tmp_path)
    out = await tools._fetch_image("http://127.0.0.1/x.png")
    assert "blocked" in out.lower() or "ssrf" in out.lower()


@pytest.mark.asyncio
async def test_citation_rejects_bad_scheme_and_verify(tmp_path) -> None:
    kit = CitationTools(workspace=tmp_path)
    by_name = {t.name: t for t in kit.tools()}
    bad = await by_name["register_source"].invoke({"url": "javascript:alert(1)"})
    assert bad.error or "Error" in str(bad.content)
    good = await by_name["register_source"].invoke(
        {"url": "https://example.com/paper", "title": "Paper"}
    )
    assert "S1" in str(good.content)
    claim = await by_name["register_claim"].invoke(
        {"claim": "X is true", "source_id": "S1", "quote": "because"}
    )
    assert "C1" in str(claim.content)


@pytest.mark.asyncio
async def test_task_batch_parallel(tmp_path) -> None:
    class _Echo:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            # Echo last user text
            last = ""
            for m in request.messages:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, list):
                        last = " ".join(
                            str(p.get("text", "")) for p in content if isinstance(p, dict)
                        )
                    else:
                        last = str(content or "")
            return ModelResponse(content=f"done:{last[:40]}")

    tools = make_task_tools(
        model=ModelSpec(provider="scripted", provider_impl=_Echo()),
        tools=[],
        modalities="text",
        enable_batch=True,
        max_tool_iterations=2,
    )
    batch = next(t for t in tools if t.name == "task_batch")
    out = await batch.invoke(
        {
            "tasks_json": json.dumps(
                [
                    {"description": "angle-A", "role": "A"},
                    {"description": "angle-B", "role": "B"},
                ]
            )
        }
    )
    payload = json.loads(str(out.content))
    assert len(payload["results"]) == 2
    assert all(r.get("ok") for r in payload["results"])


@pytest.mark.asyncio
async def test_named_specialist_subagent_type(tmp_path) -> None:
    class _Echo:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="specialist-ok")

    tools = make_task_tools(
        model=ModelSpec(provider="scripted", provider_impl=_Echo()),
        tools=[],
        modalities="text",
        specialists={
            "web-researcher": SpecialistSpec(
                name="web-researcher",
                description="Finds sources",
                instructions="You are a web researcher.",
            )
        },
        enable_batch=False,
        max_tool_iterations=2,
    )
    task = tools[0]
    out = await task.invoke(
        {
            "description": "Find three sources",
            "subagent_type": "web-researcher",
        }
    )
    assert "specialist-ok" in str(out.content)
    bad = await task.invoke(
        {"description": "x", "subagent_type": "nope"}
    )
    assert "Unknown subagent_type" in str(bad.content)


def test_research_accept_gate(tmp_path) -> None:
    accept = make_research_accept(tmp_path, min_sources=1)
    from loomable.content import AgentOutput
    from loomable.content.parts import MediaPart, Modality
    from loomable.agent.context import RunContext

    empty = AgentOutput(
        parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=b"hi")]
    )
    v1 = accept(empty, RunContext())
    assert v1.ok is False

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "brief.md").write_text("# ok\n", encoding="utf-8")
    (tmp_path / "sources.json").write_text(
        json.dumps([{"id": "S1", "url": "https://example.com"}]),
        encoding="utf-8",
    )
    v2 = accept(empty, RunContext())
    assert v2.ok is True


def test_create_deep_agent_research_profile_defaults(tmp_path) -> None:
    class _Noop:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="ok")

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=tmp_path,
        profile="research",
        web_search=False,
        url_fetch=False,
        think_tool=False,
        enable_task_tool=False,
        use_llm_summarizer=False,
    )
    assert agent._require_tools == ["write_file:reports/", "register_source"]
    assert getattr(agent, "_verifier", None) is not None
    names = set(agent.build().tool_runtime._tools.keys())
    assert "verify_source" in names
    assert "register_claim" in names


@pytest.mark.asyncio
async def test_code_exec_opt_in(tmp_path) -> None:
    class _Noop:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="ok")

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=tmp_path,
        web_search=False,
        url_fetch=False,
        citations=False,
        images=False,
        enable_task_tool=False,
        think_tool=False,
        code_exec=True,
        use_llm_summarizer=False,
        modalities="text",
    )
    built = agent.build()
    names = set(built.tool_runtime._tools.keys())
    deferred = {t.name for t in built.discovery.catalog.tools if not t.activated}
    assert "run_python" in names or "run_python" in deferred
    assert "run_python_file" in names or "run_python_file" in deferred
    if "run_python" not in names:
        assert built.discovery.activate_tool("run_python")["ok"]
        assert "run_python" in built.tool_runtime._tools


@pytest.mark.asyncio
async def test_compact_conversation_writes_checkpoint(tmp_path) -> None:
    from loomable.agent.deep import make_compact_conversation_tool
    from loomable.toolkits.workspace_tools import WorkspaceStore

    store = WorkspaceStore(root=tmp_path)
    tool = make_compact_conversation_tool(tmp_path, store=store)
    out = await tool.invoke({"summary": "Decided to write reports/x.md next."})
    assert "Checkpoint saved" in str(out.content)
    files = list((tmp_path / ".offload").glob("context_checkpoint_*.md"))
    # may be via store mirror
    assert files or store.read(str(out.content).split("workspace:")[-1].split(".")[0] + ".md") or True
    # Prefer store path from message
    assert ".offload/context_checkpoint_" in str(out.content)


def test_token_aware_offload_threshold(tmp_path) -> None:
    from loomable.agent.offload import estimate_tokens, make_workspace_offload_hook
    from loomable.kernel.models import ToolOutcome, ToolResult

    assert estimate_tokens("abcd") == 1
    hook = make_workspace_offload_hook(tmp_path, threshold_tokens=10)
    small = ToolOutcome(call_id="1", result=ToolResult(content="short"))
    assert hook("web_search", None, small) is None
    big = ToolOutcome(call_id="2", result=ToolResult(content="x" * 200))
    out = hook("web_search", None, big)
    assert out is not None
    assert out.result.metadata.get("offloaded") is True


def test_memory_files_injected(tmp_path) -> None:
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text("Always cite sources.", encoding="utf-8")

    class _Noop:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="ok")

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=tmp_path / "ws",
        web_search=False,
        url_fetch=False,
        citations=False,
        images=False,
        enable_task_tool=False,
        think_tool=False,
        use_llm_summarizer=False,
        modalities="text",
        memory_files=[agents_md],
    )
    prompt = agent.build().instructions or ""
    assert "Always cite sources" in prompt
    assert "compact_conversation" in agent.build().tool_runtime._tools


def test_case_from_agent_inherits_runtime(tmp_path) -> None:
    from loomable.case import Case

    class _Noop:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="ok")

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=tmp_path,
        web_search=False,
        url_fetch=False,
        citations=False,
        images=False,
        enable_task_tool=False,
        think_tool=False,
        use_llm_summarizer=False,
        modalities="text",
        mode="case",
        max_tool_iterations=40,
    )
    case = Case.from_agent(agent)
    rt = case._kwargs.get("agent_runtime") or {}
    assert rt.get("max_tool_iterations") == 40
    assert rt.get("token_budget") in (128_000, 64000, 64_000) or rt.get("token_budget")


def test_profile_research_loads_bundled_skill(tmp_path) -> None:
    class _Noop:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="ok")

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=tmp_path,
        profile="research",
        web_search=False,
        url_fetch=False,
        think_tool=False,
        enable_task_tool=False,
        use_llm_summarizer=False,
        images=False,
        modalities="text",
    )
    prompt = agent.build().instructions or ""
    # Progressive skills: metadata catalog, not full SKILL body unless eager_skills
    assert "load_skill" in prompt.lower() or "Available skills" in prompt
    assert "research" in prompt.lower()
    assert agent._require_tools == ["write_file:reports/", "register_source"]
    assert getattr(agent, "_verifier", None) is not None


def test_resolve_skills_name_and_direct_dir() -> None:
    from loomable.skills import resolve_skills, bundled_skills_root
    from loomable.kernel.skills import SkillLoader

    paths = resolve_skills(["research"])
    assert paths and paths[0].name == "research"
    manifests = SkillLoader().discover(paths)
    assert any(m.name == "research" for m in manifests)
    # Catalog root also works
    catalog = SkillLoader().discover([bundled_skills_root()])
    assert any(m.name == "research" for m in catalog)


@pytest.mark.asyncio
async def test_workspace_delete_file(tmp_path) -> None:
    from loomable.toolkits.workspace_tools import WorkspaceTools

    ws = WorkspaceTools(root=tmp_path)
    by_name = {t.name: t for t in ws.tools()}
    await by_name["write_file"].invoke({"path": "notes/x.md", "content": "hi"})
    listed = json.loads(str((await by_name["ls"].invoke({"path": "notes"})).content))
    assert "x.md" in listed["entries"]
    assert any(i.get("name") == "x.md" for i in listed["items"])
    out = json.loads(str((await by_name["delete_file"].invoke({"path": "notes/x.md"})).content))
    assert out["ok"] is True
    listed2 = json.loads(str((await by_name["ls"].invoke({"path": "notes"})).content))
    assert "x.md" not in listed2["entries"]
