"""Unit tests for deep-agent harness primitives."""

from __future__ import annotations

import json

import pytest

from loomable.agent import ModelSpec, create_deep_agent
from loomable.agent.deep import make_task_tool
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.toolkits import TodoTools, WorkspaceTools
from loomable.toolkits.todo_tools import TodoStore
from loomable.toolkits.workspace_tools import WorkspaceStore


def _content(result) -> str:  # noqa: ANN001
    if getattr(result, "error", None):
        raise AssertionError(result.error)
    return str(result.content)


@pytest.mark.asyncio
async def test_todo_tools_write_read_update(tmp_path) -> None:
    tools = TodoTools(workspace=tmp_path)
    write = next(t for t in tools.tools() if t.name == "write_todos")
    read = next(t for t in tools.tools() if t.name == "read_todos")
    update = next(t for t in tools.tools() if t.name == "update_todo")

    out = _content(
        await write.invoke(
            {
                "todos": json.dumps(
                    [
                        {"content": "Research topic", "status": "in_progress"},
                        {"content": "Write brief", "status": "pending"},
                    ]
                )
            }
        )
    )
    assert "Research topic" in out
    listed = json.loads(_content(await read.invoke({})))
    assert len(listed["todos"]) == 2
    await update.invoke({"index": 0, "status": "completed"})
    listed2 = json.loads(_content(await read.invoke({})))
    assert listed2["todos"][0]["status"] == "completed"
    assert (tmp_path / "todos.json").is_file()


@pytest.mark.asyncio
async def test_workspace_tools_roundtrip_edit_grep(tmp_path) -> None:
    ws = WorkspaceTools(root=tmp_path)
    by_name = {t.name: t for t in ws.tools()}
    await by_name["write_file"].invoke({"path": "notes/a.md", "content": "hello world"})
    assert "hello world" == _content(await by_name["read_file"].invoke({"path": "notes/a.md"}))
    await by_name["edit_file"].invoke(
        {"path": "notes/a.md", "old_string": "world", "new_string": "loomable"}
    )
    assert "hello loomable" == _content(await by_name["read_file"].invoke({"path": "notes/a.md"}))
    hits = json.loads(_content(await by_name["grep"].invoke({"query": "loomable"})))
    assert hits["hits"]
    files = json.loads(_content(await by_name["glob"].invoke({"pattern": "**/*.md"})))
    assert "notes/a.md" in files["matches"]
    listing = json.loads(_content(await by_name["ls"].invoke({"path": "notes"})))
    assert "a.md" in listing["entries"]


@pytest.mark.asyncio
async def test_workspace_read_file_offset_limit(tmp_path) -> None:
    ws = WorkspaceTools(root=tmp_path)
    by_name = {t.name: t for t in ws.tools()}
    body = "\n".join(f"line-{i}" for i in range(1, 11))
    await by_name["write_file"].invoke({"path": "notes/big.md", "content": body})
    sliced = _content(
        await by_name["read_file"].invoke({"path": "notes/big.md", "offset": 2, "limit": 3})
    )
    assert "line-3" in sliced
    assert "line-5" in sliced
    assert "line-1" not in sliced.split("\n", 1)[-1] or "lines 3-5" in sliced
    assert "line-10" not in sliced
    """Offload/ImageTools write disk files; WorkspaceTools must still read them."""
    ws = WorkspaceTools(root=tmp_path)
    by_name = {t.name: t for t in ws.tools()}
    off = tmp_path / ".offload" / "fetch_url_abc.txt"
    off.parent.mkdir(parents=True)
    off.write_text("EXTERNAL_BODY_MARKER", encoding="utf-8")
    assert "EXTERNAL_BODY_MARKER" == _content(
        await by_name["read_file"].invoke({"path": ".offload/fetch_url_abc.txt"})
    )
    listing = json.loads(_content(await by_name["ls"].invoke({"path": ".offload"})))
    assert "fetch_url_abc.txt" in listing["entries"]
    hits = json.loads(_content(await by_name["grep"].invoke({"query": "EXTERNAL_BODY"})))
    assert hits["hits"]


@pytest.mark.asyncio
async def test_file_tools_edit_glob_grep(tmp_path) -> None:
    from loomable.toolkits import FileTools

    ft = FileTools(base_dir=str(tmp_path))
    by_name = {t.name: t for t in ft.tools()}
    await by_name["write_file"].invoke({"path": "x.txt", "content": "alpha beta"})
    await by_name["edit_file"].invoke(
        {"path": "x.txt", "old_string": "beta", "new_string": "gamma"}
    )
    assert "alpha gamma" in _content(await by_name["read_file"].invoke({"path": "x.txt"}))
    assert "x.txt" in _content(await by_name["glob_files"].invoke({"pattern": "*.txt"}))
    assert "gamma" in _content(await by_name["grep_files"].invoke({"query": "gamma"}))


@pytest.mark.asyncio
async def test_create_deep_agent_registers_core_tools(tmp_path) -> None:
    class _Noop:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="done")

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=tmp_path,
        web_search=False,
        url_fetch=False,
        citations=True,
        images=False,
        enable_task_tool=True,
        think_tool=True,
        modalities="text",
        use_llm_summarizer=False,
    )
    built = agent.build()
    names = set(built.tool_runtime._tools.keys())
    for required in (
        "write_todos",
        "read_todos",
        "update_todo",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete_file",
        "glob",
        "grep",
        "task",
        "task_batch",
        "think",
        "compact_conversation",
        "register_source",
        "verify_source",
        "register_claim",
        "list_sources",
        "format_bibliography",
    ):
        assert required in names, f"missing {required} in {sorted(names)}"
    assert built.max_tool_iterations == 40
    assert getattr(agent, "_token_budget", None) == 128_000
    assert getattr(agent, "_max_run_tokens", None) == 0
    assert getattr(agent, "_tool_concurrency", None) == 4
    assert getattr(agent, "_tool_timeout", None) == 60.0
    assert getattr(agent, "_resilience", None) is not None

    # Multimodal research defaults register image tools
    vision = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=tmp_path / "v",
        web_search=False,
        url_fetch=False,
        citations=False,
        images=True,
        enable_task_tool=False,
        think_tool=False,
        modalities="text+image",
        use_llm_summarizer=False,
    )
    vnames = set(vision.build().tool_runtime._tools.keys())
    for required in ("fetch_image", "analyze_image", "list_images", "discover_images"):
        assert required in vnames, f"missing {required}"


@pytest.mark.asyncio
async def test_deep_agent_scripted_tool_loop(tmp_path) -> None:
    """Scripted provider walks write_todos → write_file → final answer."""

    class _DeepScript:
        def __init__(self) -> None:
            self.n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            if self.n == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            tool_name="write_todos",
                            args={
                                "todos": json.dumps(
                                    [
                                        {
                                            "content": "Draft brief",
                                            "status": "in_progress",
                                        }
                                    ]
                                )
                            },
                        )
                    ],
                )
            if self.n == 2:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="t2",
                            tool_name="write_file",
                            args={
                                "path": "reports/brief.md",
                                "content": "# Brief\nDeep agent works.",
                            },
                        )
                    ],
                )
            if self.n == 3:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="t3",
                            tool_name="update_todo",
                            args={"index": 0, "status": "completed"},
                        )
                    ],
                )
            return ModelResponse(content="Deliverable written to reports/brief.md")

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_DeepScript()),
        workspace=tmp_path,
        web_search=False,
        url_fetch=False,
        citations=False,
        images=False,
        enable_task_tool=False,
        think_tool=False,
        modalities="text",
        max_tool_iterations=10,
        use_llm_summarizer=False,
    )
    result = await agent.arun("Write a short brief about deep agents")
    text = result.output.text() or ""
    assert "brief.md" in text.lower() or "Deliverable" in text
    assert (tmp_path / "reports" / "brief.md").is_file()
    body = (tmp_path / "reports" / "brief.md").read_text(encoding="utf-8")
    assert "Deep agent works" in body
    todos = TodoStore(tmp_path / "todos.json").list()
    assert todos and todos[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_task_tool_spawns_specialist() -> None:
    class _Echo:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="specialist-ok")

    model = ModelSpec(provider="scripted", provider_impl=_Echo())
    tool = make_task_tool(model=model, modalities="text")
    out = _content(await tool.invoke({"description": "Do the thing", "role": "Researcher"}))
    assert "specialist-ok" in out


def test_workspace_store_rejects_traversal() -> None:
    store = WorkspaceStore()
    assert store.write("../etc/passwd", "x") is None
    assert store.read("../etc/passwd") is None


@pytest.mark.asyncio
async def test_task_tool_shares_workspace(tmp_path) -> None:
    """Specialists receive shared workspace tools and can write files parent can read."""

    class _Writer:
        def __init__(self) -> None:
            self.wrote = False

        async def complete(self, request: ModelRequest) -> ModelResponse:
            if not self.wrote:
                self.wrote = True
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="w1",
                            tool_name="write_file",
                            args={
                                "path": "notes/from_specialist.md",
                                "content": "specialist note",
                            },
                        )
                    ],
                )
            return ModelResponse(content="wrote notes/from_specialist.md")

    from loomable.toolkits.workspace_tools import WorkspaceTools

    model = ModelSpec(provider="scripted", provider_impl=_Writer())
    tool = make_task_tool(
        model=model,
        tools=[WorkspaceTools(root=tmp_path)],
        modalities="text",
    )
    out = _content(await tool.invoke({"description": "Write the note file", "role": "Writer"}))
    assert "specialist" in out.lower() or "wrote" in out.lower()
    assert (tmp_path / "notes" / "from_specialist.md").is_file()


def test_create_research_agent_alias(tmp_path) -> None:
    from loomable.agent.deep import create_research_agent

    class _Noop:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="ok")

    agent = create_research_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=tmp_path,
        web_search=False,
        url_fetch=False,
        think_tool=False,
        enable_task_tool=False,
        use_llm_summarizer=False,
    )
    assert agent._name == "research-agent"
    names = set(agent.build().tool_runtime._tools.keys())
    assert "register_source" in names
    assert "fetch_image" in names
