"""Tests for workspace tool-result offload."""

from __future__ import annotations

from pathlib import Path

import pytest

from loomable.agent import ModelSpec, create_deep_agent
from loomable.agent.offload import make_workspace_offload_hook, offload_tool_text
from loomable.agent.tools import FunctionTool
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall, ToolOutcome, ToolResult


def test_offload_tool_text_writes_file(tmp_path: Path) -> None:
    body = "x" * 5000
    rel, msg = offload_tool_text(tmp_path, "fetch_url", body, preview_chars=100)
    assert rel.startswith(".offload/")
    assert (tmp_path / rel).read_text(encoding="utf-8") == body
    assert "offloaded 5000 chars" in msg
    assert "read_file" in msg


def test_offload_visible_via_workspace_store(tmp_path: Path) -> None:
    from loomable.toolkits.workspace_tools import WorkspaceStore

    store = WorkspaceStore(tmp_path)
    body = "VISIBLE_" + ("z" * 200)
    rel, _ = offload_tool_text(tmp_path, "extract_text", body, store=store)
    assert store.read(rel) == body
    assert rel in store.glob(".offload/*")


def test_offload_hook_transforms_large_result(tmp_path: Path) -> None:
    hook = make_workspace_offload_hook(tmp_path, threshold=100, preview_chars=20)
    big = "y" * 200
    outcome = ToolOutcome(
        call_id="c1",
        result=ToolResult(content=big),
    )
    replaced = hook("fetch_url", ToolCall(id="c1", tool_name="fetch_url", args={}), outcome)
    assert isinstance(replaced, ToolOutcome)
    assert replaced.result is not None
    assert "offloaded" in str(replaced.result.content)
    assert replaced.result.metadata.get("offloaded") is True
    # Small results unchanged
    small = ToolOutcome(call_id="c2", result=ToolResult(content="ok"))
    assert hook("fetch_url", None, small) is None


@pytest.mark.asyncio
async def test_deep_agent_offloads_large_tool_result(tmp_path: Path) -> None:
    big_html = "<html>" + ("p" * 15_000) + "</html>"

    async def huge_page(url: str = "") -> str:
        """Return a huge HTML blob to trigger offload."""
        return big_html

    class _Script:
        def __init__(self) -> None:
            self.n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            if self.n == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="1", tool_name="huge_page", args={"url": "https://x"})
                    ],
                )
            # Capture last tool message content for assertion via side channel
            for msg in reversed(request.messages or []):
                if msg.get("role") == "tool":
                    self.last_tool = msg.get("content")
                    break
            return ModelResponse(content="done after offload")

    provider = _Script()
    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=provider),
        workspace=tmp_path,
        web_search=False,
        url_fetch=False,
        citations=False,
        images=False,
        enable_task_tool=False,
        think_tool=False,
        modalities="text",
        discovery=False,
        board=False,
        offload_large_tools=True,
        offload_threshold=1000,
        tools=[FunctionTool(huge_page, name="huge_page")],
        max_tool_iterations=5,
        use_llm_summarizer=False,
    )
    result = await agent.arun("Fetch the page")
    assert "done" in (result.output.text() or "").lower()
    offloads = list((tmp_path / ".offload").glob("*.txt"))
    assert offloads, "expected offload file"
    assert big_html in offloads[0].read_text(encoding="utf-8")
    tool_msg = getattr(provider, "last_tool", "")
    assert "offloaded" in str(tool_msg)
    # Agent can still read the full body via workspace tools
    from loomable.toolkits.workspace_tools import WorkspaceTools

    ws = WorkspaceTools(root=tmp_path)
    read = next(t for t in ws.tools() if t.name == "read_file")
    rel = str(offloads[0].relative_to(tmp_path)).replace("\\", "/")
    recovered = str((await read.invoke({"path": rel})).content)
    assert big_html in recovered
