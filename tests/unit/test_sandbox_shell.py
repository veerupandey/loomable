"""Sandbox + ShellTools unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from loomable.agent import ModelSpec, create_deep_agent
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.sandbox import SubprocessSandbox, make_sandbox
from loomable.sandbox.subprocess_backend import shell_command_allowed
from loomable.skills import list_bundled_skills, resolve_skills
from loomable.toolkits import PythonTools, ShellTools


class _Noop:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="done")


def test_shell_policy_blocks_destructive() -> None:
    assert shell_command_allowed("rm -rf /") is not None
    assert shell_command_allowed("sudo ls") is not None
    assert shell_command_allowed("echo hello") is None


@pytest.mark.asyncio
async def test_subprocess_sandbox_python_and_shell(tmp_path: Path) -> None:
    sb = SubprocessSandbox(root=tmp_path, timeout=10)
    py = await sb.run_python("print('hi-sandbox')")
    assert py.returncode == 0
    assert "hi-sandbox" in py.stdout

    sh = await sb.run_shell("echo shell-ok")
    assert sh.returncode == 0
    assert "shell-ok" in sh.stdout

    blocked = await sb.run_shell("sudo id")
    assert blocked.returncode == 126
    assert "blocked" in (blocked.error or "").lower()


@pytest.mark.asyncio
async def test_shell_tools_and_python_tools_share_root(tmp_path: Path) -> None:
    sb = make_sandbox(str(tmp_path), timeout=10)
    py = PythonTools(sandbox=sb)
    sh = ShellTools(sandbox=sb)
    out = await py._run_python("print(123)")
    assert "123" in out
    out2 = await sh._run_shell("pwd")
    assert str(tmp_path) in out2 or out2.strip()


@pytest.mark.asyncio
async def test_python_file_path_escape_blocked(tmp_path: Path) -> None:
    sb = SubprocessSandbox(root=tmp_path, timeout=10)
    tools = PythonTools(sandbox=sb)
    result = await tools._run_python_file("../../etc/passwd")
    assert "escapes" in result.lower() or "error" in result.lower()


def test_browser_skill_bundled() -> None:
    assert "browser" in list_bundled_skills()
    paths = resolve_skills(["browser"])
    assert paths and (paths[0] / "SKILL.md").is_file()


@pytest.mark.asyncio
async def test_create_deep_agent_code_exec_and_shell(tmp_path: Path) -> None:
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
        shell=True,
        modalities="text",
        use_llm_summarizer=False,
        discovery=True,
    )
    built = agent.build()
    # Deferred under discovery core — present in catalog.
    names = {t.name for t in built.discovery.catalog.tools}
    assert "run_python" in names or "run_python" in built.tool_runtime._tools
    assert "run_shell" in names or "run_shell" in built.tool_runtime._tools
    # HITL confirmation required for exec tools
    assert "run_python" in (agent._require_confirmation or [])
    assert "run_shell" in (agent._require_confirmation or [])
