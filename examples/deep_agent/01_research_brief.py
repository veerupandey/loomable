"""Deep Agent — long-horizon harness on loomable (``create_deep_agent``).

Pillars:
  1. write_todos planning
  2. Workspace FS (ls/read/write/edit/glob/grep)
  3. task subagent tool
  4. think + optional memory / Case spine

Live by default when an API key is set (via ``examples/_provider.py``).
Force the offline scripted path with ``DEEP_AGENT_SCRIPTED=1``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import has_live_provider, require_provider  # noqa: E402

from loomable.agent import ModelSpec, create_deep_agent
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall

ROOT = Path(__file__).resolve().parent / ".workspace"
ROOT.mkdir(parents=True, exist_ok=True)


class _ScriptedDeepProvider:
    """Deterministic deep-agent loop for CI / offline demos."""

    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        tool_name="write_todos",
                        args={
                            "todos": json.dumps(
                                [
                                    {
                                        "content": "Outline research questions",
                                        "status": "completed",
                                    },
                                    {
                                        "content": "Draft findings to workspace",
                                        "status": "in_progress",
                                    },
                                    {
                                        "content": "Write final brief",
                                        "status": "pending",
                                    },
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
                        id="2",
                        tool_name="write_file",
                        args={
                            "path": "notes/findings.md",
                            "content": (
                                "# Findings\n"
                                "- Deep agents need planning + filesystem offload.\n"
                                "- Subagents keep the parent context small.\n"
                            ),
                        },
                    )
                ],
            )
        if self.n == 3:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="3",
                        tool_name="write_file",
                        args={
                            "path": "reports/brief.md",
                            "content": (
                                "# Research Brief: Deep Agents\n\n"
                                "Deep agents succeed on long-horizon work by combining "
                                "explicit todos, a workspace filesystem, and delegated "
                                "sub-tasks — exactly what loomable's create_deep_agent "
                                "wires by default.\n"
                            ),
                        },
                    )
                ],
            )
        if self.n == 4:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="4",
                        tool_name="update_todo",
                        args={"index": 1, "status": "completed"},
                    ),
                    ToolCall(
                        id="5",
                        tool_name="update_todo",
                        args={"index": 2, "status": "completed"},
                    ),
                ],
            )
        return ModelResponse(
            content=(
                "Brief ready at reports/brief.md. "
                "Key insight: plan → offload → delegate → deliver."
            )
        )


def _resolve_model():
    # Prefer a live provider whenever a key is present; set
    # DEEP_AGENT_SCRIPTED=1 to force the offline scripted path (CI).
    if os.environ.get("DEEP_AGENT_SCRIPTED") == "1":
        return ModelSpec(provider="scripted", provider_impl=_ScriptedDeepProvider())
    if has_live_provider() or os.environ.get("DEEP_AGENT_LIVE"):
        return require_provider()
    return ModelSpec(provider="scripted", provider_impl=_ScriptedDeepProvider())


async def main() -> None:
    model = _resolve_model()
    live = not (isinstance(model, ModelSpec) and model.provider == "scripted")
    agent = create_deep_agent(
        model,
        workspace=ROOT,
        web_search=live,
        url_fetch=live,
        citations=live,
        images=False,
        enable_task_tool=True,
        think_tool=True,
        modalities="text",
        use_llm_summarizer=False,
        instructions=(
            "Topic: why deep agents need todos + filesystem + subagents. "
            "Write the final answer to reports/brief.md."
        ),
    )
    result = await agent.arun(
        "Produce a short research brief on deep-agent architecture. "
        "Use todos and the workspace. Final file: reports/brief.md"
    )
    print("--- output ---")
    print(result.output.text())
    brief = ROOT / "reports" / "brief.md"
    if brief.is_file():
        print("--- reports/brief.md ---")
        print(brief.read_text(encoding="utf-8")[:1200])
    else:
        print("(no reports/brief.md yet — live models may choose another path)")


if __name__ == "__main__":
    asyncio.run(main())
