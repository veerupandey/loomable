"""Progressive capability discovery — metadata, then load/activate, then call.

Shows the "search → load/activate → call" pattern that ``discovery=True``
wires into every :func:`create_deep_agent`:

  1. ``search_skills`` — discover the bundled ``research`` skill by query
     (name + description only; the body is not in context yet).
  2. ``load_skill`` — pull the skill's full instructions into context.
  3. ``search_tools`` — find a deferred tool (PDF reading, in this case)
     without ever advertising its full schema up front.
  4. ``activate_tool`` — register it so the model can call it, and get its
     schema back in the same tool result.

Everything below is scripted (no network / API key required) so it runs in
CI. See ``01_research_brief.py`` for the general deep-agent loop and
``04_live_multimodal_research.py`` for a live-model run.

Run::

    python examples/deep_agent/02_progressive_discovery.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from loomable.agent import ModelSpec, create_deep_agent
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall

ROOT = Path(__file__).resolve().parent / ".workspace_discovery"
ROOT.mkdir(parents=True, exist_ok=True)

_DISCOVERY_TOOLS = {"search_skills", "load_skill", "search_tools", "activate_tool"}


class _ScriptedDiscoveryProvider:
    """Deterministic search → load → search → activate → deliver walk."""

    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        if self.n == 1:
            # Step 1: discover the research skill by query (metadata only).
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id="1", tool_name="search_skills", args={"query": "research"})
                ],
            )
        if self.n == 2:
            # Step 2: load it — full SKILL.md body now enters context.
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id="2", tool_name="load_skill", args={"name": "research"})
                ],
            )
        if self.n == 3:
            # Step 3: search for a deferred tool instead of assuming it exists.
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id="3", tool_name="search_tools", args={"query": "pdf"})
                ],
            )
        if self.n == 4:
            # Step 4: activate the tool the search surfaced. Its schema comes
            # back in the tool result, and the next turn's request.tools
            # includes it automatically — no schema paid until now.
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id="4", tool_name="activate_tool", args={"name": "read_pdf"})
                ],
            )
        if self.n == 5:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="5",
                        tool_name="write_file",
                        args={
                            "path": "reports/discovery.md",
                            "content": (
                                "# Progressive discovery demo\n\n"
                                "Loaded the research skill and activated read_pdf "
                                "on demand instead of advertising every tool "
                                "schema up front.\n"
                            ),
                        },
                    )
                ],
            )
        return ModelResponse(
            content=(
                "Delivered reports/discovery.md. Discovered the research skill "
                "and the read_pdf tool progressively, via search + activate — "
                "never paid the full schema/body cost up front."
            )
        )


async def main() -> None:
    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_ScriptedDiscoveryProvider()),
        workspace=ROOT,
        web_search=False,
        url_fetch=False,
        citations=False,
        images=False,
        documents=True,  # registers read_pdf/search_pdf — deferred under discovery
        enable_task_tool=False,
        think_tool=False,
        modalities="text",
        use_llm_summarizer=False,
    )

    result = await agent.arun(
        "Use the research skill's workflow, then confirm a PDF tool is "
        "available before writing the deliverable."
    )

    print("--- discovery tool calls (search/load/activate) ---")
    for outcome in result.tool_activity:
        metadata = (outcome.result.metadata or {}) if outcome.result else {}
        tool_name = str(metadata.get("tool_name", ""))
        if tool_name not in _DISCOVERY_TOOLS:
            continue
        content = outcome.result.content if outcome.result else outcome.error
        print(f"\n[{tool_name}]")
        try:
            print(json.dumps(json.loads(str(content)), indent=2)[:600])
        except (TypeError, ValueError):
            print(str(content)[:600])

    print("\n--- output ---")
    print(result.output.text())

    report = ROOT / "reports" / "discovery.md"
    if report.is_file():
        print("\n--- reports/discovery.md ---")
        print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
