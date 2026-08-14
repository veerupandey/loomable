"""Live multimodal research deep agent — beat LangGraph deepagents on loomable.

Demonstrates (real APIs when keys are set):
  - web_search + fetch_url / extract_text
  - fetch_image + analyze_image (vision)
  - CitationTools + WorkspaceTools offload
  - Memory.compose conversation + user facts
  - task subagents sharing the workspace
  - research skill (SKILL.md)

Scripted CI path (no API key)::

    python examples/deep_agent/04_live_multimodal_research.py

Live::

    DEEP_AGENT_LIVE=1 GEMINI_API_KEY=... \\
      python examples/deep_agent/04_live_multimodal_research.py

    # or OpenAI
    DEEP_AGENT_LIVE=1 OPENAI_API_KEY=... DEEP_MODEL=openai:gpt-4o-mini \\
      python examples/deep_agent/04_live_multimodal_research.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable.agent import ModelSpec, NoteStore, create_deep_agent
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.memory import ConversationMemory, Memory, UserMemory, open_session_store

ROOT = Path(__file__).resolve().parent / ".workspace_research"
ROOT.mkdir(parents=True, exist_ok=True)

TOPIC = os.environ.get(
    "DEEP_RESEARCH_TOPIC",
    "How LangGraph deep agents use filesystem offload and subagents",
)


class _ScriptedResearchProvider:
    """Deterministic multimodal research loop for CI."""

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
                                        "content": "Search and fetch sources",
                                        "status": "in_progress",
                                    },
                                    {
                                        "content": "Register citations + analyze image",
                                        "status": "pending",
                                    },
                                    {
                                        "content": "Write reports/research.md",
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
                            "path": "notes/search_dump.md",
                            "content": (
                                "# Search dump\n"
                                "- Deep agents offload large tool results to a filesystem.\n"
                                "- Subagents isolate context; loomable also shares workspace.\n"
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
                        tool_name="register_source",
                        args={
                            "url": "https://docs.langchain.com/oss/python/deepagents/overview",
                            "title": "Deep Agents overview",
                            "summary": "Official deepagents harness docs",
                            "quote": "filesystem, subagents, context management",
                        },
                    )
                ],
            )
        if self.n == 4:
            img = ROOT / "images" / "diagram.png"
            img.parent.mkdir(parents=True, exist_ok=True)
            if not img.exists():
                img.write_bytes(b"\x89PNG\r\n\x1a\nfake-diagram")
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="4",
                        tool_name="write_file",
                        args={
                            "path": "images/analysis_diagram.md",
                            "content": (
                                "# Image analysis: images/diagram.png\n\n"
                                "Schematic of filesystem offload and subagent fan-out.\n"
                            ),
                        },
                    )
                ],
            )
        if self.n == 5:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="5",
                        tool_name="format_bibliography",
                        args={},
                    )
                ],
            )
        if self.n == 6:
            bib = ""
            for msg in reversed(request.messages or []):
                if msg.get("role") == "tool":
                    content = msg.get("content")
                    if isinstance(content, list):
                        parts = []
                        for p in content:
                            if isinstance(p, dict) and p.get("type") == "text":
                                parts.append(str(p.get("text") or ""))
                        bib = "\n".join(parts)
                    else:
                        bib = str(content or "")
                    break
            report = (
                "# Research Brief: Deep Agents\n\n"
                "## Summary\n"
                "Deep agent harnesses combine todos, filesystem offload, and subagents.\n"
                "Loomable adds shared workspace, citations, and vision in one framework.\n\n"
                "## Findings\n"
                "- Offload beats truncate for long research fetches.\n"
                "- Shared workspace lets specialists write files the parent can read.\n\n"
                "## Visual evidence\n"
                "See images/analysis_diagram.md\n\n"
                f"{bib}\n"
            )
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="6",
                        tool_name="write_file",
                        args={"path": "reports/research.md", "content": report},
                    )
                ],
            )
        if self.n == 7:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="7",
                        tool_name="update_todo",
                        args={"index": 0, "status": "completed"},
                    ),
                    ToolCall(
                        id="8",
                        tool_name="update_todo",
                        args={"index": 1, "status": "completed"},
                    ),
                    ToolCall(
                        id="9",
                        tool_name="update_todo",
                        args={"index": 2, "status": "completed"},
                    ),
                ],
            )
        return ModelResponse(
            content="Deliverable ready at reports/research.md with citations and image notes."
        )


def _pick_live_model() -> object:
    if os.environ.get("DEEP_MODEL"):
        raw = os.environ["DEEP_MODEL"]
        if raw.startswith("gemini:") or (
            "GEMINI_API_KEY" in os.environ and not raw.startswith("openai:")
        ):
            from loomable.providers.gemini import GeminiProvider

            model_name = raw.split(":", 1)[-1] if ":" in raw else raw
            return GeminiProvider(
                model=model_name,
                api_key=os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY"),
            )
        return raw
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        from loomable.providers.gemini import GeminiProvider

        return GeminiProvider(
            model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
            api_key=os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY"),
        )
    if os.environ.get("OPENAI_API_KEY"):
        return "openai:gpt-4o-mini"
    raise RuntimeError("Set GEMINI_API_KEY or OPENAI_API_KEY (or DEEP_MODEL)")


class _Emb:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0, 0.0]


async def main() -> None:
    live = os.environ.get("DEEP_AGENT_LIVE", "").strip() in {"1", "true", "yes"}
    conv_store = open_session_store("file", path=str(ROOT / "sessions"))
    notes = NoteStore(long_term=LongTermStore(), embedder=_Emb())
    memory = Memory.compose(
        conversation=ConversationMemory(store=conv_store, window=16),
        user=UserMemory(note_store=notes, auto_extract=True, memory_tool=True),
    )

    if live:
        model: object = _pick_live_model()
        print(f"Live research model: {model}")
    else:
        model = ModelSpec(provider="scripted", provider_impl=_ScriptedResearchProvider())
        print("Scripted research demo (set DEEP_AGENT_LIVE=1 for real APIs)")

    agent = create_deep_agent(
        model,
        profile="research",
        workspace=ROOT,
        session_id="research-demo",
        memory=memory,
        skills=["research"],  # bundled topic-agnostic research skill
        memory_tool=True,
        web_search=live,
        url_fetch=live,
        images=live,  # live: real fetch_image/analyze_image; scripted writes notes via write_file
        think_tool=False,
        enable_task_tool=live,
        use_llm_summarizer=False if not live else True,
        max_tool_iterations=40 if live else 12,
        debug=os.environ.get("DEEP_DEBUG", "") in {"1", "true"},
    )

    prompt = (
        f"Research: {TOPIC}\n\n"
        "Search the web at most twice, then extract_text on the best source "
        "(prefer https://docs.langchain.com/oss/python/deepagents/overview when relevant). "
        "register_source for sources you use, write reports/research.md with a bibliography, "
        "download/analyze an image only if easy, and complete your todos."
    )
    result = await agent.arun(prompt)
    print(result.output.text() or "(no text)")

    report = ROOT / "reports" / "research.md"
    sources = ROOT / "sources.json"
    print(f"workspace: {ROOT}")
    print(f"report exists: {report.is_file()}")
    print(f"sources exists: {sources.is_file()}")
    if report.is_file():
        print("--- report preview ---")
        print(report.read_text(encoding="utf-8")[:1200])

    # Second turn: memory recall (scripted or live)
    follow = await agent.arun(
        "In one sentence, what deliverable path did you write for the previous research?"
    )
    print("follow-up:", follow.output.text() or "(no text)")


if __name__ == "__main__":
    asyncio.run(main())
