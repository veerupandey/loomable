"""Build a research agent with L1/L2/L3 memory, PDFs, web search, and subagent fan-out.

USE WHEN: You need one agent that:
  - Remembers the conversation (L1), compacts old turns (L2), and stores user facts (L3)
  - Reads multi-page PDFs (``read_pdf`` / ``search_pdf``)
  - Searches the web for fresh information
  - Fans out to specialist subagents (or ``plan()``) on complex, multi-part questions

You do **not** call ``agent.build()`` for normal runs — ``await agent.arun(...)`` builds once
and caches the runtime automatically.

Run (requires a live LLM key — see repo ``.env.example``)::

    python examples/agents/09_research_memory_agent.py

Or import the factory::

    from examples.agents.research_memory_agent import build_research_agent
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Sequence

from loomable import (
    Agent,
    ConversationMemory,
    KnowledgeMemory,
    Memory,
    UserMemory,
    open_session_store,
)
from loomable.agent import NoteStore
from loomable.kernel.long_term import LongTermStore
from loomable.providers.vector_store import open_vector_store
from loomable.toolkits import WebSearchTools, WorkspaceTools

def _pdf_tools_available() -> bool:
    """True when ``pypdf`` is installed (``loomable[pdf]`` extra)."""
    try:
        import pypdf  # noqa: F401

        return True
    except ImportError:
        return False


_PDF_AVAILABLE = _pdf_tools_available()


def _extract_pdf_text_chunks(
    pdf_paths: Sequence[Path],
    *,
    max_chars_per_doc: int = 12_000,
) -> list[str]:
    """Pre-load PDF text into L3 knowledge snippets (optional at build time)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return []

    chunks: list[str] = []
    for path in pdf_paths:
        if not path.is_file():
            continue
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text)
            if sum(len(p) for p in parts) >= max_chars_per_doc:
                break
        body = "\n\n".join(parts)[:max_chars_per_doc]
        if body:
            chunks.append(f"PDF {path.name}:\n{body}")
    return chunks


def build_research_agent(
    model: Any,
    *,
    embedder: Any,
    workspace: str | Path,
    session_id: str,
    user_id: str,
    pdf_paths: Sequence[str | Path] | None = None,
    web_search_provider: str = "duckduckgo",
    web_search_api_key: str | None = None,
    scopes: dict[str, str] | None = None,
) -> Agent:
    """Return a configured :class:`~loomable.agent.builder.Agent` (call ``arun``, not ``build``).

    Memory layers
    -------------
    - **L1** — recent conversation turns (``ConversationMemory.window``)
    - **L2** — compacted summaries when turns exceed ``compaction_threshold``
    - **L3** — durable user facts + optional PDF knowledge (``UserMemory`` + ``KnowledgeMemory``)

    Complex questions
    -----------------
    - ``plan_tool=True`` — runtime plan → parallel step workers → synthesize
    - ``subagents`` — ``delegate_to_pdf_analyst``, ``delegate_to_web_researcher``, etc.
    """
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    docs_dir = root / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # --- L1/L2: conversation store (file-backed by default) ---
    session_store = open_session_store("file", path=str(root / "sessions"))
    conversation = ConversationMemory(
        store=session_store,
        window=12,
        compaction_threshold=24,
        use_llm_summarizer=True,
    )

    # --- L3: scoped long-term notes ---
    try:
        long_term = LongTermStore()
    except ImportError:
        long_term = open_vector_store(engine="memory")
    note_store = NoteStore(long_term=long_term, embedder=embedder)

    # Optional: index uploaded PDFs into passive knowledge (RAG at recall time)
    knowledge_docs: list[str] = []
    resolved_pdfs = [Path(p) for p in (pdf_paths or [])]
    if not resolved_pdfs and docs_dir.is_dir():
        resolved_pdfs = sorted(docs_dir.glob("*.pdf"))
    knowledge_docs.extend(_extract_pdf_text_chunks(resolved_pdfs))

    user = UserMemory(
        note_store=note_store,
        memory_tool=True,
        auto_extract=True,
        user_id=user_id,
        scopes=scopes,
    )
    knowledge = (
        KnowledgeMemory(documents=knowledge_docs, embedder=embedder, top_k=4)
        if knowledge_docs
        else None
    )

    memory = Memory.compose(
        conversation=conversation,
        user=user,
        knowledge=knowledge,
    )

    provider = model
    workspace_kit = WorkspaceTools(root=root)

    pdf_tools: list[Any] = []
    if _PDF_AVAILABLE:
        from loomable.toolkits import PDFTools

        pdf_tools = [PDFTools()]

    pdf_specialist_tools = [*pdf_tools, workspace_kit] if pdf_tools else [workspace_kit]
    pdf_specialist = Agent(
        model=provider,
        role="PDF Analyst",
        goal="Extract and analyze facts from PDF documents",
        instructions=(
            "You specialize in PDFs. Use read_pdf with page ranges for long files "
            "(e.g. pages='1-10', then '11-20'). Use search_pdf to find keywords. "
            "Return concise bullet facts with page references when possible."
            if pdf_tools
            else "PDF toolkit unavailable — ask the user to install loomable[pdf]."
        ),
        tools=pdf_specialist_tools,
        max_tool_iterations=16,
    )

    web_specialist = Agent(
        model=provider,
        role="Web Researcher",
        goal="Find current information on the public web",
        instructions=(
            "Search the web for authoritative, recent sources. "
            "Return titles, URLs, and short snippets. Prefer primary sources."
        ),
        tools=[
            WebSearchTools(
                provider=web_search_provider,
                api_key=web_search_api_key,
            ),
            workspace_kit,
        ],
        max_tool_iterations=12,
    )

    synthesis_specialist = Agent(
        model=provider,
        role="Research Synthesizer",
        goal="Merge PDF and web findings into one coherent answer",
        instructions=(
            "Combine inputs from other specialists into a structured brief: "
            "Executive summary, Key findings, Sources, Open questions."
        ),
        tools=[workspace_kit],
        max_tool_iterations=8,
    )

    pdf_hint = ""
    if resolved_pdfs:
        names = ", ".join(p.name for p in resolved_pdfs)
        pdf_hint = f"\nPre-loaded PDFs in knowledge: {names}. Also use read_pdf for full text."

    return Agent(
        model=provider,
        role="Research Lead",
        goal="Answer complex research questions using documents, web, and specialists",
        instructions=(
            "You coordinate PDF analysis, web research, and synthesis.\n\n"
            "For **simple** questions: answer directly with your tools.\n\n"
            "For **complex** questions (multiple sub-questions, compare/contrast, "
            "multi-document + web, or explicit 'research deeply'):\n"
            "1. Call plan() to decompose into parallel steps, OR\n"
            "2. Delegate: delegate_to_pdf_analyst, delegate_to_web_researcher, "
            "then delegate_to_research_synthesizer.\n\n"
            "Always cite PDF page ranges or web URLs. Save long outputs under "
            f"reports/ in the workspace ({root})."
            f"{pdf_hint}"
        ),
        memory=memory,
        session_id=session_id,
        user_id=user_id,
        scopes=scopes,
        tools=[
            *pdf_tools,
            WebSearchTools(
                provider=web_search_provider,
                api_key=web_search_api_key,
            ),
            workspace_kit,
        ],
        subagents=[pdf_specialist, web_specialist, synthesis_specialist],
        think_tool=True,
        plan_tool=True,
        memory_window=12,
        compaction_threshold=24,
        use_llm_summarizer=True,
        max_tool_iterations=40,
        max_delegations=8,
        max_depth=3,
    )


async def demo() -> None:
    """Run a short demo: simple question, then a complex multi-track question."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _provider import make_embedder, require_provider  # noqa: E402

    model = require_provider()
    embedder = make_embedder()
    root = Path(__file__).resolve().parent / ".workspace_research_agent"
    root.mkdir(parents=True, exist_ok=True)

    # Drop sample PDF paths here, or put PDFs in .workspace_research_agent/documents/
    agent = build_research_agent(
        model,
        embedder=embedder,
        workspace=root,
        session_id="research-demo-1",
        user_id="researcher",
        pdf_paths=[],  # e.g. [root / "documents" / "report.pdf"]
    )

    print("=== Turn 1: preference (L3 memory) ===")
    r1 = await agent.arun(
        "My name is Sam. I prefer bullet summaries and care most about EU AI regulation."
    )
    print(r1.output.text()[:800])

    print("\n=== Turn 2: simple recall ===")
    r2 = await agent.arun("What do you remember about my preferences?")
    print(r2.output.text()[:800])

    print("\n=== Turn 3: complex (plan + subagents + web) ===")
    r3 = await agent.arun(
        "Research deeply: What are the top 3 EU AI Act obligations for "
        "general-purpose AI providers in 2025?\n"
        "Compare official guidance with recent news. "
        "If I uploaded PDFs, cross-check against those too. "
        "Structure: summary, findings, sources, gaps."
    )
    print(r3.output.text()[:2000])
    if r3.tool_activity:
        tools = [
            (o.result.metadata or {}).get("tool_name", "?")
            for o in r3.tool_activity
            if o.result
        ]
        print("\nTools used:", ", ".join(t for t in tools if t))


if __name__ == "__main__":
    asyncio.run(demo())
