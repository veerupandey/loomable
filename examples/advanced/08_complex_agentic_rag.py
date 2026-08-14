"""Complex multi-format RAG — live Agent + Workflow (framework does the work).

What the framework handles (you do not wire this by hand):
  - ingest of md / rst / html / json / csv / code (and pdf/docx/pptx/URLs)
  - chunking (``strategy="auto"``)
  - named ``knowledge_base=`` collections → ``search_docs`` / ``search_code`` tools
  - the model calling those tools and answering
  - Workflow passing one Agent's output to the next

Requires a live LLM key — see ``.env.example``.

Run::

    python examples/advanced/08_complex_agentic_rag.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, Workflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "sample_rag_corpus"
DOCS = CORPUS / "docs"
CODE = CORPUS / "code"

# Cross-format questions — the Agent must search, not invent.
QUESTIONS = [
    "How do clients authenticate? Cite the doc filename.",
    "What does require_bearer raise on a bad header?",
    "What SKU includes SSO, and what is ap-south latency?",
    "What webhook secret should we rotate after a breach?",
]


async def main() -> None:
    model = require_provider()
    kb = {"docs": [DOCS], "code": [CODE]}

    # --- One Agent: framework ingests corpora and exposes search_* tools ---
    agent = Agent(
        model,
        knowledge_base=kb,
        instructions=(
            "You have search_docs and search_code. "
            "Always search before answering. Cite filenames. Be concise."
        ),
        max_tool_iterations=8,
    )
    print("tools:", sorted(agent.build().tool_runtime._tools))
    print()

    for question in QUESTIONS:
        print(f"=== {question} ===")
        result = await agent.arun(question)
        print((result.output.text() or "").strip())
        print()

    # --- Workflow: researcher (with KB) → writer (reads prior Agent output) ---
    researcher = Agent(
        model,
        role="RAG Researcher",
        knowledge_base=kb,
        instructions=(
            "Use search_docs and search_code. "
            "Return short bullets with filename citations only."
        ),
        max_tool_iterations=8,
    )
    writer = Agent(
        model,
        role="Brief Writer",
        instructions=(
            "You receive the researcher's bullets as your input. "
            "Write a 5-line ops brief. Do not invent facts."
        ),
    )
    wf = (
        Workflow("complex-rag-brief")
        .step("research", researcher)
        .step("write", writer)
    )
    brief = await wf.arun(
        "Produce an ops brief covering auth, webhook rotation, and enterprise SSO SKU."
    )
    print("=== Workflow brief (research → write) ===")
    print((brief.output.text() or "").strip())


if __name__ == "__main__":
    asyncio.run(main())
