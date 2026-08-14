"""Checkpointing — live Agents as Workflow steps.

Each Agent receives the previous Agent's output automatically.
Requires a real LLM key (``GEMINI_API_KEY`` / OpenAI / Azure) — see ``.env.example``.

Run::

    python examples/advanced/03_checkpointing.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, JsonFileCheckpointer, Workflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

ROOT = Path(__file__).resolve().parent / ".checkpoint_demo"
SESSION = "checkpoint-demo"


async def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    model = require_provider()
    drafter = Agent(
        model,
        role="Drafter",
        goal="Write a short draft on the topic",
        instructions="Write 2-3 clear sentences. No preamble.",
    )
    reviewer = Agent(
        model,
        role="Reviewer",
        goal="Review the draft and approve or request changes",
        instructions=(
            "You receive the previous agent's draft as your input. "
            "Reply with APPROVED: followed by a one-line justification, "
            "or CHANGES: with specific fixes."
        ),
    )

    cp = JsonFileCheckpointer(str(ROOT / "checkpoints"))
    wf = (
        Workflow("checkpoint-demo", session_id=SESSION, checkpointer=cp)
        .step("draft", drafter)
        .step("review", reviewer)
    )
    result = await wf.arun("Why testing matters for flaky deployments")
    print(result.output.text())

    saved = await cp.get(SESSION)
    assert saved is not None and saved.complete is True
    print(f"checkpoint complete={saved.complete} step={saved.step}")


if __name__ == "__main__":
    asyncio.run(main())
