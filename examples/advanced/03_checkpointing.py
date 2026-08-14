"""Checkpointing — Durable Workflow state and resume.

USE WHEN: Your workflow is long-running and you need to pause/resume
it across process restarts, or you have human-in-the-loop approval
gates that may take hours.

Prefer ``Workflow(..., checkpointer=...)`` (or ``sequential(..., checkpointer=...)``).
For a fuller kill/resume exam, see ``escalation_war_room/05_checkpoint_resume.py``.

This demo is offline (scripted steps) and writes a complete checkpoint.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from loomable import Workflow
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, Text
from loomable.persist import JsonFileCheckpointer

ROOT = Path(__file__).resolve().parent / ".checkpoint_demo"
SESSION = "checkpoint-demo"


async def draft(inp, *, context=None):
    return RunResult(
        output=AgentOutput(parts=[Text(f"DRAFT: Testing matters because {inp}.")]),
        session_id=SESSION,
    )


async def review(inp, *, context=None):
    text = inp.text() if hasattr(inp, "text") and callable(inp.text) else str(inp)
    return RunResult(
        output=AgentOutput(parts=[Text(f"APPROVED: {text}")]),
        session_id=SESSION,
        structured={"ok": True},
    )


async def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    # File-backed checkpointer (InMemoryCheckpointer() also works for tests)
    cp = JsonFileCheckpointer(str(ROOT / "checkpoints"))
    wf = (
        Workflow("checkpoint-demo", session_id=SESSION, checkpointer=cp)
        .step("draft", draft)
        .step("review", review)
    )
    result = await wf.arun("flaky deployments hurt users")
    print(result.output.text())

    saved = await cp.get(SESSION)
    assert saved is not None and saved.complete is True
    print(f"checkpoint complete={saved.complete} step={saved.step}")


if __name__ == "__main__":
    asyncio.run(main())
