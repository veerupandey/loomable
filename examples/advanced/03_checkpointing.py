"""Checkpointing — Agents as Workflow steps (framework passes outputs).

USE WHEN: A multi-step process needs pause/resume across restarts.

Do **not** write parse helpers between steps. Put Agents on the Workflow;
each Agent already receives the previous Agent's output as its input::

    Workflow(...).step("draft", drafter).step("review", reviewer)

Same idea for ``sequential(drafter, reviewer)`` and ``Team(mode="sequential")``.

Fuller kill/resume exam: ``escalation_war_room/05_checkpoint_resume.py``.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from loomable import Agent, JsonFileCheckpointer, Workflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _offline import scripted_model  # noqa: E402

ROOT = Path(__file__).resolve().parent / ".checkpoint_demo"
SESSION = "checkpoint-demo"


async def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    drafter = Agent(
        scripted_model(["DRAFT: Testing matters because flaky deployments hurt users."]),
        role="Drafter",
        goal="Write a short draft on the topic",
        use_llm_summarizer=False,
    )
    # Reviewer sees the draft text automatically (prior AgentOutput → user turn).
    reviewer = Agent(
        scripted_model([{"echo": "APPROVED: {input}"}]),
        role="Reviewer",
        goal="Review the draft and approve or request changes",
        use_llm_summarizer=False,
    )

    cp = JsonFileCheckpointer(str(ROOT / "checkpoints"))
    wf = (
        Workflow("checkpoint-demo", session_id=SESSION, checkpointer=cp)
        .step("draft", drafter)
        .step("review", reviewer)
    )
    result = await wf.arun("flaky deployments hurt users")
    print(result.output.text())

    saved = await cp.get(SESSION)
    assert saved is not None and saved.complete is True
    print(f"checkpoint complete={saved.complete} step={saved.step}")


if __name__ == "__main__":
    asyncio.run(main())
