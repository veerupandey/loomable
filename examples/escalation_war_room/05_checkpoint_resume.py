"""Checkpoint resume — live Agents; prior output chains automatically.

Simulate a crash after gather, then resume so gather is skipped and scribe
runs on the gather Agent's output (no parse helpers).
"""

from __future__ import annotations

import asyncio
import shutil

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, JsonFileCheckpointer, Workflow
from loomable.agent.context import RunContext
from loomable.flow.state import SharedState
from loomable.persist.checkpoint import Checkpoint

from _common import ROOT, make_provider

CKPT_DIR = ROOT / ".checkpoints_phase_b"
SESSION = "inc-88421-resume"


async def main() -> None:
    if CKPT_DIR.exists():
        shutil.rmtree(CKPT_DIR)
    CKPT_DIR.mkdir(parents=True)

    cp = JsonFileCheckpointer(str(CKPT_DIR))
    model = make_provider()

    gatherer = Agent(
        model,
        role="Gatherer",
        goal="Collect incident evidence from the email",
        instructions="Summarize the incident in 2-4 short lines with key facts.",
    )
    scribe = Agent(
        model,
        role="Scribe",
        goal="Turn evidence into an escalation packet",
        instructions=(
            "You receive the previous gatherer's output as your input. "
            "Write a short escalation packet summarizing impact and next actions."
        ),
    )
    (
        Workflow("war-room-resume", session_id=SESSION, checkpointer=cp)
        .step("gather", gatherer)
        .step("scribe", scribe)
    )

    # Drive gather only, then write incomplete checkpoint (simulate kill).
    state = SharedState()
    ctx = RunContext(shared_state=state)
    gather_result = await gatherer.arun(
        "SEV-1: UPI settlement batches stuck since 18:40 IST for BharatNova.",
        context=ctx,
    )
    state.write("gather", gather_result.output)
    await cp.put(
        Checkpoint(
            thread_id=SESSION,
            step=1,
            session_state={
                "shared_state": state.snapshot(),
                "completed_node_ids": ["gather"],
            },
            complete=False,
        )
    )
    print("[kill] incomplete checkpoint after gather Agent")
    print(f"    gather={(gather_result.output.text() or '')[:160]}")

    # Fresh Agents on resume — gather skipped; scribe sees prior Agent output.
    gather2 = Agent(
        make_provider(),
        role="Gatherer",
        goal="Collect incident evidence",
        instructions="Summarize the incident briefly.",
    )
    scribe2 = Agent(
        make_provider(),
        role="Scribe",
        goal="Turn evidence into an escalation packet",
        instructions=(
            "You receive the previous gatherer's output. "
            "Write a short escalation packet."
        ),
    )
    wf2 = (
        Workflow("war-room-resume", session_id=SESSION, checkpointer=cp)
        .step("gather", gather2)
        .step("scribe", scribe2)
    )
    result = await wf2.arun(
        "SEV-1: UPI settlement batches stuck since 18:40 IST for BharatNova.",
        resume=True,
    )

    skipped = result.metadata.get("skipped_nodes") or []
    assert result.metadata.get("resumed") is True
    assert "gather" in skipped, skipped
    out = (result.output.text() or "").strip()
    assert out, out

    await wf2.clear_checkpoint()
    try:
        await wf2.arun("x", resume=True)
        raise AssertionError("expected resume=True to fail without checkpoint")
    except RuntimeError as exc:
        assert "no incomplete checkpoint" in str(exc).lower()

    print("[ok] Phase B kill/resume (live Agents, seamless handoff)")
    print(f"    skipped={skipped}")
    print(f"    output={out[:240]}")


if __name__ == "__main__":
    asyncio.run(main())
