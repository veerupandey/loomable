"""Phase B gate — kill mid-workflow and resume (live Agents).

Gather and scribe are real Agents. After gather finishes we write an incomplete
checkpoint (simulate crash), then resume so gather is skipped and scribe runs
on the prior Agent's output.

Requires ``GEMINI_API_KEY`` (or OpenAI / Azure) — see ``.env.example``.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, JsonFileCheckpointer, Workflow
from loomable.agent.context import RunContext
from loomable.flow.state import SharedState
from loomable.persist.checkpoint import Checkpoint

from _common import ROOT, make_provider

CKPT_DIR = ROOT / ".checkpoints_phase_b"
SESSION = "inc-88421-resume"


class _CountingProvider:
    """Wrap a live provider and count ``complete`` calls."""

    def __init__(self, inner, counter: dict[str, int], key: str) -> None:
        self._inner = inner
        self._counter = counter
        self._key = key

    async def complete(self, request):
        self._counter[self._key] += 1
        return await self._inner.complete(request)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


async def main() -> None:
    if CKPT_DIR.exists():
        shutil.rmtree(CKPT_DIR)
    CKPT_DIR.mkdir(parents=True)

    cp = JsonFileCheckpointer(str(CKPT_DIR))
    base = make_provider()

    gatherer = Agent(
        base,
        role="Gatherer",
        goal="Collect incident evidence from the email",
        instructions=(
            "Summarize the incident in 2-4 short lines. "
            "Start the first line with EVIDENCE: then key facts."
        ),
    )
    scribe = Agent(
        base,
        role="Scribe",
        goal="Turn evidence into an escalation packet",
        instructions=(
            "You receive the previous gatherer's output as your input. "
            "Write a short escalation packet. Start with PACKET: then the summary."
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
    print(f"    gather output={ (gather_result.output.text() or '')[:160]}")

    # Resume with fresh Agents; count model calls so we prove gather is skipped.
    calls = {"gather": 0, "scribe": 0}
    gather2 = Agent(
        _CountingProvider(make_provider(), calls, "gather"),
        role="Gatherer",
        goal="Collect incident evidence",
        instructions="Start with EVIDENCE:",
    )
    scribe2 = Agent(
        _CountingProvider(make_provider(), calls, "scribe"),
        role="Scribe",
        goal="Turn evidence into an escalation packet",
        instructions=(
            "You receive the previous gatherer's output. "
            "Start with PACKET: then summarize."
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

    assert calls["gather"] == 0, f"gather Agent should be skipped, got {calls}"
    assert calls["scribe"] >= 1, f"scribe Agent should run, got {calls}"
    assert result.metadata.get("resumed") is True
    assert "gather" in (result.metadata.get("skipped_nodes") or [])
    out = result.output.text() or ""
    assert out.strip(), out

    await wf2.clear_checkpoint()
    try:
        await wf2.arun("x", resume=True)
        raise AssertionError("expected resume=True to fail without checkpoint")
    except RuntimeError as exc:
        assert "no incomplete checkpoint" in str(exc).lower()

    print("[ok] Phase B kill/resume gate (live Agents)")
    print(f"    skipped={result.metadata.get('skipped_nodes')}")
    print(f"    output={out[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
