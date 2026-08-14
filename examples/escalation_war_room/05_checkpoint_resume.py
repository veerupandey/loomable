"""Phase B gate — kill mid-workflow and resume from checkpoint.

Simulates a crash after the gather step: incomplete checkpoint is written,
process "dies", then a new Workflow with the same session_id + checkpointer
resumes and skips gather.

NOTE: This exam uses tiny callables only to force an incomplete checkpoint.
For normal apps, put ``Agent``s on ``Workflow.step(...)`` — Agents already
consume the previous Agent's output (see ``examples/advanced/03_checkpointing.py``
and ``examples/patterns/02_pipeline.py``).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from loomable import InMemoryCheckpointer, Step, Workflow
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, Text
from loomable.persist.checkpoint import Checkpoint

from _common import ROOT

CKPT_DIR = ROOT / ".checkpoints_phase_b"
SESSION = "inc-88421-resume"


async def gather(inp, *, context=None):
    return RunResult(
        output=AgentOutput(parts=[Text(f"EVIDENCE:{inp}")]),
        session_id=SESSION,
    )


async def scribe(inp, *, context=None):
    if hasattr(inp, "text") and callable(inp.text):
        text = inp.text()
    else:
        text = str(inp)
    return RunResult(
        output=AgentOutput(parts=[Text(f"PACKET:{text}")]),
        session_id=SESSION,
        structured={"ok": True, "from": text},
    )


async def main() -> None:
    if CKPT_DIR.exists():
        shutil.rmtree(CKPT_DIR)
    CKPT_DIR.mkdir(parents=True)

    from loomable.persist import JsonFileCheckpointer

    cp = JsonFileCheckpointer(str(CKPT_DIR))

    # --- Run 1: execute only gather, then "crash" by writing incomplete CP ---
    wf1 = (
        Workflow("war-room-resume", session_id=SESSION, checkpointer=cp)
        .step("gather", gather)
        .step("scribe", scribe)
    )
    # Manually drive gather then write incomplete checkpoint (simulate kill)
    from loomable.agent.context import RunContext
    from loomable.flow.state import SharedState

    flow = wf1.flow
    state = SharedState()
    ctx = RunContext(shared_state=state)
    gather_result = await gather("SEV-1 email")
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
    print("[kill] incomplete checkpoint after gather")

    # --- Run 2: resume — gather must be skipped ---
    calls = {"gather": 0, "scribe": 0}

    async def gather2(inp, *, context=None):
        calls["gather"] += 1
        return await gather(inp, context=context)

    async def scribe2(inp, *, context=None):
        calls["scribe"] += 1
        return await scribe(inp, context=context)

    wf2 = (
        Workflow("war-room-resume", session_id=SESSION, checkpointer=cp)
        .step("gather", gather2)
        .step("scribe", scribe2)
    )
    result = await wf2.arun("SEV-1 email", resume=True)

    assert calls["gather"] == 0, f"gather should be skipped, got {calls}"
    assert calls["scribe"] == 1, f"scribe should run once, got {calls}"
    assert result.metadata.get("resumed") is True
    assert "gather" in (result.metadata.get("skipped_nodes") or [])
    out = result.output.text()
    assert "PACKET:" in out and "EVIDENCE:" in out, out
    assert result.structured and result.structured.get("ok") is True

    # resume=True with no incomplete CP must fail
    await wf2.clear_checkpoint()
    try:
        await wf2.arun("x", resume=True)
        raise AssertionError("expected resume=True to fail without checkpoint")
    except RuntimeError as exc:
        assert "no incomplete checkpoint" in str(exc).lower()

    print("[ok] Phase B kill/resume gate")
    print(f"    skipped={result.metadata.get('skipped_nodes')}")
    print(f"    output={result.output.text()[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
