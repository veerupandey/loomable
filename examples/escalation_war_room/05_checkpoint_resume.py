"""Phase B gate — kill mid-workflow and resume from checkpoint.

Agents are the Workflow steps. The framework passes each Agent the previous
Agent's output — no parse helpers.

This exam still simulates a crash by writing an incomplete checkpoint after
the gather Agent finishes, then resumes so gather is skipped and scribe runs.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from loomable import Agent, JsonFileCheckpointer, Workflow
from loomable.agent.context import RunContext
from loomable.flow.state import SharedState
from loomable.persist.checkpoint import Checkpoint

from _common import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _offline import scripted_model  # noqa: E402

CKPT_DIR = ROOT / ".checkpoints_phase_b"
SESSION = "inc-88421-resume"


def _counting_agent(role: str, steps: list, counter: dict[str, int], key: str) -> Agent:
    """Agent whose model increments ``counter[key]`` each time it is invoked."""
    base = scripted_model(steps)
    inner = base.provider_impl

    class _Counting:
        async def complete(self, request):
            counter[key] += 1
            return await inner.complete(request)

    from loomable.agent import ModelSpec

    return Agent(
        ModelSpec(provider=f"count-{key}", provider_impl=_Counting()),
        role=role,
        use_llm_summarizer=False,
    )


async def main() -> None:
    if CKPT_DIR.exists():
        shutil.rmtree(CKPT_DIR)
    CKPT_DIR.mkdir(parents=True)

    cp = JsonFileCheckpointer(str(CKPT_DIR))

    # --- Run 1: gather Agent completes, then we "kill" before scribe ---
    gatherer = Agent(
        scripted_model(["EVIDENCE:SEV-1 email"]),
        role="Gatherer",
        goal="Collect incident evidence",
        use_llm_summarizer=False,
    )
    scribe = Agent(
        scripted_model([{"echo": "PACKET:{input}"}]),
        role="Scribe",
        goal="Turn evidence into an escalation packet",
        use_llm_summarizer=False,
    )
    wf1 = (
        Workflow("war-room-resume", session_id=SESSION, checkpointer=cp)
        .step("gather", gatherer)
        .step("scribe", scribe)
    )

    # Drive gather only, then write incomplete checkpoint (simulate process kill).
    state = SharedState()
    ctx = RunContext(shared_state=state)
    gather_result = await gatherer.arun("SEV-1 email", context=ctx)
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

    # --- Run 2: resume — gather skipped; scribe reads prior Agent output ---
    calls = {"gather": 0, "scribe": 0}
    gather2 = _counting_agent(
        "Gatherer",
        ["EVIDENCE:should-not-run"],
        calls,
        "gather",
    )
    scribe2 = _counting_agent(
        "Scribe",
        [{"echo": "PACKET:{input}"}],
        calls,
        "scribe",
    )
    wf2 = (
        Workflow("war-room-resume", session_id=SESSION, checkpointer=cp)
        .step("gather", gather2)
        .step("scribe", scribe2)
    )
    result = await wf2.arun("SEV-1 email", resume=True)

    assert calls["gather"] == 0, f"gather Agent should be skipped, got {calls}"
    assert calls["scribe"] == 1, f"scribe Agent should run once, got {calls}"
    assert result.metadata.get("resumed") is True
    assert "gather" in (result.metadata.get("skipped_nodes") or [])
    out = result.output.text() or ""
    assert "PACKET:" in out and "EVIDENCE:" in out, out

    # resume=True with no incomplete CP must fail
    await wf2.clear_checkpoint()
    try:
        await wf2.arun("x", resume=True)
        raise AssertionError("expected resume=True to fail without checkpoint")
    except RuntimeError as exc:
        assert "no incomplete checkpoint" in str(exc).lower()

    print("[ok] Phase B kill/resume gate (Agents as steps)")
    print(f"    skipped={result.metadata.get('skipped_nodes')}")
    print(f"    output={out[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
