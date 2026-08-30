"""map_over must not write complete=True on parent workflow session."""

from __future__ import annotations

import pytest

from loomable import Workflow
from loomable.flow.send import Send
from loomable.persist.checkpoint import InMemoryCheckpointer


@pytest.mark.asyncio
async def test_map_over_does_not_complete_parent_checkpoint_mid_run() -> None:
    cp = InMemoryCheckpointer()
    session_id = "parent-thread"
    complete_flags: list[bool | None] = []

    async def worker(item: str) -> str:
        checkpoint = await cp.get(session_id)
        complete_flags.append(checkpoint.complete if checkpoint else None)
        return item.upper()

    wf = (
        Workflow("parent", session_id=session_id, checkpointer=cp)
        .map_over(worker, over="tasks")
        .step("after", lambda _: "done")
    )

    wf._ensure_compiled()
    from loomable.agent.context import RunContext
    from loomable.flow.state import SharedState

    ctx = RunContext()
    ctx.shared_state = SharedState()
    ctx.shared_state.write("tasks", [Send("w", "a"), Send("w", "b")])

    flow = wf._compiled_flow
    assert flow is not None
    await flow.arun("go", context=ctx)

    assert complete_flags, "worker should have run"
    assert all(flag is False for flag in complete_flags if flag is not None)
