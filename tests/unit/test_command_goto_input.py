"""Command.goto must not poison the next step's input."""

from __future__ import annotations

import pytest

from loomable import Workflow
from loomable.flow.command import Command


@pytest.mark.asyncio
async def test_command_goto_does_not_poison_next_step_input() -> None:
    seen: list[str] = []

    def chooser(_: str) -> Command:
        return Command(goto="branch_b", update={"flag": True})

    async def branch_b(inp) -> str:
        from loomable.content import AgentOutput

        if isinstance(inp, AgentOutput):
            seen.append(inp.text())
        else:
            seen.append(str(inp))
        return "done"

    wf = Workflow("goto-chain").step("choose", chooser).step("branch_b", branch_b)
    await wf.arun("start")
    assert seen == [""]
    assert "branch_b" not in seen[0]
