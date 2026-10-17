"""Restart and uncertain side-effect behavior for durable agent runs."""
import asyncio
import os
import sys
from pathlib import Path

import pytest

from loomable import Agent, SQLiteCheckpointer, tool
from loomable.agent import AgentPaused, DurabilityOptions, UnknownToolOutcome
from loomable.kernel.models import ModelResponse, ToolCall, ToolResult


async def exercise(path, mode, run_id=None):
    side_effects = Path(path).with_suffix(".calls")

    @tool
    async def change(value: str):
        with side_effects.open("a") as out:
            out.write(value + "\n")
        if mode == "crash":
            os._exit(23)
        return value

    class Model:
        async def complete(self, request):
            if any(m["role"] == "tool" for m in request.messages):
                return ModelResponse(content="done")
            return ModelResponse(tool_calls=[ToolCall("effect", "change", {"value": "once"})])

    agent = Agent(Model(), tools=[change], session_id="session",
                  require_confirmation=["change"],
                  durability=DurabilityOptions(SQLiteCheckpointer(path)))
    if mode == "pause":
        with pytest.raises(AgentPaused) as paused:
            await agent.arun("Run the tool")
        return paused.value.run_id
    if mode == "crash":
        await agent.acontinue_run(run_id, approvals={"effect": True})
    else:
        with pytest.raises(UnknownToolOutcome, match="effect"):
            await agent.acontinue_run(run_id)
        result = await agent.acontinue_run(run_id, outcomes={"effect": ToolResult(content="once")})
        assert result.text == "done"
        assert side_effects.read_text().splitlines() == ["once"]


async def test_approval_and_reconciliation_across_process_restart(tmp_path):
    path = str(tmp_path / "journal.sqlite")
    run_id = await exercise(path, "pause")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tests.integration.test_durable_agent", path, run_id,
    )
    assert await asyncio.wait_for(proc.wait(), 10) == 23
    await exercise(path, "reconcile", run_id)


if __name__ == "__main__":
    asyncio.run(exercise(sys.argv[1], "crash", sys.argv[2]))


async def test_completed_tool_is_replayed_after_model_failure(tmp_path):
    calls = []
    run_ids = []

    @tool
    def change():
        calls.append("written")
        return "written"

    class Model:
        def __init__(self, fail):
            self.fail = fail

        async def complete(self, request):
            if any(m["role"] == "tool" for m in request.messages):
                if self.fail:
                    raise RuntimeError("connection lost")
                return ModelResponse(content="done")
            return ModelResponse(tool_calls=[ToolCall("write", "change")])

    def agent(fail):
        return Agent(Model(fail), tools=[change], session_id="replay",
                     durability=DurabilityOptions(SQLiteCheckpointer(str(tmp_path / "replay.sqlite"))))

    original = agent(True)
    original._get_built().events = type("Capture", (), {
        "emit": lambda self, event: run_ids.append(event.attributes["run_id"])
        if event.kind == "run_start" else None
    })()
    with pytest.raises(RuntimeError, match="connection lost"):
        await original.arun("Run the tool")
    result = await agent(False).acontinue_run(run_ids[0])
    assert result.text == "done"
    assert calls == ["written"]
