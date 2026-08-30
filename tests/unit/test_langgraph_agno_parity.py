"""LangGraph / Agno replacement parity tests.

Covers control-plane APIs needed to replace LangGraph + Agno workflows:
- Workflow.route N-way (Agno Router)
- Command(goto/update) from choosers
- get_state / update_state / list_states
- Workflow(reducers=) for parallel joins
"""

from __future__ import annotations

import pytest

from loomable import Command, Route, Step, Workflow
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.state import extend
from loomable.persist.checkpoint import InMemoryCheckpointer


def _text(s: str) -> RunResult:
    return RunResult(
        output=AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=s.encode("utf-8"),
                )
            ]
        ),
        session_id="",
    )


class TestWorkflowRoute:
    @pytest.mark.asyncio
    async def test_n_way_route_by_name(self):
        def choose(inp):  # noqa: A002
            return str(inp)

        async def quick(inp, *, context=None):  # noqa: A002
            return _text("QUICK")

        async def full(inp, *, context=None):  # noqa: A002
            return _text("FULL")

        async def human(inp, *, context=None):  # noqa: A002
            return _text("HUMAN")

        wf = Workflow("review").route(
            choose, quick=quick, full=full, human=human
        )
        result = await wf.arun("full")
        assert "FULL" in result.output.text()
        decision = wf.state.get("_route_decision")
        assert decision["selected"] == "full"

    @pytest.mark.asyncio
    async def test_route_with_command_goto_and_update(self):
        def choose(inp):  # noqa: A002
            return Command(goto="full", update={"severity": "high", "ticket": inp})

        async def quick(inp, *, context=None):  # noqa: A002
            return _text("QUICK")

        async def full(inp, *, context=None):  # noqa: A002
            sev = context.shared_state.get("severity") if context else None
            return _text(f"FULL:{sev}")

        wf = Workflow("review").route(choose, quick=quick, full=full)
        result = await wf.arun("INC-1")
        assert "FULL:high" in result.output.text()
        assert wf.state.get("severity") == "high"
        assert wf.state.get("ticket") == "INC-1"


class TestCommandStandalone:
    @pytest.mark.asyncio
    async def test_function_runnable_command_metadata(self):
        from loomable.flow.runnable import FunctionRunnable

        async def fn(inp):  # noqa: A002
            return Command(goto="b", update={"x": 1})

        result = await FunctionRunnable(fn).arun("a")
        assert result.metadata["selection"] == "b"
        assert result.metadata["state_updates"]["x"] == 1
        cmd = Command.from_metadata(result.metadata)
        assert cmd is not None
        assert cmd.goto == "b"


class TestControlPlane:
    @pytest.mark.asyncio
    async def test_get_state_after_run(self):
        async def a(inp, *, context=None):  # noqa: A002
            if context and context.shared_state:
                context.shared_state.write("note", "hello")
            return _text("A")

        wf = Workflow("s").step("a", a)
        await wf.arun("x")
        state = await wf.get_state()
        assert "note" in state["values"] or state["values"].get("a") is not None

    @pytest.mark.asyncio
    async def test_update_state_then_resume(self):
        cp = InMemoryCheckpointer()

        async def a(inp, *, context=None):  # noqa: A002
            return _text("A")

        async def b(inp, *, context=None):  # noqa: A002
            flag = context.shared_state.get("flag") if context else None
            return _text(f"B:{flag}")

        wf = Workflow(
            "pipe", session_id="t1", checkpointer=cp
        ).step("a", a).step("b", b)

        # Seed incomplete checkpoint as if "a" finished
        from loomable.persist.checkpoint import Checkpoint
        from loomable.content import AgentOutput, MediaPart, Modality

        await cp.put(
            Checkpoint(
                thread_id="t1",
                step=1,
                session_state={
                    "shared_state": {
                        "a": {
                            "__type__": "AgentOutput",
                            "parts": [
                                {
                                    "__type__": "MediaPart",
                                    "modality": "text",
                                    "media_type": "text/plain",
                                    "data_b64": "QQ==",  # "A"
                                    "uri": None,
                                }
                            ],
                        }
                    },
                    "completed_node_ids": ["a"],
                },
                complete=False,
            )
        )

        await wf.update_state({"flag": "patched"})
        state = await wf.get_state()
        assert state["values"]["flag"] == "patched"
        assert "a" in state["completed"]
        assert state["complete"] is False

        result = await wf.arun("x", resume=True)
        assert "B:patched" in result.output.text()

    @pytest.mark.asyncio
    async def test_list_states(self):
        cp = InMemoryCheckpointer()
        wf = Workflow("s", session_id="hist", checkpointer=cp).step(
            "a", lambda i: "ok"
        )
        await wf.arun("x")
        states = await wf.list_states()
        assert isinstance(states, list)
        assert len(states) >= 1

    @pytest.mark.asyncio
    async def test_fork_session(self):
        cp = InMemoryCheckpointer()
        wf = Workflow("s", session_id="src", checkpointer=cp).step(
            "a", lambda i: "ok"
        )
        await wf.arun("x")
        forked = await wf.fork_session("dst")
        assert forked["thread_id"] == "dst"
        assert wf._session_id == "dst"
        dst = await cp.get("dst")
        assert dst is not None
        src = await cp.get("src")
        assert src is not None


class TestWorkflowReducers:
    @pytest.mark.asyncio
    async def test_extend_reducer_on_parallel_writes(self):
        async def left_u(inp, *, context=None):  # noqa: A002
            return RunResult(
                output=_text("L").output,
                session_id="",
                metadata={"state_updates": {"items": ["L"]}},
            )

        async def right_u(inp, *, context=None):  # noqa: A002
            return RunResult(
                output=_text("R").output,
                session_id="",
                metadata={"state_updates": {"items": ["R"]}},
            )

        async def join(inp, *, context=None):  # noqa: A002
            items = None
            if context and context.shared_state:
                items = context.shared_state.get("items")
            return _text(f"JOIN:{items}")

        wf = (
            Workflow("p", reducers={"items": extend})
            .parallel(Step("left", left_u), Step("right", right_u))
            .step("join", join)
        )
        result = await wf.arun("x")
        text = result.output.text()
        assert "L" in text and "R" in text
        # Must be flat — no nested lists, no double-apply from parent engine
        assert wf.state.get("items") == ["L", "R"]

    @pytest.mark.asyncio
    async def test_parallel_does_not_double_apply_state_updates(self):
        async def left(inp, *, context=None):  # noqa: A002
            return RunResult(
                output=_text("L").output,
                session_id="",
                metadata={"state_updates": {"n": 1}},
            )

        async def right(inp, *, context=None):  # noqa: A002
            return RunResult(
                output=_text("R").output,
                session_id="",
                metadata={"state_updates": {"n": 2}},
            )

        from loomable.flow.parallel_group import Parallel_Group

        pg = Parallel_Group(Step("left", left), Step("right", right))
        result = await pg.arun("x")
        assert "state_updates" not in (result.metadata or {})


class TestRouteComposable:
    def test_route_requires_choices(self):
        with pytest.raises(ValueError, match="choice"):
            Route(lambda i: "a", {})
