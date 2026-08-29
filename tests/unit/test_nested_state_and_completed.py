"""Regression tests for nested SharedState and accurate completed_node_ids."""

from __future__ import annotations

import pytest

from loomable import Step, Workflow
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
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


class TestNestedSharedStatePreserved:
    @pytest.mark.asyncio
    async def test_parent_keys_survive_parallel(self):
        async def pre(inp, *, context=None):  # noqa: A002
            assert context is not None and context.shared_state is not None
            # Use a key that does not collide with the step node_id
            context.shared_state.write("phase", "kept")
            return _text("pre")

        async def p1(inp, *, context=None):  # noqa: A002
            # Parent key must still be visible inside the parallel group
            assert context is not None
            assert context.shared_state.get("phase") == "kept"
            return _text("p1")

        async def post(inp, *, context=None):  # noqa: A002
            phase = context.shared_state.get("phase") if context else None
            return _text(f"post:{phase}")

        wf = (
            Workflow("nest")
            .step("pre", pre)
            .parallel(Step("p1", p1))
            .step("post", post)
        )
        result = await wf.arun("x")
        assert result.output.text() == "post:kept"
        assert wf.state.get("phase") == "kept"
        assert wf.state.get("p1") is not None
        assert wf.state.get("pre") is not None  # node output still present


class TestCompletedNodeIdsAccurate:
    @pytest.mark.asyncio
    async def test_unselected_route_branches_not_marked_completed(self):
        cp = InMemoryCheckpointer()
        wf = Workflow("r", session_id="t1", checkpointer=cp).route(
            lambda i: "a",  # noqa: A002
            a=lambda i: "A",  # noqa: A002
            b=lambda i: "B",  # noqa: A002
        )
        await wf.arun("x")
        state = await wf.get_state()
        completed = state["completed"]
        assert any("a" in c for c in completed)
        assert not any(c.endswith("_b") or c.endswith("_route_0_b") for c in completed)
        # Explicitly: unselected branch id must be absent
        assert "_route_0_b" not in completed
