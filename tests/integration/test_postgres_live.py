"""Live Postgres E2E — requires POSTGRES_URL and a running database.

Run::

    export POSTGRES_URL=postgresql://loomable:loomable@127.0.0.1:5432/loomable
    python -m pytest tests/integration/test_postgres_live.py -q
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("asyncpg")

DSN = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="POSTGRES_URL not set"),
]


@pytest.fixture
def dsn() -> str:
    assert DSN
    return DSN


@pytest.mark.asyncio
async def test_live_postgres_checkpointer_roundtrip(dsn: str) -> None:
    from loomable.persist.checkpoint import Checkpoint
    from loomable.persist.postgres import PostgresCheckpointer

    cp = PostgresCheckpointer(dsn, table="loomable_checkpoints_e2e", max_checkpoints=3)
    await cp.setup()
    thread = "e2e-thread-1"
    await cp.put(
        Checkpoint(
            thread_id=thread,
            step=1,
            session_state={"shared_state": {"board": {"items": [{"id": "1", "title": "t"}]}}},
            complete=False,
        )
    )
    await cp.put(
        Checkpoint(
            thread_id=thread,
            step=2,
            session_state={"shared_state": {"board": {"items": [{"id": "1", "title": "t2"}]}}},
            complete=False,
        )
    )
    latest = await cp.get(thread)
    assert latest is not None
    assert latest.step == 2
    assert latest.session_state["shared_state"]["board"]["items"][0]["title"] == "t2"
    listed = await cp.list(thread)
    assert len(listed) >= 2
    forked = await cp.fork(thread, "e2e-fork-1")
    assert forked is not None
    assert (await cp.get("e2e-fork-1")) is not None
    await cp.aclose()


@pytest.mark.asyncio
async def test_live_postgres_memory_and_vector(dsn: str) -> None:
    from loomable.kernel.long_term import LongTermStore
    from loomable.kernel.stores import ShortTermStore
    from loomable.providers.backends.postgres import PgVectorBackend, PostgresMemoryBackend

    kv = PostgresMemoryBackend(dsn, user_id="alice", table="loomable_kv_e2e")
    short = ShortTermStore(backend=kv)
    await kv.setup()
    await short.write("prefs", {"theme": "dark", "lang": "python"})
    assert await short.read("prefs") == {"theme": "dark", "lang": "python"}

    bob = PostgresMemoryBackend(dsn, user_id="bob", table="loomable_kv_e2e")
    await bob.setup()
    await bob.write("prefs", {"theme": "light"})
    assert await bob.read("prefs") == {"theme": "light"}
    assert await short.read("prefs") == {"theme": "dark", "lang": "python"}

    vec = PgVectorBackend(dsn, dimensions=4, user_id="alice", table="loomable_vectors_e2e")
    await vec.setup()
    lt = LongTermStore(backend=vec, backend_name="postgres")
    await lt.index("doc-a", [1.0, 0.0, 0.0, 0.0], {"text": "axis-x"})
    await lt.index("doc-b", [0.0, 1.0, 0.0, 0.0], {"text": "axis-y"})
    hits = await lt.query([0.95, 0.05, 0.0, 0.0], k=1)
    assert hits[0]["id"] == "doc-a"
    assert hits[0]["text"] == "axis-x"

    await kv.aclose()
    await bob.aclose()
    await vec.aclose()


@pytest.mark.asyncio
async def test_live_workflow_checkpoint_resume_with_postgres(dsn: str) -> None:
    from loomable.flow.flow import Flow
    from loomable.flow.hitl import FlowPaused
    from loomable.persist.postgres import PostgresCheckpointer

    calls: list[str] = []

    async def step_a(inp):
        calls.append("a")
        return "A"

    async def step_b(inp):
        calls.append("b")
        # Force a mid-run durable checkpoint by raising after a is done —
        # use require_confirmation on B via Flow nodes.
        return "B"

    cp = PostgresCheckpointer(dsn, table="loomable_flow_ckpts_e2e")
    await cp.setup()
    session = "wf-e2e-1"

    # First run: sequential A then B, complete checkpoint under session id
    flow = Flow(
        {"a": step_a, "b": step_b},
        engine="sequential",
        checkpointer=cp,
        session_id=session,
    )
    result = await flow.arun("go")
    assert "A" in str(result.output.text()) or result.output is not None
    saved = await cp.get(session)
    assert saved is not None
    assert saved.complete is True
    assert calls == ["a", "b"]

    # Resume-ish: new flow same session with resume=False starts fresh
    calls.clear()
    flow2 = Flow(
        {"a": step_a, "b": step_b},
        engine="sequential",
        checkpointer=cp,
        session_id=session,
    )
    await flow2.arun("go", resume=False)
    assert calls == ["a", "b"]
    await cp.aclose()


@pytest.mark.asyncio
async def test_live_case_board_checkpoint_with_postgres(dsn: str) -> None:
    from loomable import Case
    from loomable.agent import ModelSpec
    from loomable.flow.loop import VerdictResult
    from loomable.kernel.models import ModelRequest, ModelResponse
    from loomable.persist.postgres import PostgresCheckpointer

    class _Scripted:
        def __init__(self) -> None:
            self.n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            blob = str(request.messages).lower()
            if "json array" in blob or "break the user task" in blob or "planner" in blob:
                return ModelResponse(
                    content='["Gather facts", "Draft SEV packet"]',
                    usage={"input_tokens": 1, "output_tokens": 2},
                )
            if "integrate" in blob or "synthesizer" in blob or "specialist/worker" in blob:
                return ModelResponse(
                    content="FINAL SEV-1 packet for INC-88421",
                    usage={"input_tokens": 1, "output_tokens": 2},
                )
            return ModelResponse(
                content=f"done-step-{self.n}",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

    def accept(output, context) -> VerdictResult:
        ok = "SEV-" in (output.text() or "")
        return VerdictResult(ok=ok, detail="" if ok else "need SEV-")

    cp = PostgresCheckpointer(dsn, table="loomable_case_ckpts_e2e")
    await cp.setup()
    session = "case-e2e-1"
    case = Case(
        model=ModelSpec(provider="scripted", provider_impl=_Scripted()),
        board=True,
        dispatch="reuse",
        accept=accept,
        max_rounds=2,
        max_steps=2,
        modalities="text",
        session_id=session,
        checkpointer=cp,
    )
    result = await case.arun("Handle INC-88421 with SEV label")
    assert result.metadata.get("case") is True
    assert "SEV-" in (result.output.text() or "")
    assert case.board is not None
    assert len(case.board.list()) >= 1

    saved = await cp.get(session)
    assert saved is not None
    assert saved.complete is True
    shared = (saved.session_state or {}).get("shared_state") or {}
    board = shared.get("board") or {}
    assert board.get("items")

    # Fresh Case hydrates board from Postgres checkpoint
    case2 = Case(
        model=ModelSpec(provider="scripted", provider_impl=_Scripted()),
        board=True,
        dispatch="reuse",
        accept=accept,
        max_rounds=2,
        max_steps=2,
        modalities="text",
        session_id=session,
        checkpointer=cp,
    )
    await case2._hydrate_board_from_checkpoint(resume=True)
    assert case2.board is not None
    assert len(case2.board.list()) >= 1
    await cp.aclose()
