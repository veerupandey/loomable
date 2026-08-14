"""Live E2E: Agent / Team / Case / Flow / FastAPI memory parity on Postgres.

Requires::

    export POSTGRES_URL=postgresql://loomable:loomable@127.0.0.1:5432/loomable
    python -m pytest tests/integration/test_memory_parity_live.py -q
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("asyncpg")

DSN = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="POSTGRES_URL not set"),
]


class _Echo:
    async def complete(self, request):  # noqa: ANN001
        from loomable.kernel.models import ModelResponse

        blob = str(request.messages)
        if "Alex" in blob and "What is my name" in blob:
            return ModelResponse(content="Your name is Alex.")
        if "teal" in blob.lower() and ("color" in blob.lower() or "prefer" in blob.lower()):
            return ModelResponse(content="Got it — teal noted.")
        if "favorite color" in blob.lower() or "what color" in blob.lower():
            if "teal" in blob.lower():
                return ModelResponse(content="Your favorite color is teal.")
            return ModelResponse(content="I don't know your color.")
        return ModelResponse(content="ack")


class _Emb:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0, 0.0]


def _model():
    from loomable.agent import ModelSpec

    return ModelSpec(provider="scripted", provider_impl=_Echo())


@pytest.fixture
def dsn() -> str:
    assert DSN
    return DSN


@pytest.mark.asyncio
async def test_e2e_agent_postgres_l1_plus_zvec_l3(dsn: str) -> None:
    from loomable.agent import Agent, NoteStore
    from loomable.kernel.long_term import LongTermStore, open_vector_store
    from loomable.memory import open_session_store

    sid = f"e2e-agent-{uuid.uuid4().hex[:8]}"
    store = open_session_store("postgres", url=dsn, user_id="e2e-alice")
    notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())

    a1 = Agent(
        model=_model(),
        session_id=sid,
        session_store=store,
        note_store=notes,
        memory_tool=True,
        modalities="text",
    )
    await a1.arun("My name is Alex")
    await notes.write("who", "User name is Alex", tags=["identity"])

    # New Agent instance — Postgres L1/L2 + same-process zvec L3
    a2 = Agent(
        model=_model(),
        session_id=sid,
        session_store=store,
        resume=True,
        note_store=notes,
        memory_tool=True,
        modalities="text",
    )
    r2 = await a2.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")
    recalled = await notes.recall("name", k=1)
    assert recalled and "Alex" in recalled[0].text


@pytest.mark.asyncio
async def test_e2e_agent_bind_session_http_style(dsn: str) -> None:
    from loomable.agent import Agent
    from loomable.memory import open_session_store

    store = open_session_store("postgres", url=dsn, user_id="e2e-bind")
    sid = f"e2e-bind-{uuid.uuid4().hex[:8]}"
    agent = Agent(
        model=_model(),
        session_id=sid,
        session_store=store,
        modalities="text",
    )
    await agent.arun("My name is Alex")

    # Simulate another HTTP request: mutate session via bind_session
    agent.bind_session(sid)
    assert agent._resume is True
    r2 = await agent.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")

    missing = f"e2e-missing-{uuid.uuid4().hex[:8]}"
    agent.bind_session(missing)
    assert agent._resume is False


@pytest.mark.asyncio
async def test_e2e_team_coordinator_postgres_memory(dsn: str) -> None:
    from loomable.agent import Agent, Team
    from loomable.memory import open_session_store

    store = open_session_store("postgres", url=dsn, user_id="e2e-team")
    sid = f"e2e-team-{uuid.uuid4().hex[:8]}"
    member = Agent(model=_model(), role="Worker", modalities="text")
    team = Team(
        [member],
        model=_model(),
        mode="coordinate",
        session_id=sid,
        session_store=store,
        memory_window=8,
    )
    assert team.agent._session_store is store
    await team.agent.arun("My name is Alex")

    team.bind_session(sid)
    r2 = await team.agent.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")


@pytest.mark.asyncio
async def test_e2e_case_copies_memory_and_checkpointer(dsn: str) -> None:
    from loomable.agent import Agent, NoteStore
    from loomable.case import Case
    from loomable.flow.loop import VerdictResult
    from loomable.kernel.long_term import LongTermStore, open_vector_store
    from loomable.kernel.models import ModelRequest, ModelResponse
    from loomable.memory import open_session_store
    from loomable.persist.postgres import PostgresCheckpointer

    class _CaseScript:
        def __init__(self) -> None:
            self.n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            blob = str(request.messages).lower()
            if "json array" in blob or "break the user task" in blob:
                return ModelResponse(content='["Gather facts", "Draft packet"]')
            if "integrate" in blob or "merge" in blob:
                return ModelResponse(content="FINAL SEV-1 packet ready")
            return ModelResponse(content=f"done-{self.n}")

    def accept(output, context) -> VerdictResult:  # noqa: ANN001
        ok = "SEV-" in (output.text() or "")
        return VerdictResult(ok=ok, detail="" if ok else "need SEV-")

    from loomable.agent import ModelSpec

    store = open_session_store("postgres", url=dsn, user_id="e2e-case")
    notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())
    cp = PostgresCheckpointer(dsn, table="loomable_case_mem_ckpts_e2e")
    await cp.setup()
    sid = f"e2e-case-{uuid.uuid4().hex[:8]}"

    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_CaseScript()),
        mode="case",
        session_id=sid,
        session_store=store,
        note_store=notes,
        memory_tool=True,
        checkpointer=cp,
        accept=accept,
        max_rounds=2,
        max_plan_steps=2,
        modalities="text",
        board=True,
    )
    case = Case.from_agent(agent)
    mem = case._kwargs.get("agent_memory") or {}
    assert mem.get("session_store") is store
    assert mem.get("note_store") is notes
    assert mem.get("memory_tool") is True
    assert case._kwargs.get("checkpointer") is cp

    result = await case.arun("Handle INC with SEV label")
    assert "SEV-" in (result.output.text() or "")
    saved = await cp.get(sid)
    assert saved is not None and saved.complete is True

    # L3 still shared on Case memory pack
    await notes.write("inc", "INC handled as SEV-1", tags=["ops"])
    assert (await notes.recall("SEV", k=1))[0].text.startswith("INC")
    await cp.aclose()


@pytest.mark.asyncio
async def test_e2e_flow_agent_step_keeps_postgres_session(dsn: str) -> None:
    from loomable.agent import Agent
    from loomable.agent.context import RunContext
    from loomable.flow.workflow import Workflow
    from loomable.memory import open_session_store

    store = open_session_store("postgres", url=dsn, user_id="e2e-flow")
    sid = f"e2e-flow-{uuid.uuid4().hex[:8]}"
    agent = Agent(
        model=_model(),
        session_id=sid,
        session_store=store,
        modalities="text",
    )
    seen: dict[str, object] = {}

    async def step(inp, *, context=None):  # noqa: ANN001
        seen["ctx"] = context
        return await agent.arun(inp, context=context)

    wf = Workflow("e2e-flow-mem")
    wf.step("talk", step)
    await wf.arun("My name is Alex")
    assert isinstance(seen.get("ctx"), RunContext)

    agent2 = Agent(
        model=_model(),
        session_id=sid,
        session_store=store,
        resume=True,
        modalities="text",
    )
    r2 = await agent2.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")


@pytest.mark.asyncio
async def test_e2e_fastapi_bind_session_postgres(dsn: str) -> None:
    """HTTP session_id must rebuild Agent L1/L2 from Postgres via bind_session."""
    from fastapi.testclient import TestClient

    from loomable.agent import Agent
    from loomable.memory import open_session_store
    from loomable.serve import FastAPIAdapter

    store = open_session_store("postgres", url=dsn, user_id="e2e-http")
    sid = f"e2e-http-{uuid.uuid4().hex[:8]}"
    agent = Agent(
        model=_model(),
        session_id=sid,
        session_store=store,
        modalities="text",
    )
    app = FastAPIAdapter(agent).app()
    client = TestClient(app)

    def _payload(text: str, session: str) -> dict:
        return {
            "session_id": session,
            "messages": [
                {"role": "user", "parts": [{"modality": "text", "text": text}]}
            ],
        }

    r1 = client.post("/run", json=_payload("My name is Alex", sid))
    assert r1.status_code == 200, r1.text

    # Fresh adapter would be a new process; here we bind same agent to same sid
    r2 = client.post("/run", json=_payload("What is my name?", sid))
    assert r2.status_code == 200, r2.text
    body = r2.json()
    texts = [p.get("text", "") for p in body.get("output", [])]
    assert any("Alex" in t for t in texts), body
