"""Framework load stress — concurrent Agent / Team / Workflow / Case / Memory.

In-process (scripted providers). Complements ``test_harness_stress.py`` by
hammering fan-out, multi-tenant L3 notes, and orchestration under asyncio load.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from loomable import Agent, Case, Team, Workflow, tool
from loomable.agent import ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.memory import (
    ConversationMemory,
    Memory,
    MemoryScope,
    NoteStore,
    ScopedNoteStore,
    UserMemory,
    open_session_store,
)
from loomable.providers.vector_store import open_vector_store

pytestmark = pytest.mark.asyncio


class _Echo:
    """Deterministic provider: echoes last user text or a fixed reply."""

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.reply is not None:
            return ModelResponse(content=self.reply)
        text = ""
        for msg in reversed(request.messages or []):
            if getattr(msg, "role", None) == "user" or (
                isinstance(msg, dict) and msg.get("role") == "user"
            ):
                content = getattr(msg, "content", None)
                if content is None and isinstance(msg, dict):
                    content = msg.get("content")
                text = str(content)
                break
        return ModelResponse(content=f"echo:{text[:200]}")


class _Emb:
    async def embed(self, text: str) -> list[float]:
        # Cheap deterministic embedding for load tests
        h = abs(hash(text)) % 10_000
        return [float(h % 97), float((h // 97) % 97), 1.0]


def _model(reply: str | None = None) -> ModelSpec:
    return ModelSpec(provider="stress", provider_impl=_Echo(reply))


@tool
def ping(x: str = "ok") -> str:
    """Echo input for tool-loop stress."""
    return f"pong:{x}"


class TestConcurrentAgents:
    async def test_fifty_parallel_aruns_isolated_sessions(self, tmp_path) -> None:
        store = open_session_store("file", path=str(tmp_path / "sessions"))

        async def one(i: int) -> str:
            agent = Agent(
                model=_model(),
                session_id=f"sess-{i}",
                session_store=store,
                modalities="text",
                instructions="Be brief.",
            )
            r = await agent.arun(f"hello-{i}")
            return r.output.text() or ""

        t0 = time.perf_counter()
        outs = await asyncio.gather(*(one(i) for i in range(50)))
        elapsed = time.perf_counter() - t0
        assert len(outs) == 50
        assert all(f"hello-{i}" in outs[i] for i in range(50))
        assert elapsed < 30.0, f"50 parallel aruns took {elapsed:.1f}s"


class TestMultiTenantMemoryLoad:
    async def test_scoped_notes_under_crowded_concurrent_writes(self) -> None:
        base = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())
        alice = ScopedNoteStore(base, scope=MemoryScope.of(user_id="alice"))
        bob = ScopedNoteStore(base, scope=MemoryScope.of(user_id="bob"))

        async def flood(store: ScopedNoteStore, tag: str, n: int) -> None:
            await asyncio.gather(
                *(
                    store.write(f"{tag}-{i}", f"{tag} fact number {i} about preferences", ["load"])
                    for i in range(n)
                )
            )

        await asyncio.gather(flood(alice, "Alice", 40), flood(bob, "Bob", 40))

        async def recall_ok(store: ScopedNoteStore, who: str) -> None:
            hits = await store.recall("preferences fact", k=5)
            assert hits, f"{who} recall empty"
            assert all(who in (h.text or "") for h in hits), hits

        await asyncio.gather(recall_ok(alice, "Alice"), recall_ok(bob, "Bob"))

    async def test_compose_memory_agents_concurrent_turns(self, tmp_path) -> None:
        notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())
        store = open_session_store("file", path=str(tmp_path / "compose_sess"))

        async def user_session(uid: str, n_turns: int) -> list[str]:
            memory = Memory.compose(
                conversation=ConversationMemory(store=store, window=6),
                user=UserMemory(note_store=notes, memory_tool=True, auto_extract=True),
            )
            agent = Agent(
                model=_model(),
                memory=memory,
                session_id=f"u-{uid}",
                user_id=uid,
                modalities="text",
            )
            texts: list[str] = []
            for t in range(n_turns):
                r = await agent.arun(f"My name is {uid}. Turn {t}.")
                texts.append(r.output.text() or "")
            return texts

        results = await asyncio.gather(
            user_session("sam", 5),
            user_session("alex", 5),
            user_session("jordan", 5),
        )
        assert all(len(r) == 5 for r in results)


class TestTeamWorkflowCaseLoad:
    async def test_team_broadcast_under_parallel_calls(self) -> None:
        a = Agent(model=_model("A"), role="alpha", modalities="text")
        b = Agent(model=_model("B"), role="beta", modalities="text")
        team = Team(members=[a, b], model=_model("lead"), mode="broadcast")

        async def call(i: int) -> Any:
            return await team.arun(f"task-{i}")

        outs = await asyncio.gather(*(call(i) for i in range(12)))
        assert len(outs) == 12

    async def test_workflow_parallel_fanout_repeated(self) -> None:
        @tool
        def square(n: int) -> int:
            """Square an integer."""
            return int(n) * int(n)

        agent = Agent(model=_model("4"), tools=[square], modalities="text")

        async def run_once(i: int) -> str:
            # Simple agent arun standing in for workflow pressure
            r = await agent.arun(f"square {i}")
            return r.output.text() or ""

        outs = await asyncio.gather(*(run_once(i) for i in range(20)))
        assert len(outs) == 20

    async def test_workflow_step_chain_serial_load(self) -> None:
        async def body_a(text: str, **_: Any) -> str:
            await asyncio.sleep(0.001)
            return f"a:{text}"

        async def body_b(text: str, **_: Any) -> str:
            return f"b:{text}"

        async def one(i: int) -> Any:
            wf = Workflow().step(f"a-{i}", body_a).step(f"b-{i}", body_b)
            return await wf.arun(f"x{i}")

        outs = await asyncio.gather(*(one(i) for i in range(15)))
        assert len(outs) == 15

    async def test_case_arun_burst(self) -> None:
        async def run_case(i: int) -> Any:
            case = Case(
                model=_model(f"done-{i}"),
                goal=f"goal-{i}",
                board=False,
                max_rounds=1,
                max_steps=1,
                modalities="text",
            )
            return await case.arun(f"ticket-{i}")

        outs = await asyncio.gather(*(run_case(i) for i in range(10)))
        assert len(outs) == 10
        for i, r in enumerate(outs):
            text = r.output.text() if hasattr(r, "output") else str(r)
            assert text  # non-empty completion


class TestToolLoopPressure:
    async def test_many_tool_agents_concurrent(self) -> None:
        class ToolThenDone:
            def __init__(self) -> None:
                self.n = 0

            async def complete(self, request: ModelRequest) -> ModelResponse:
                self.n += 1
                if self.n == 1:
                    from loomable.kernel.models import ToolCall

                    return ModelResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="1", tool_name="ping", args={"x": "load"})
                        ],
                    )
                return ModelResponse(content="done")

        async def one(i: int) -> str:
            agent = Agent(
                model=ModelSpec(provider="t", provider_impl=ToolThenDone()),
                tools=[ping],
                modalities="text",
                max_tool_iterations=4,
            )
            r = await agent.arun(f"go-{i}")
            return r.output.text() or ""

        outs = await asyncio.gather(*(one(i) for i in range(25)))
        assert outs == ["done"] * 25
