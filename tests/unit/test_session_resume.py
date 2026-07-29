"""Unit tests for high-level persistent memory and session resume (task 9.4).

Covers Requirement 15:
- 15.1/15.3: an agent created with ``resume=True`` and an existing session id restores
  the persisted state so prior turns are available.
- 15.2: running an agent with a session id persists conversational + session state
  after the run via the kernel SessionStore.
- 15.4: resuming an unknown session id raises ``SessionNotFoundError`` naming the id.
- default (``resume=False``) creates a fresh session.
"""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.kernel.errors import SessionNotFoundError
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.kernel.stores import SessionStore


class EchoProvider:
    """A fake provider that echoes the first user text part back as content."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        text = ""
        for message in request.messages:
            if message["role"] == "user":
                for part in message["content"]:
                    if part.get("type") == "text":
                        text = part["text"]
                        break
                if text:
                    break
        return ModelResponse(
            content=f"echo: {text}",
            usage={"input_tokens": 1, "output_tokens": 1},
        )


def _agent(store: SessionStore, *, session_id: str, resume: bool = False) -> Agent:
    """A text-in / text-out agent sharing the given session store."""
    return Agent(
        model=ModelSpec(provider="echo", provider_impl=EchoProvider()),
        session_id=session_id,
        resume=resume,
        session_store=store,
    )


async def test_run_persists_state_and_resume_restores_prior_turns() -> None:
    """A resumed agent restores prior turns from a shared store (Req 15.1/15.2/15.3)."""
    store = SessionStore()  # single in-memory db shared across both agents
    first = _agent(store, session_id="sess-1")

    await first.arun("hello")
    built_first = first._get_built()
    # After the run, state was persisted: a user + assistant turn and step advanced.
    assert len(built_first.session.l1) == 2
    assert built_first.session.step == 1

    # A SECOND agent sharing the SAME store resumes the same session id.
    second = _agent(store, session_id="sess-1", resume=True)
    built_second = second._get_built()

    # Prior turns are restored (Req 15.3).
    assert built_second.session.step == 1
    assert len(built_second.session.l1) == 2
    assert built_second.session.l1[0].role == "user"
    assert built_second.session.l1[0].content == "hello"
    assert built_second.session.l1[1].role == "assistant"
    assert built_second.session.l1[1].content == "echo: hello"

    # Running the resumed agent continues to accumulate state (Req 15.2).
    await second.arun("again")
    assert built_second.session.step == 2
    assert len(built_second.session.l1) == 4


async def test_resume_unknown_session_id_raises_not_found() -> None:
    """Resuming an unknown session id raises SessionNotFoundError naming it (Req 15.4)."""
    store = SessionStore()
    agent = _agent(store, session_id="does-not-exist", resume=True)

    with pytest.raises(SessionNotFoundError) as exc_info:
        agent.build()
    assert exc_info.value.session_id == "does-not-exist"


async def test_default_creates_fresh_session() -> None:
    """Without resume, a fresh session is created even for a known id (Req 15.1)."""
    store = SessionStore()
    first = _agent(store, session_id="sess-fresh")
    await first.arun("hello")

    # A new agent with resume=False starts fresh: no prior turns, step at 0.
    second = _agent(store, session_id="sess-fresh", resume=False)
    built_second = second._get_built()
    assert built_second.session.l1 == []
    assert built_second.session.step == 0
