import pytest

from loomable import Agent
from loomable.kernel.errors import SessionNotFoundError
from loomable.kernel.models import Session
from loomable.kernel.stores import SessionStore


def test_session_cache_keeps_identity_of_inflight_reference():
    agent = Agent("openai:gpt-5.4-mini", max_cached_sessions=1)
    held = agent._for_session("first")
    agent._for_session("second")
    assert list(agent._session_agents) == ["second"]
    assert agent._for_session("first") is held
    assert list(agent._session_agents) == ["first"]


def test_default_session_store_retains_recently_saved_sessions():
    store = SessionStore(max_sessions=2)
    try:
        for name in ("first", "second", "first", "third"):
            store.save(Session(session_id=name, agent_config_ref="openai"))
        with pytest.raises(SessionNotFoundError):
            store.resume("second")
        assert store.resume("first").session_id == "first"
        assert store.resume("third").session_id == "third"
    finally:
        store.close()
