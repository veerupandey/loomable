"""Unit tests for SessionStore with SQLite persistence and resume."""

import pytest

from loomable.kernel.errors import SessionNotFoundError
from loomable.kernel.models import Session, StructuredSummary, Turn
from loomable.kernel.stores import SessionStore


class TestSessionStoreSaveAndResume:
    """Tests for SessionStore.save() and resume()."""

    def test_save_and_resume_empty_session(self) -> None:
        """A session with no turns or summaries round-trips correctly."""
        store = SessionStore()
        session = Session(
            session_id="sess-001",
            agent_config_ref="config-ref-abc",
            l1=[],
            l2=[],
            step=0,
        )
        store.save(session)
        restored = store.resume("sess-001")

        assert restored.session_id == "sess-001"
        assert restored.agent_config_ref == "config-ref-abc"
        assert restored.l1 == []
        assert restored.l2 == []
        assert restored.step == 0

    def test_save_and_resume_with_turns(self) -> None:
        """A session with L1 turns round-trips correctly."""
        store = SessionStore()
        turns = [
            Turn(role="user", content="Hello", tokens=5, step=1),
            Turn(role="assistant", content="Hi there!", tokens=8, step=2),
        ]
        session = Session(
            session_id="sess-002",
            agent_config_ref="my-config",
            l1=turns,
            l2=[],
            step=2,
        )
        store.save(session)
        restored = store.resume("sess-002")

        assert len(restored.l1) == 2
        assert restored.l1[0].role == "user"
        assert restored.l1[0].content == "Hello"
        assert restored.l1[0].tokens == 5
        assert restored.l1[0].step == 1
        assert restored.l1[1].role == "assistant"
        assert restored.l1[1].content == "Hi there!"
        assert restored.l1[1].tokens == 8
        assert restored.l1[1].step == 2
        assert restored.step == 2

    def test_save_and_resume_with_summaries(self) -> None:
        """A session with L2 summaries round-trips correctly."""
        store = SessionStore()
        summaries = [
            StructuredSummary(
                covers_steps=range(0, 5),
                objectives=["Accomplish task A"],
                decisions=["Use approach B"],
                text="Summary of steps 0-4",
                tokens=20,
            ),
        ]
        session = Session(
            session_id="sess-003",
            agent_config_ref="cfg-x",
            l1=[],
            l2=summaries,
            step=5,
        )
        store.save(session)
        restored = store.resume("sess-003")

        assert len(restored.l2) == 1
        s = restored.l2[0]
        assert s.covers_steps == range(0, 5)
        assert s.objectives == ["Accomplish task A"]
        assert s.decisions == ["Use approach B"]
        assert s.text == "Summary of steps 0-4"
        assert s.tokens == 20

    def test_save_overwrites_existing_session(self) -> None:
        """Saving a session with the same id overwrites the previous state."""
        store = SessionStore()
        session_v1 = Session(
            session_id="sess-overwrite",
            agent_config_ref="v1",
            l1=[],
            l2=[],
            step=1,
        )
        store.save(session_v1)

        session_v2 = Session(
            session_id="sess-overwrite",
            agent_config_ref="v2",
            l1=[Turn(role="user", content="updated", tokens=3, step=2)],
            l2=[],
            step=2,
        )
        store.save(session_v2)
        restored = store.resume("sess-overwrite")

        assert restored.agent_config_ref == "v2"
        assert restored.step == 2
        assert len(restored.l1) == 1
        assert restored.l1[0].content == "updated"

    def test_resume_unknown_id_raises_session_not_found_error(self) -> None:
        """Resuming an unknown session id raises SessionNotFoundError naming the id."""
        store = SessionStore()
        with pytest.raises(SessionNotFoundError) as exc_info:
            store.resume("nonexistent-id")

        assert exc_info.value.session_id == "nonexistent-id"
        assert "nonexistent-id" in str(exc_info.value)

    def test_multiple_sessions_independent(self) -> None:
        """Multiple sessions can be stored and resumed independently."""
        store = SessionStore()
        session_a = Session(
            session_id="a", agent_config_ref="cfg-a", l1=[], l2=[], step=10
        )
        session_b = Session(
            session_id="b", agent_config_ref="cfg-b", l1=[], l2=[], step=20
        )
        store.save(session_a)
        store.save(session_b)

        assert store.resume("a").step == 10
        assert store.resume("b").step == 20

    def test_default_uses_sqlite_without_extra_config(self) -> None:
        """SessionStore uses SQLite by default (in-memory) without extra config."""
        store = SessionStore()
        # Verify internal connection is a sqlite3 connection
        assert store._conn is not None
        session = Session(
            session_id="default-test",
            agent_config_ref="ref",
            l1=[],
            l2=[],
            step=0,
        )
        store.save(session)
        restored = store.resume("default-test")
        assert restored.session_id == "default-test"
