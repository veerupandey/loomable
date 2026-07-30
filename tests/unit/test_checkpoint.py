"""Tests for the Checkpointer protocol, JsonFileCheckpointer, and SQLiteCheckpointer."""

from __future__ import annotations

import os
import tempfile

import pytest

from loomable.persist.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    Checkpointer,
    JsonFileCheckpointer,
    PendingAction,
    SQLiteCheckpointer,
)


# ---------------------------------------------------------------------------
# Shared test suite (run against both providers)
# ---------------------------------------------------------------------------


class CheckpointerContractTests:
    """Shared tests that every Checkpointer implementation must pass."""

    async def test_put_and_get_latest(self, checkpointer):
        cp1 = Checkpoint(thread_id="t1", step=1, session_state={"turns": []})
        cp2 = Checkpoint(thread_id="t1", step=2, session_state={"turns": ["hello"]})
        await checkpointer.put(cp1)
        await checkpointer.put(cp2)

        latest = await checkpointer.get("t1")
        assert latest is not None
        assert latest.step == 2
        assert latest.session_state == {"turns": ["hello"]}

    async def test_get_nonexistent_returns_none(self, checkpointer):
        result = await checkpointer.get("nonexistent")
        assert result is None

    async def test_list_returns_commit_order(self, checkpointer):
        for i in range(5):
            cp = Checkpoint(thread_id="t1", step=i, session_state={"step": i})
            await checkpointer.put(cp)

        cps = await checkpointer.list("t1")
        assert len(cps) == 5
        assert [cp.step for cp in cps] == [0, 1, 2, 3, 4]

    async def test_separate_threads_are_isolated(self, checkpointer):
        await checkpointer.put(Checkpoint(thread_id="t1", step=1, session_state={"a": 1}))
        await checkpointer.put(Checkpoint(thread_id="t2", step=99, session_state={"b": 2}))

        t1 = await checkpointer.get("t1")
        t2 = await checkpointer.get("t2")
        assert t1.step == 1
        assert t2.step == 99

    async def test_stream_text_roundtrip(self, checkpointer):
        cp = Checkpoint(
            thread_id="t1", step=3, session_state={},
            stream_text="Hello world so far", complete=False,
        )
        await checkpointer.put(cp)
        retrieved = await checkpointer.get("t1")
        assert retrieved.stream_text == "Hello world so far"
        assert retrieved.complete is False

    async def test_usage_roundtrip(self, checkpointer):
        cp = Checkpoint(
            thread_id="t1", step=1, session_state={},
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        await checkpointer.put(cp)
        retrieved = await checkpointer.get("t1")
        assert retrieved.usage == {"input_tokens": 100, "output_tokens": 50}

    async def test_pending_actions_roundtrip(self, checkpointer):
        """PendingAction (durable HITL) should survive put/get."""
        cp = Checkpoint(
            thread_id="t1", step=5, session_state={},
            pending=[
                PendingAction(tool_name="send_email", call_id="c1",
                              args={"to": "x@y.com"}, status="pending"),
                PendingAction(tool_name="delete_record", call_id="c2",
                              args={"id": 42}, status="approved"),
            ],
        )
        await checkpointer.put(cp)
        retrieved = await checkpointer.get("t1")
        assert len(retrieved.pending) == 2
        assert retrieved.pending[0].tool_name == "send_email"
        assert retrieved.pending[0].status == "pending"
        assert retrieved.pending[1].status == "approved"
        assert retrieved.pending[1].args == {"id": 42}

    async def test_event_kind_roundtrip(self, checkpointer):
        cp = Checkpoint(
            thread_id="t1", step=1, session_state={},
            event_kind="tool_call",
        )
        await checkpointer.put(cp)
        retrieved = await checkpointer.get("t1")
        assert retrieved.event_kind == "tool_call"

    async def test_protocol_compliance(self, checkpointer):
        assert isinstance(checkpointer, Checkpointer)


# ---------------------------------------------------------------------------
# JsonFileCheckpointer tests
# ---------------------------------------------------------------------------


class TestJsonFileCheckpointer(CheckpointerContractTests):

    @pytest.fixture
    def checkpointer(self, tmp_path):
        return JsonFileCheckpointer(location=str(tmp_path / "checkpoints"))

    async def test_files_are_human_readable(self, tmp_path):
        """Checkpoint files should be valid JSON readable by any editor."""
        import json

        ckpt = JsonFileCheckpointer(location=str(tmp_path / "ck"))
        cp = Checkpoint(thread_id="t1", step=1, session_state={"key": "value"})
        await ckpt.put(cp)

        thread_dir = os.path.join(str(tmp_path / "ck"), "t1")
        files = os.listdir(thread_dir)
        assert len(files) == 1
        assert files[0].endswith(".json")

        with open(os.path.join(thread_dir, files[0])) as f:
            data = json.load(f)
        assert data["session_state"] == {"key": "value"}
        assert data["step"] == 1

    async def test_pruning(self, tmp_path):
        """max_checkpoints should prune oldest files."""
        ckpt = JsonFileCheckpointer(location=str(tmp_path / "ck"), max_checkpoints=3)
        for i in range(5):
            await ckpt.put(Checkpoint(thread_id="t1", step=i, session_state={}))

        cps = await ckpt.list("t1")
        assert len(cps) == 3
        # Should retain the 3 most recent
        assert cps[-1].step == 4

    async def test_fork(self, tmp_path):
        """fork() should copy a checkpoint into a new thread."""
        ckpt = JsonFileCheckpointer(location=str(tmp_path / "ck"))
        await ckpt.put(Checkpoint(
            thread_id="original", step=10,
            session_state={"history": ["msg1", "msg2"]},
            pending=[PendingAction(tool_name="deploy", call_id="c1", args={})],
        ))

        forked = await ckpt.fork("original", "branch-1")
        assert forked is not None
        assert forked.thread_id == "branch-1"
        assert forked.step == 10
        assert forked.session_state == {"history": ["msg1", "msg2"]}
        assert forked.event_kind == "fork"
        assert len(forked.pending) == 1

        # Should be independently retrievable
        retrieved = await ckpt.get("branch-1")
        assert retrieved.thread_id == "branch-1"

    async def test_fork_nonexistent_returns_none(self, tmp_path):
        ckpt = JsonFileCheckpointer(location=str(tmp_path / "ck"))
        result = await ckpt.fork("nonexistent", "new")
        assert result is None


# ---------------------------------------------------------------------------
# SQLiteCheckpointer tests
# ---------------------------------------------------------------------------


class TestSQLiteCheckpointer(CheckpointerContractTests):

    @pytest.fixture
    def checkpointer(self):
        return SQLiteCheckpointer(":memory:")

    async def test_pruning(self):
        ckpt = SQLiteCheckpointer(":memory:", max_checkpoints=3)
        for i in range(5):
            await ckpt.put(Checkpoint(thread_id="t1", step=i, session_state={}))

        cps = await ckpt.list("t1")
        assert len(cps) == 3
        assert cps[-1].step == 4

    async def test_fork(self):
        ckpt = SQLiteCheckpointer(":memory:")
        await ckpt.put(Checkpoint(
            thread_id="original", step=7,
            session_state={"data": "important"},
        ))

        forked = await ckpt.fork("original", "alt-timeline")
        assert forked is not None
        assert forked.thread_id == "alt-timeline"
        assert forked.step == 7
        assert forked.event_kind == "fork"

        retrieved = await ckpt.get("alt-timeline")
        assert retrieved.session_state == {"data": "important"}


# ---------------------------------------------------------------------------
# Checkpoint dataclass tests
# ---------------------------------------------------------------------------


class TestCheckpointDataclass:

    def test_defaults(self):
        cp = Checkpoint(thread_id="t1", step=0)
        assert cp.session_state == {}
        assert cp.stream_text == ""
        assert cp.usage == {}
        assert cp.complete is True
        assert cp.pending == []
        assert cp.event_kind == ""
        assert cp.lineage_id == ""
        assert cp.timestamp > 0

    def test_roundtrip_via_dict(self):
        cp = Checkpoint(
            thread_id="t1", step=5,
            session_state={"k": "v"},
            pending=[PendingAction(tool_name="x", call_id="c1", args={"a": 1})],
            event_kind="model_call",
            lineage_id="lin-1",
        )
        data = cp.to_dict()
        restored = Checkpoint.from_dict(data)
        assert restored.thread_id == cp.thread_id
        assert restored.step == cp.step
        assert restored.pending[0].tool_name == "x"
        assert restored.event_kind == "model_call"
        assert restored.lineage_id == "lin-1"


# ---------------------------------------------------------------------------
# CheckpointConfig tests
# ---------------------------------------------------------------------------


class TestCheckpointConfig:

    def test_defaults(self):
        cfg = CheckpointConfig()
        assert cfg.on_events == ["run_end"]
        assert cfg.max_checkpoints is None
        assert cfg.location == ".checkpoints"

    def test_custom_events(self):
        cfg = CheckpointConfig(on_events=["model_call", "tool_call"], max_checkpoints=10)
        assert "model_call" in cfg.on_events
        assert cfg.max_checkpoints == 10


# ---------------------------------------------------------------------------
# PendingAction tests (durable HITL)
# ---------------------------------------------------------------------------


class TestPendingAction:

    def test_defaults(self):
        p = PendingAction(tool_name="deploy", call_id="c1")
        assert p.args == {}
        assert p.status == "pending"

    def test_status_transitions(self):
        p = PendingAction(tool_name="deploy", call_id="c1", status="approved")
        assert p.status == "approved"
