import json

import pytest

from loomable.kernel.errors import SessionNotFoundError
from loomable.kernel.models import Session
from loomable.kernel.stores import FileSessionStore, serialize_session
from loomable.persist.checkpoint import Checkpoint, JsonFileCheckpointer
from loomable.stream import AsyncStreamBus, StreamEvent


@pytest.mark.parametrize("thread_id", ["../outside", "/outside", "..", "nested/thread", "a\\b"])
async def test_checkpoint_rejects_path_components(tmp_path, thread_id):
    store = JsonFileCheckpointer(str(tmp_path / "checkpoints"))
    with pytest.raises(ValueError, match="single path component"):
        await store.put(Checkpoint(thread_id=thread_id, step=0))
    with pytest.raises(ValueError, match="single path component"):
        await store.get(thread_id)


async def test_checkpoint_rejects_symlink_directory(tmp_path):
    root = tmp_path / "checkpoints"
    store = JsonFileCheckpointer(str(root))
    (root / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        await store.put(Checkpoint(thread_id="alias", step=0))


def test_legacy_session_lookup_checks_identity(tmp_path):
    store = FileSessionStore(tmp_path)
    legacy = Session(session_id="user/a", agent_config_ref="test")
    (tmp_path / "user_a.json").write_text(json.dumps(serialize_session(legacy)))
    assert store.resume("user/a").session_id == "user/a"
    with pytest.raises(SessionNotFoundError):
        store.resume("user_a")
    store.save(Session(session_id="user_a", agent_config_ref="test"))
    assert store.resume("user_a").session_id == "user_a"
    assert store.resume("user/a").session_id == "user/a"


async def test_full_stream_drains_after_sync_close():
    bus = AsyncStreamBus(max_pending=1)
    event = StreamEvent(type="RUN_STARTED", run_id="r")
    bus.publish_nowait(event)
    bus.close_sync()
    assert [item async for item in bus.events()] == [event]
