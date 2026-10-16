"""Adversarial regressions using disposable data and no external services."""

import asyncio
import json
import os
import socket
import sqlite3

import pytest

from loomable.flow import Workflow
from loomable.media import Image
from loomable.media.types import MediaResolveError
from loomable.persist.checkpoint import Checkpoint, InMemoryCheckpointer, JsonFileCheckpointer
from loomable.sandbox.subprocess_backend import SubprocessSandbox
from loomable.toolkits.file_tools import FileTools
from loomable.toolkits.net_safety import pinned_http_target
from loomable.toolkits.sql_tools import SQLTools
from loomable.toolkits.url_tools import URLTools
from loomable.toolkits.workspace_tools import WorkspaceStore


@pytest.mark.asyncio
async def test_workspace_and_checkpoint_ignore_external_symlinks(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("PRIVATE_FIXTURE")
    root = tmp_path / "workspace"
    root.mkdir()
    try:
        (root / "link.txt").symlink_to(marker)
        (root / "directory").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")

    files = FileTools(base_dir=root)
    assert await files._grep_files("PRIVATE_FIXTURE") == "(no matches)"
    assert "link.txt" not in await files._glob_files()
    workspace = WorkspaceStore(root)
    assert workspace.read("link.txt") is None
    assert workspace.write("directory/new.txt", "blocked") is None
    assert not workspace.delete("directory/marker.txt")
    assert marker.read_text() == "PRIVATE_FIXTURE"
    assert not (outside / "new.txt").exists()

    checkpointer = JsonFileCheckpointer(str(tmp_path / "checkpoints"))
    fixture = outside / "checkpoint.json"
    fixture.write_text(json.dumps(Checkpoint(thread_id="other", step=1).to_dict()))
    thread_dir = tmp_path / "checkpoints" / "session"
    thread_dir.mkdir()
    (thread_dir / "checkpoint.json").symlink_to(fixture)
    assert await checkpointer.get("session") is None
    assert await checkpointer.list("session") == []


@pytest.mark.asyncio
async def test_sql_read_only_rejects_uri_fragment_and_external_writes(tmp_path):
    db = tmp_path / "db.sqlite"
    shadow = tmp_path / "db.sqlite#fragment"
    shadow.touch()
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sample (value TEXT)")
        conn.execute("INSERT INTO sample VALUES ('before')")
    tools = SQLTools(read_only=True)
    assert "Error:" in await tools._run_sql(str(shadow), "UPDATE sample SET value='after'")
    for statement in ("VACUUM INTO '{}'", "ATTACH DATABASE '{}' AS auxiliary"):
        target = tmp_path / "new.sqlite"
        assert "Error:" in await tools._run_sql(str(db), statement.format(target))
        assert not target.exists()
    assert "before" in await tools._run_sql(str(db), "SELECT value FROM sample")


def test_outbound_connections_are_pinned_and_media_rejects_loopback(monkeypatch, tmp_path):
    original = socket.getaddrinfo
    lookups = []

    def rebind(host, port, *args, **kwargs):
        lookups.append(host)
        address = "93.184.216.34" if len(lookups) == 1 else "127.0.0.1"
        return original(address, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", rebind)
    target, options = pinned_http_target("https://rebind.example:8443/image.png")
    assert target.host == "93.184.216.34"
    assert options["headers"]["Host"] == "rebind.example:8443"
    assert options["extensions"]["sni_hostname"] == "rebind.example"
    with pytest.raises(ValueError, match="blocked host"):
        pinned_http_target("https://rebind.example:8443/redirect")
    with pytest.raises(MediaResolveError, match="blocked host"):
        Image(url="http://127.0.0.1/image.png").save(tmp_path / "unused.png")


@pytest.mark.asyncio
async def test_url_tool_sends_pinned_request_and_blocks_private_redirect(monkeypatch):
    httpx = pytest.importorskip("httpx")
    client = httpx.AsyncClient
    original_resolve = socket.getaddrinfo
    requests = []

    def resolve(host, port, *args, **kwargs):
        return original_resolve("93.184.216.34", port, *args, **kwargs)

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/fixture"})

    def client_factory(**kwargs):
        assert kwargs["trust_env"] is False
        return client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    result = await URLTools()._fetch_url("https://public.example/document")
    assert "blocked host" in result
    request, = requests
    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == "public.example"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process group cleanup")
async def test_subprocess_timeout_kills_child(tmp_path):
    marker = tmp_path / "child-survived"
    child = f"import time; from pathlib import Path; time.sleep(.6); Path({str(marker)!r}).touch()"
    code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "print('child started', flush=True); time.sleep(10)"
    )
    result = await SubprocessSandbox(root=tmp_path).run_python(code, timeout=.3)
    assert result.timed_out
    await asyncio.sleep(.75)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_workflow_mount_serializes_session_changes():
    httpx = pytest.importorskip("httpx")
    fastapi = pytest.importorskip("fastapi")
    from loomable.serve import mount_workflow

    class BlockingCheckpointer(InMemoryCheckpointer):
        def __init__(self):
            super().__init__()
            self.block = False
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def get(self, thread_id):
            result = await super().get(thread_id)
            if self.block and thread_id == "alice":
                self.block = False
                self.entered.set()
                await self.release.wait()
            return result

    cp = BlockingCheckpointer()
    workflow = Workflow("audit", checkpointer=cp, session_id="alice").step(
        "capture", lambda value: {"private": value}
    )
    app = fastapi.FastAPI()
    mount_workflow(app, workflow, prefix="/wf", api_key="fixture-key")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "fixture-key"}
    ) as client:
        assert (await client.post("/wf/run", json={"input_text": "ALICE"})).status_code == 200
        cp.block = True
        patch = asyncio.create_task(client.patch("/wf/state", json={"values": {"patch": "ALICE_PATCH"}}))
        async with asyncio.timeout(5):
            await cp.entered.wait()
            bob = asyncio.create_task(client.post(
                "/wf/run", json={"input_text": "BOB", "session_id": "bob"}
            ))
            await asyncio.sleep(0)
            cp.release.set()
            assert (await patch).status_code == 200
            assert (await bob).status_code == 200
        alice_state = (await client.get("/wf/state")).json()
        bob_state = (await client.get("/wf/state", params={"session_id": "bob"})).json()
        empty_state = (await client.get("/wf/state", params={"session_id": "new"})).json()
        assert "ALICE_PATCH" in str(alice_state["values"])
        assert "ALICE" not in str(bob_state["values"])
        assert empty_state["values"] == {}
        assert (await client.post(
            "/wf/run", json={"input_text": "escape", "session_id": "../escape"}
        )).status_code == 422
