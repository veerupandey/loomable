"""Pluggable Agent memory stores (L1/L2 session persistence).

Use with::

    from loomable.memory import open_session_store
    from loomable import Agent

    store = open_session_store("sqlite", path="sessions.db")
    # store = open_session_store("file", path="./.sessions")
    # store = open_session_store("postgres", url=POSTGRES_URL, user_id="alice")
    # store = open_session_store("memory")

    agent = Agent(model=..., session_id="chat-1", session_store=store)
    await agent.arun("hi")
    # later process / new Agent:
    agent2 = Agent(model=..., session_id="chat-1", session_store=store, resume=True)

Or pass a :class:`~loomable.kernel.contracts.MemoryBackend` directly::

    agent = Agent(
        model=...,
        session_id="chat-1",
        memory_backend=PostgresMemoryBackend(DSN, user_id="alice"),
    )
    # later: same memory_backend + resume=True
"""

from __future__ import annotations

from typing import Any, Literal

from loomable.kernel.stores import (
    BackendSessionStore,
    FileSessionStore,
    InMemoryMemoryBackend,
    SessionStore,
    SessionStoreProtocol,
    ShortTermStore,
    SQLiteMemoryBackend,
)

StoreKind = Literal["sqlite", "file", "postgres", "memory"]

__all__ = [
    "StoreKind",
    "open_session_store",
    "SessionStore",
    "FileSessionStore",
    "BackendSessionStore",
    "SessionStoreProtocol",
    "ShortTermStore",
    "SQLiteMemoryBackend",
    "InMemoryMemoryBackend",
]


def open_session_store(
    kind: StoreKind | str = "sqlite",
    *,
    path: str | None = None,
    url: str | None = None,
    user_id: str | None = None,
    **kwargs: Any,
) -> SessionStoreProtocol:
    """Create an Agent L1/L2 session store.

    Parameters
    ----------
    kind:
        ``"sqlite"`` | ``"file"`` | ``"postgres"`` | ``"memory"``
    path:
        SQLite db path or file-store directory.
    url:
        Postgres DSN (``kind="postgres"``).
    user_id:
        Optional tenant scope for Postgres KV rows.
    """
    kind_l = str(kind).strip().lower()
    if kind_l == "sqlite":
        return SessionStore(path or kwargs.get("db_path", ":memory:"))
    if kind_l == "file":
        if not path:
            raise ValueError('open_session_store("file") requires path=')
        return FileSessionStore(path)
    if kind_l == "memory":
        return BackendSessionStore(InMemoryMemoryBackend())
    if kind_l in ("postgres", "postgresql", "pg"):
        if not url:
            raise ValueError('open_session_store("postgres") requires url=')
        from loomable.providers.backends.postgres import PostgresMemoryBackend

        backend = PostgresMemoryBackend(url, user_id=user_id, **{
            k: v for k, v in kwargs.items() if k in ("table", "schema", "min_size", "max_size")
        })
        # Ensure table exists before first Agent save/resume
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(backend.setup())
        else:
            from loomable.kernel.stores import _run_sync

            _run_sync(backend.setup())
        return BackendSessionStore(backend)
    raise ValueError(
        f"Unknown session store kind {kind!r}. "
        'Use "sqlite", "file", "postgres", or "memory".'
    )
