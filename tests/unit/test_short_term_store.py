"""Unit tests for ShortTermStore and SQLiteMemoryBackend."""

from __future__ import annotations

import pytest

from loomable.kernel.contracts import MemoryBackend
from loomable.kernel.errors import MemoryBackendError
from loomable.kernel.stores import ShortTermStore, SQLiteMemoryBackend


# ---------------------------------------------------------------------------
# SQLiteMemoryBackend satisfies the MemoryBackend protocol
# ---------------------------------------------------------------------------


class TestSQLiteMemoryBackendProtocol:
    """Verify SQLiteMemoryBackend structurally satisfies MemoryBackend."""

    def test_is_memory_backend(self) -> None:
        backend = SQLiteMemoryBackend()
        assert isinstance(backend, MemoryBackend)


# ---------------------------------------------------------------------------
# SQLiteMemoryBackend basic operations
# ---------------------------------------------------------------------------


class TestSQLiteMemoryBackend:
    @pytest.fixture
    def backend(self) -> SQLiteMemoryBackend:
        return SQLiteMemoryBackend(":memory:")

    @pytest.mark.asyncio
    async def test_write_and_read_string(self, backend: SQLiteMemoryBackend) -> None:
        await backend.write("greeting", "hello")
        result = await backend.read("greeting")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_write_and_read_dict(self, backend: SQLiteMemoryBackend) -> None:
        data = {"name": "Alice", "age": 30}
        await backend.write("user", data)
        result = await backend.read("user")
        assert result == data

    @pytest.mark.asyncio
    async def test_write_and_read_list(self, backend: SQLiteMemoryBackend) -> None:
        data = [1, 2, 3, "four"]
        await backend.write("items", data)
        result = await backend.read("items")
        assert result == data

    @pytest.mark.asyncio
    async def test_read_missing_key_returns_none(
        self, backend: SQLiteMemoryBackend
    ) -> None:
        result = await backend.read("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_exists_true(self, backend: SQLiteMemoryBackend) -> None:
        await backend.write("key", "value")
        assert await backend.exists("key") is True

    @pytest.mark.asyncio
    async def test_exists_false(self, backend: SQLiteMemoryBackend) -> None:
        assert await backend.exists("missing") is False

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, backend: SQLiteMemoryBackend) -> None:
        await backend.write("key", "value")
        await backend.delete("key")
        assert await backend.exists("key") is False
        assert await backend.read("key") is None

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(
        self, backend: SQLiteMemoryBackend
    ) -> None:
        await backend.write("key", "v1")
        await backend.write("key", "v2")
        result = await backend.read("key")
        assert result == "v2"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key_no_error(
        self, backend: SQLiteMemoryBackend
    ) -> None:
        # Should not raise
        await backend.delete("missing")


# ---------------------------------------------------------------------------
# ShortTermStore with default (SQLite) backend
# ---------------------------------------------------------------------------


class TestShortTermStoreDefault:
    """ShortTermStore with default SQLite backend."""

    @pytest.fixture
    def store(self) -> ShortTermStore:
        return ShortTermStore()

    def test_default_backend_is_sqlite(self, store: ShortTermStore) -> None:
        assert isinstance(store.backend, SQLiteMemoryBackend)

    @pytest.mark.asyncio
    async def test_write_persists_and_read_returns(
        self, store: ShortTermStore
    ) -> None:
        await store.write("session:1", {"turns": ["hi", "hello"]})
        result = await store.read("session:1")
        assert result == {"turns": ["hi", "hello"]}

    @pytest.mark.asyncio
    async def test_exists_reflects_writes(self, store: ShortTermStore) -> None:
        assert await store.exists("key") is False
        await store.write("key", 42)
        assert await store.exists("key") is True

    @pytest.mark.asyncio
    async def test_delete_removes(self, store: ShortTermStore) -> None:
        await store.write("key", "val")
        await store.delete("key")
        assert await store.exists("key") is False


# ---------------------------------------------------------------------------
# ShortTermStore with custom backend (no agent changes)
# ---------------------------------------------------------------------------


class InMemoryBackend:
    """A trivial in-memory backend satisfying MemoryBackend protocol."""

    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    async def read(self, key: str) -> object:
        return self._data.get(key)

    async def write(self, key: str, value: object) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._data


class TestShortTermStorePluggable:
    """Demonstrate alternative backends require no agent changes."""

    @pytest.fixture
    def store(self) -> ShortTermStore:
        return ShortTermStore(backend=InMemoryBackend())

    @pytest.mark.asyncio
    async def test_custom_backend_write_read(self, store: ShortTermStore) -> None:
        await store.write("x", [1, 2, 3])
        assert await store.read("x") == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_custom_backend_is_not_sqlite(self, store: ShortTermStore) -> None:
        assert not isinstance(store.backend, SQLiteMemoryBackend)
        assert isinstance(store.backend, MemoryBackend)


# ---------------------------------------------------------------------------
# Unavailable backend returns MemoryBackendError naming the backend
# ---------------------------------------------------------------------------


class BrokenBackend:
    """A backend that always raises MemoryBackendError."""

    def __init__(self, backend_id: str = "broken-db") -> None:
        self._backend_id = backend_id

    async def read(self, key: str) -> object:
        raise MemoryBackendError(self._backend_id)

    async def write(self, key: str, value: object) -> None:
        raise MemoryBackendError(self._backend_id)

    async def delete(self, key: str) -> None:
        raise MemoryBackendError(self._backend_id)

    async def exists(self, key: str) -> bool:
        raise MemoryBackendError(self._backend_id)


class TestShortTermStoreUnavailableBackend:
    """MemoryBackendError names the backend when unavailable."""

    @pytest.fixture
    def store(self) -> ShortTermStore:
        return ShortTermStore(backend=BrokenBackend("postgres:prod"))

    @pytest.mark.asyncio
    async def test_read_raises_naming_backend(self, store: ShortTermStore) -> None:
        with pytest.raises(MemoryBackendError) as exc_info:
            await store.read("key")
        assert exc_info.value.backend_id == "postgres:prod"

    @pytest.mark.asyncio
    async def test_write_raises_naming_backend(self, store: ShortTermStore) -> None:
        with pytest.raises(MemoryBackendError) as exc_info:
            await store.write("key", "val")
        assert exc_info.value.backend_id == "postgres:prod"

    @pytest.mark.asyncio
    async def test_exists_raises_naming_backend(self, store: ShortTermStore) -> None:
        with pytest.raises(MemoryBackendError) as exc_info:
            await store.exists("key")
        assert exc_info.value.backend_id == "postgres:prod"

    @pytest.mark.asyncio
    async def test_delete_raises_naming_backend(self, store: ShortTermStore) -> None:
        with pytest.raises(MemoryBackendError) as exc_info:
            await store.delete("key")
        assert exc_info.value.backend_id == "postgres:prod"
