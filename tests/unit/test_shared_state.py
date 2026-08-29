"""Unit tests for loomable.flow.state.SharedState and reducers.

Validates:
- Default overwrite reducer (Req 7.3)
- Append reducer accumulates into a list (Req 7.2)
- Merge reducer shallow-merges dicts (Req 7.2)
- Snapshot/restore roundtrip produces equivalent state (Req 7.5)
- Schema validation rejects unknown keys and wrong types (optional typed schema)
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from loomable.flow.state import SharedState, append, merge, overwrite


# ---------------------------------------------------------------------------
# Tests: default overwrite reducer
# ---------------------------------------------------------------------------


class TestOverwriteReducer:
    """Req 7.3: Where a key has no configured Reducer, default is last-write-wins."""

    def test_first_write_stores_value(self):
        state = SharedState()
        state.write("x", 42)
        assert state.get("x") == 42

    def test_second_write_overwrites(self):
        state = SharedState()
        state.write("x", 1)
        state.write("x", 2)
        assert state.get("x") == 2

    def test_overwrite_with_different_types(self):
        state = SharedState()
        state.write("x", "hello")
        state.write("x", 99)
        assert state.get("x") == 99

    def test_get_missing_key_returns_none(self):
        state = SharedState()
        assert state.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Tests: append reducer
# ---------------------------------------------------------------------------


class TestAppendReducer:
    """Req 7.2: Append reducer accumulates values into a list."""

    def test_first_write_stores_value_directly(self):
        state = SharedState(reducers={"log": append})
        state.write("log", "first")
        assert state.get("log") == "first"

    def test_subsequent_writes_append_to_list(self):
        state = SharedState(reducers={"log": append})
        state.write("log", "first")
        state.write("log", "second")
        assert state.get("log") == ["first", "second"]

    def test_multiple_appends(self):
        state = SharedState(reducers={"log": append})
        state.write("log", "a")
        state.write("log", "b")
        state.write("log", "c")
        assert state.get("log") == ["a", "b", "c"]

    def test_append_does_not_affect_other_keys(self):
        state = SharedState(reducers={"log": append})
        state.write("log", "event1")
        state.write("other", "plain")
        assert state.get("log") == "event1"
        assert state.get("other") == "plain"

    def test_append_wraps_list_payload(self):
        state = SharedState(reducers={"log": append})
        state.write("log", "a")
        state.write("log", ["b"])
        assert state.get("log") == ["a", ["b"]]


class TestExtendReducer:
    def test_extend_concatenates_lists(self):
        from loomable.flow.state import extend

        state = SharedState(reducers={"items": extend})
        state.write("items", ["a"])
        state.write("items", ["b", "c"])
        assert state.get("items") == ["a", "b", "c"]

    def test_extend_wraps_scalars(self):
        from loomable.flow.state import extend

        state = SharedState(reducers={"items": extend})
        state.write("items", "a")
        state.write("items", "b")
        assert state.get("items") == ["a", "b"]


# ---------------------------------------------------------------------------
# Tests: merge reducer
# ---------------------------------------------------------------------------


class TestMergeReducer:
    """Req 7.2: Merge reducer shallow-merges dicts."""

    def test_first_write_stores_dict(self):
        state = SharedState(reducers={"config": merge})
        state.write("config", {"a": 1})
        assert state.get("config") == {"a": 1}

    def test_merge_combines_dicts(self):
        state = SharedState(reducers={"config": merge})
        state.write("config", {"a": 1})
        state.write("config", {"b": 2})
        assert state.get("config") == {"a": 1, "b": 2}

    def test_merge_overwrites_duplicate_keys(self):
        state = SharedState(reducers={"config": merge})
        state.write("config", {"a": 1, "b": 2})
        state.write("config", {"b": 99, "c": 3})
        assert state.get("config") == {"a": 1, "b": 99, "c": 3}

    def test_merge_on_none_existing(self):
        """First write with merge stores the incoming dict directly."""
        state = SharedState(reducers={"config": merge})
        state.write("config", {"x": 10})
        assert state.get("config") == {"x": 10}


# ---------------------------------------------------------------------------
# Tests: snapshot and restore
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    """Req 7.5: SharedState is serializable for checkpointing."""

    def test_snapshot_returns_current_data(self):
        state = SharedState()
        state.write("a", 1)
        state.write("b", [1, 2, 3])
        snap = state.snapshot()
        assert snap == {"a": 1, "b": [1, 2, 3]}

    def test_snapshot_is_a_deep_copy(self):
        state = SharedState()
        state.write("items", [1, 2])
        snap = state.snapshot()
        # Mutating the snapshot should not affect the state.
        snap["items"].append(3)
        assert state.get("items") == [1, 2]

    def test_restore_recreates_state(self):
        state = SharedState(reducers={"log": append})
        state.write("log", "a")
        state.write("log", "b")
        state.write("count", 42)

        snap = state.snapshot()
        restored = SharedState.restore(snap, reducers={"log": append})

        assert restored.get("log") == state.get("log")
        assert restored.get("count") == state.get("count")

    def test_restore_continues_with_reducers(self):
        """Restored state can continue writing with the same reducers."""
        state = SharedState(reducers={"log": append})
        state.write("log", "a")
        state.write("log", "b")

        snap = state.snapshot()
        restored = SharedState.restore(snap, reducers={"log": append})
        restored.write("log", "c")

        assert restored.get("log") == ["a", "b", "c"]

    def test_restore_data_is_independent_of_source(self):
        """Modifying the source dict after restore doesn't affect restored state."""
        data = {"key": [1, 2, 3]}
        restored = SharedState.restore(data)
        data["key"].append(4)
        assert restored.get("key") == [1, 2, 3]

    def test_empty_snapshot(self):
        state = SharedState()
        assert state.snapshot() == {}

    def test_roundtrip_preserves_nested_structures(self):
        state = SharedState()
        state.write("nested", {"a": {"b": [1, 2, {"c": 3}]}})
        snap = state.snapshot()
        restored = SharedState.restore(snap)
        assert restored.get("nested") == {"a": {"b": [1, 2, {"c": 3}]}}


# ---------------------------------------------------------------------------
# Tests: schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Optional typed schema: write validates keys/types and rejects unknown keys."""

    @dataclass
    class MySchema:
        count: int
        name: str
        items: list

    def test_write_allowed_key_succeeds(self):
        state = SharedState(schema=self.MySchema)
        state.write("count", 5)
        assert state.get("count") == 5

    def test_write_unknown_key_raises_key_error(self):
        state = SharedState(schema=self.MySchema)
        with pytest.raises(KeyError, match="does not allow key 'unknown'"):
            state.write("unknown", "value")

    def test_write_wrong_type_raises_type_error(self):
        state = SharedState(schema=self.MySchema)
        with pytest.raises(TypeError, match="expects type 'int'.*got 'str'"):
            state.write("count", "not an int")

    def test_write_correct_types_all_succeed(self):
        state = SharedState(schema=self.MySchema)
        state.write("count", 10)
        state.write("name", "hello")
        state.write("items", [1, 2, 3])
        assert state.get("count") == 10
        assert state.get("name") == "hello"
        assert state.get("items") == [1, 2, 3]

    def test_no_schema_allows_any_key(self):
        state = SharedState()
        state.write("anything_goes", "fine")
        assert state.get("anything_goes") == "fine"

    def test_schema_with_reducers(self):
        """Schema validation works alongside reducers."""
        state = SharedState(
            reducers={"items": append},
            schema=self.MySchema,
        )
        state.write("items", "a")
        state.write("items", "b")
        assert state.get("items") == ["a", "b"]

    def test_schema_rejects_unknown_even_with_no_reducers(self):
        state = SharedState(schema=self.MySchema)
        with pytest.raises(KeyError):
            state.write("surprise", 42)


# ---------------------------------------------------------------------------
# Tests: multiple reducers on different keys
# ---------------------------------------------------------------------------


class TestMixedReducers:
    """Different keys can have different reducers simultaneously."""

    def test_mixed_reducers(self):
        state = SharedState(reducers={"log": append, "config": merge})
        state.write("log", "event1")
        state.write("log", "event2")
        state.write("config", {"a": 1})
        state.write("config", {"b": 2})
        state.write("plain", "just a value")
        state.write("plain", "overwritten")

        assert state.get("log") == ["event1", "event2"]
        assert state.get("config") == {"a": 1, "b": 2}
        assert state.get("plain") == "overwritten"
