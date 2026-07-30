"""SharedState and Reducer primitives.

SharedState is a typed key/value store threaded through a Flow run. Each key
has a Reducer that merges concurrent writes. Supports snapshot/restore for
checkpointing.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, get_type_hints

__all__ = [
    "Reducer",
    "SharedState",
    "overwrite",
    "append",
    "merge",
]

# A Reducer takes (existing_value, incoming_value) and returns the merged result.
Reducer = Callable[[Any, Any], Any]


# ---------------------------------------------------------------------------
# Built-in reducers
# ---------------------------------------------------------------------------


def overwrite(existing: Any, incoming: Any) -> Any:
    """Default reducer: last-write-wins (Req 7.3)."""
    return incoming


def append(existing: Any, incoming: Any) -> Any:
    """Append reducer: accumulates values into a list."""
    base = existing if isinstance(existing, list) else ([] if existing is None else [existing])
    return base + [incoming]


def merge(existing: Any, incoming: Any) -> Any:
    """Merge reducer: shallow-merges dicts."""
    base = existing if isinstance(existing, dict) else ({} if existing is None else {})
    return {**base, **incoming}


# ---------------------------------------------------------------------------
# SharedState
# ---------------------------------------------------------------------------


class SharedState:
    """Typed key/value store with per-key reducers for deterministic merging.

    Parameters
    ----------
    reducers:
        Mapping of key names to Reducer functions. Keys not listed here
        use the ``overwrite`` reducer by default.
    schema:
        An optional type (dataclass or TypedDict) whose annotations define
        the allowed keys and their expected types. When supplied, ``write``
        rejects unknown keys and validates value types, turning silent data
        bugs into loud errors at the write site.
    """

    def __init__(
        self,
        reducers: dict[str, Reducer] | None = None,
        schema: type | None = None,
    ) -> None:
        self._data: dict[str, Any] = {}
        self._reducers: dict[str, Reducer] = dict(reducers) if reducers else {}
        self._schema = schema
        # Pre-compute allowed keys and their types from the schema if provided.
        self._schema_hints: dict[str, Any] | None = None
        if schema is not None:
            self._schema_hints = get_type_hints(schema)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any:
        """Read the current value for *key*, returning ``None`` if unset."""
        return self._data.get(key)

    def write(self, key: str, value: Any) -> None:
        """Write *value* to *key*, applying the key's reducer.

        If a schema is configured, validates the key is allowed. Type
        validation is performed only when no custom reducer is configured
        for the key (since reducers transform values and intermediate
        writes may not match the declared final type).
        """
        reducer = self._reducers.get(key, overwrite)
        has_custom_reducer = key in self._reducers

        self._validate_schema(key, value, skip_type_check=has_custom_reducer)

        existing = self._data.get(key)

        if key in self._data:
            self._data[key] = reducer(existing, value)
        else:
            # First write — no existing value to merge; store directly.
            self._data[key] = value

    def snapshot(self) -> dict:
        """Return a serializable deep copy of the current state.

        The returned dict can be persisted for checkpointing and later
        passed to ``restore()`` to recreate the SharedState.
        """
        return copy.deepcopy(self._data)

    @classmethod
    def restore(
        cls,
        data: dict,
        reducers: dict[str, Reducer] | None = None,
        schema: type | None = None,
    ) -> "SharedState":
        """Recreate a SharedState from a previously snapshotted dict.

        Parameters
        ----------
        data:
            The dict returned by a prior ``snapshot()`` call.
        reducers:
            Same reducer configuration as the original state.
        schema:
            Same schema as the original state.
        """
        state = cls(reducers=reducers, schema=schema)
        state._data = copy.deepcopy(data)
        return state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_schema(self, key: str, value: Any, *, skip_type_check: bool = False) -> None:
        """Validate key and value against the schema if one is configured."""
        if self._schema_hints is None:
            return

        # Reject unknown keys.
        if key not in self._schema_hints:
            allowed = sorted(self._schema_hints.keys())
            raise KeyError(
                f"SharedState schema does not allow key {key!r}. "
                f"Allowed keys: {allowed}"
            )

        if skip_type_check:
            return

        # Validate value type (best-effort for simple types).
        expected_type = self._schema_hints[key]
        # Skip validation for complex typing constructs (Optional, Union, etc.)
        if isinstance(expected_type, type) and not isinstance(value, expected_type):
            raise TypeError(
                f"SharedState schema expects type {expected_type.__name__!r} "
                f"for key {key!r}, got {type(value).__name__!r}"
            )

    def __repr__(self) -> str:
        keys = list(self._data.keys())
        return f"SharedState(keys={keys})"
