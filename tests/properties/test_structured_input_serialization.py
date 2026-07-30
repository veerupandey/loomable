# Feature: agent-ergonomics, Property 1
"""Property 1: Structured input is serialized and passed through.

For any Pydantic model, dataclass, or dict input, the coerced AgentInput is a
single user message whose text is the JSON serialization of the value; strings
and AgentInput values pass through unchanged.

**Validates: Requirements 1.1, 1.2, 1.3**
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import pydantic
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.content import AgentInput, Message, Text, to_agent_input


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# JSON-safe primitives for building dicts/models
json_primitives = st.one_of(
    st.text(max_size=50),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)

# Recursive JSON-safe values (dicts, lists, primitives)
json_values = st.recursive(
    json_primitives,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
    ),
    max_leaves=20,
)

# Strategy: non-empty dict with string keys and JSON-safe values
json_dicts = st.dictionaries(
    st.text(min_size=1, max_size=20),
    json_values,
    min_size=1,
    max_size=10,
)

# Strategy: arbitrary strings (including empty)
arbitrary_strings = st.text(max_size=200)


def _extract_text(agent_input: AgentInput) -> str:
    """Extract the text content from a single-message AgentInput."""
    assert len(agent_input.messages) == 1
    msg = agent_input.messages[0]
    assert msg.role == "user"
    assert len(msg.parts) == 1
    part = msg.parts[0]
    assert part.data is not None
    return part.data.decode("utf-8")


# ---------------------------------------------------------------------------
# Property tests: Dict input
# ---------------------------------------------------------------------------


class TestDictInputSerialization:
    """Dict inputs are JSON-serialized into a single user message."""

    @settings(max_examples=100)
    @given(d=json_dicts)
    def test_dict_becomes_json_user_message(self, d: dict) -> None:
        result = to_agent_input(d)

        # Must be a single user message
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert len(result.messages[0].parts) == 1

        # Text content is the JSON serialization
        text = _extract_text(result)
        assert json.loads(text) == d


# ---------------------------------------------------------------------------
# Property tests: Pydantic model input
# ---------------------------------------------------------------------------


class TestPydanticModelSerialization:
    """Pydantic model inputs are JSON-serialized via model_dump_json."""

    @settings(max_examples=100)
    @given(
        name=st.text(min_size=1, max_size=30),
        age=st.integers(min_value=0, max_value=200),
        tags=st.lists(st.text(max_size=20), max_size=5),
    )
    def test_pydantic_model_becomes_json_user_message(
        self, name: str, age: int, tags: list[str]
    ) -> None:
        # Dynamically create a Pydantic model instance with the generated data
        class SampleModel(pydantic.BaseModel):
            name: str
            age: int
            tags: list[str] = []

        model = SampleModel(name=name, age=age, tags=tags)
        result = to_agent_input(model)

        # Must be a single user message
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert len(result.messages[0].parts) == 1

        # Text content matches model_dump_json()
        text = _extract_text(result)
        assert json.loads(text) == model.model_dump()


# ---------------------------------------------------------------------------
# Property tests: Dataclass input
# ---------------------------------------------------------------------------


@dataclass
class SampleDataclass:
    """A sample dataclass for testing serialization."""

    label: str
    value: int
    items: list[str] = field(default_factory=list)


class TestDataclassSerialization:
    """Dataclass inputs are JSON-serialized via json.dumps(asdict(...))."""

    @settings(max_examples=100)
    @given(
        label=st.text(min_size=1, max_size=30),
        value=st.integers(min_value=-(2**31), max_value=2**31),
        items=st.lists(st.text(max_size=20), max_size=5),
    )
    def test_dataclass_becomes_json_user_message(
        self, label: str, value: int, items: list[str]
    ) -> None:
        dc = SampleDataclass(label=label, value=value, items=items)
        result = to_agent_input(dc)

        # Must be a single user message
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert len(result.messages[0].parts) == 1

        # Text content matches json.dumps(asdict(dc))
        text = _extract_text(result)
        expected = dataclasses.asdict(dc)
        assert json.loads(text) == expected


# ---------------------------------------------------------------------------
# Property tests: Strings pass through unchanged
# ---------------------------------------------------------------------------


class TestStringPassthrough:
    """String inputs become a single user message with the exact string text."""

    @settings(max_examples=100)
    @given(s=arbitrary_strings)
    def test_string_becomes_user_message_unchanged(self, s: str) -> None:
        result = to_agent_input(s)

        # Must be a single user message
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert len(result.messages[0].parts) == 1

        # Text content is the exact original string
        text = _extract_text(result)
        assert text == s


# ---------------------------------------------------------------------------
# Property tests: AgentInput pass-through unchanged
# ---------------------------------------------------------------------------


class TestAgentInputPassthrough:
    """AgentInput values pass through to_agent_input unchanged (identity)."""

    @settings(max_examples=100)
    @given(text_content=st.text(min_size=1, max_size=100))
    def test_agent_input_passes_through_unchanged(self, text_content: str) -> None:
        original = AgentInput(
            messages=[Message(role="user", parts=[Text(text_content)])]
        )
        result = to_agent_input(original)

        # Must be the exact same object (identity)
        assert result is original
