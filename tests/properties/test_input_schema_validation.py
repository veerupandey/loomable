# Feature: agent-ergonomics, Property 2
"""Property 2: Input schema validation gates the run.

For any input and a configured input_schema, a dict/model that conforms SHALL be
accepted (validated into the schema), and a non-conforming dict/model SHALL raise
InputValidationError before any model call; a plain string SHALL bypass validation.

**Validates: Requirements 1.4, 1.5, 1.6**
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pydantic
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.agent.builder import BuiltAgent
from loomable.agent.errors import InputValidationError
from loomable.content import AgentInput, ModelCapabilities
from loomable.kernel.memory import MemoryManager
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.models import Session
from loomable.kernel.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Helpers: minimal BuiltAgent construction with mocked providers
# ---------------------------------------------------------------------------


def _make_built_agent(input_schema: type | None = None) -> BuiltAgent:
    """Create a minimal BuiltAgent with mocked internals for input validation testing.

    Providers/HTTP/MCP are mocked — we never hit any external service.
    """
    mock_provider = MagicMock()
    mock_provider.invoke = AsyncMock()

    model_interface = ModelInterface(
        providers={"mock": mock_provider},
        default_provider="mock",
    )

    return BuiltAgent(
        model_interface=model_interface,
        memory=MemoryManager(),
        tool_runtime=ToolRuntime(tools={}),
        session=Session(session_id="test", agent_config_ref="test"),
        capabilities=ModelCapabilities(),
        input_schema=input_schema,
    )


# ---------------------------------------------------------------------------
# Schemas used for testing
# ---------------------------------------------------------------------------


class PydanticSchema(pydantic.BaseModel):
    """A Pydantic model used as the input_schema for property tests."""

    name: str
    age: int
    active: bool = True


@dataclass
class DataclassSchema:
    """A dataclass used as the input_schema for property tests."""

    name: str
    age: int
    active: bool = True


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: generate dicts that conform to PydanticSchema
conforming_pydantic_dicts = st.fixed_dictionaries(
    {
        "name": st.text(min_size=1, max_size=50),
        "age": st.integers(min_value=0, max_value=200),
    },
    optional={"active": st.booleans()},
)

# Strategy: generate dicts that conform to DataclassSchema
conforming_dataclass_dicts = st.fixed_dictionaries(
    {
        "name": st.text(min_size=1, max_size=50),
        "age": st.integers(min_value=0, max_value=200),
    },
    optional={"active": st.booleans()},
)

# Strategy: generate dicts that do NOT conform to PydanticSchema
# Pydantic is very flexible with coercion (e.g. "0" -> 0), so we need values
# that truly cannot be coerced: missing required fields or un-coercible types.
non_conforming_pydantic_dicts = st.one_of(
    # Missing required field "name" (age present)
    st.fixed_dictionaries({"age": st.integers()}),
    # Missing required field "age" (name present)
    st.fixed_dictionaries({"name": st.text(min_size=1, max_size=20)}),
    # Wrong type for "age" (a list/dict instead of int — not coercible)
    st.fixed_dictionaries(
        {"name": st.text(min_size=1, max_size=20), "age": st.lists(st.integers(), min_size=1)}
    ),
    # Wrong type for "name" (a list instead of string — not coercible)
    st.fixed_dictionaries(
        {"name": st.lists(st.integers(), min_size=1), "age": st.integers()}
    ),
    # Empty dict (both required fields missing)
    st.just({}),
)

# Strategy: generate dicts that do NOT conform to DataclassSchema
# Dataclass constructors raise TypeError on unexpected keyword args or missing args.
non_conforming_dataclass_dicts = st.one_of(
    # Missing required field "name"
    st.fixed_dictionaries({"age": st.integers()}),
    # Missing required field "age"
    st.fixed_dictionaries({"name": st.text(min_size=1, max_size=20)}),
    # Empty dict (both required fields missing)
    st.just({}),
    # Extra unknown field that won't match dataclass constructor
    st.fixed_dictionaries(
        {
            "name": st.text(min_size=1, max_size=20),
            "age": st.integers(),
            "unknown_field_xyz": st.integers(),
        }
    ),
)

# Strategy: arbitrary strings (including empty)
arbitrary_strings = st.text(max_size=200)


# ---------------------------------------------------------------------------
# Property tests: Conforming dict input is accepted (Pydantic schema)
# ---------------------------------------------------------------------------


class TestConformingDictAcceptedPydantic:
    """A dict that conforms to a Pydantic input_schema is validated and accepted."""

    @settings(max_examples=100)
    @given(d=conforming_pydantic_dicts)
    def test_conforming_dict_passes_validation_pydantic(self, d: dict) -> None:
        agent = _make_built_agent(input_schema=PydanticSchema)

        # _coerce_input should NOT raise — it should accept the conforming dict
        result = agent._coerce_input(d)

        # The result should be a valid AgentInput
        assert isinstance(result, AgentInput)
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"


# ---------------------------------------------------------------------------
# Property tests: Conforming dict input is accepted (dataclass schema)
# ---------------------------------------------------------------------------


class TestConformingDictAcceptedDataclass:
    """A dict that conforms to a dataclass input_schema is validated and accepted."""

    @settings(max_examples=100)
    @given(d=conforming_dataclass_dicts)
    def test_conforming_dict_passes_validation_dataclass(self, d: dict) -> None:
        agent = _make_built_agent(input_schema=DataclassSchema)

        # _coerce_input should NOT raise — it should accept the conforming dict
        result = agent._coerce_input(d)

        # The result should be a valid AgentInput
        assert isinstance(result, AgentInput)
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"


# ---------------------------------------------------------------------------
# Property tests: Non-conforming dict raises InputValidationError (Pydantic)
# ---------------------------------------------------------------------------


class TestNonConformingDictRaisesPydantic:
    """A non-conforming dict raises InputValidationError before any model call."""

    @settings(max_examples=100)
    @given(d=non_conforming_pydantic_dicts)
    def test_non_conforming_dict_raises_pydantic(self, d: dict) -> None:
        agent = _make_built_agent(input_schema=PydanticSchema)

        with pytest.raises(InputValidationError):
            agent._coerce_input(d)

        # Model was never called (validation gates the run)
        agent.model_interface._providers["mock"].invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Property tests: Non-conforming dict raises InputValidationError (dataclass)
# ---------------------------------------------------------------------------


class TestNonConformingDictRaisesDataclass:
    """A non-conforming dict raises InputValidationError before any model call."""

    @settings(max_examples=100)
    @given(d=non_conforming_dataclass_dicts)
    def test_non_conforming_dict_raises_dataclass(self, d: dict) -> None:
        agent = _make_built_agent(input_schema=DataclassSchema)

        with pytest.raises(InputValidationError):
            agent._coerce_input(d)

        # Model was never called (validation gates the run)
        agent.model_interface._providers["mock"].invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Property tests: Conforming Pydantic model instance is accepted
# ---------------------------------------------------------------------------


class TestConformingModelInstanceAccepted:
    """A Pydantic model instance that matches the schema is accepted."""

    @settings(max_examples=100)
    @given(
        name=st.text(min_size=1, max_size=50),
        age=st.integers(min_value=0, max_value=200),
        active=st.booleans(),
    )
    def test_conforming_model_instance_accepted(
        self, name: str, age: int, active: bool
    ) -> None:
        agent = _make_built_agent(input_schema=PydanticSchema)
        model_input = PydanticSchema(name=name, age=age, active=active)

        result = agent._coerce_input(model_input)

        assert isinstance(result, AgentInput)
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"


# ---------------------------------------------------------------------------
# Property tests: Plain string bypasses validation (Req 1.6)
# ---------------------------------------------------------------------------


class TestStringBypassesValidation:
    """A plain string bypasses input_schema validation entirely."""

    @settings(max_examples=100)
    @given(s=arbitrary_strings)
    def test_string_bypasses_schema_validation(self, s: str) -> None:
        agent = _make_built_agent(input_schema=PydanticSchema)

        # Should NOT raise, even though string doesn't match the schema
        result = agent._coerce_input(s)

        assert isinstance(result, AgentInput)
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"

        # The text content should be the exact original string
        part = result.messages[0].parts[0]
        assert part.data is not None
        assert part.data.decode("utf-8") == s
