"""Unit tests for structured output on the high-level run flow (task 9.1).

Covers Req 13.1-13.4:
- A dataclass ``output_schema`` is parsed/validated into ``RunResult.structured``.
- A ``pydantic`` model ``output_schema`` is parsed/validated into ``structured``.
- Invalid JSON (or schema mismatch) raises ``StructuredOutputError`` (Req 13.3).
- With no ``output_schema``, ``structured`` stays ``None`` and output is unchanged
  (Req 13.4).

A fake provider returns a fixed JSON (or non-JSON) string as its response content.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from loomable.agent import Agent, ModelSpec, StructuredOutputError
from loomable.kernel.models import ModelRequest, ModelResponse


class JSONProvider:
    """A fake provider that returns a fixed string as its response content.

    Records the last request so tests can assert the schema hint was appended.
    """

    def __init__(self, content: str) -> None:
        self.content = content
        self.last_request: ModelRequest | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.last_request = request
        return ModelResponse(
            content=self.content,
            usage={"input_tokens": 1, "output_tokens": 1},
        )


def _agent(content: str) -> Agent:
    """A text agent whose provider always returns ``content``."""
    return Agent(model=ModelSpec(provider="json", provider_impl=JSONProvider(content)))


# --- dataclass schema -------------------------------------------------------


@dataclass
class Weather:
    city: str
    temp_c: int


async def test_dataclass_schema_parsed_into_structured() -> None:
    """A JSON response is coerced into the dataclass schema (Req 13.2)."""
    agent = _agent('{"city": "Vancouver", "temp_c": 12}')

    result = await agent.arun("weather?", output_schema=Weather)

    assert isinstance(result.structured, Weather)
    assert result.structured == Weather(city="Vancouver", temp_c=12)
    # The text output is still populated alongside the structured object.
    assert result.output.text() == '{"city": "Vancouver", "temp_c": 12}'


async def test_dataclass_schema_hint_appended_to_request() -> None:
    """A schema instruction is appended so the model is asked for JSON (Req 13.1)."""
    provider = JSONProvider('{"city": "Victoria", "temp_c": 9}')
    agent = Agent(model=ModelSpec(provider="json", provider_impl=provider))

    await agent.arun("weather?", output_schema=Weather)

    assert provider.last_request is not None
    last_message = provider.last_request.messages[-1]
    assert last_message["role"] == "system"
    assert "JSON" in last_message["content"][0]["text"]
    assert "Weather" in last_message["content"][0]["text"]


# --- pydantic schema --------------------------------------------------------


async def test_pydantic_schema_parsed_into_structured() -> None:
    """A JSON response is validated into a pydantic model (Req 13.2)."""
    pydantic = pytest.importorskip("pydantic")

    class Person(pydantic.BaseModel):
        name: str
        age: int

    agent = _agent('{"name": "Ada", "age": 36}')

    result = await agent.arun("who?", output_schema=Person)

    assert isinstance(result.structured, Person)
    assert result.structured.name == "Ada"
    assert result.structured.age == 36


async def test_pydantic_validation_failure_raises() -> None:
    """A response that violates the pydantic schema raises StructuredOutputError."""
    pydantic = pytest.importorskip("pydantic")

    class Person(pydantic.BaseModel):
        name: str
        age: int

    # ``age`` is a non-numeric string that cannot be coerced to int.
    agent = _agent('{"name": "Ada", "age": "not-a-number"}')

    with pytest.raises(StructuredOutputError):
        await agent.arun("who?", output_schema=Person)


# --- invalid JSON -----------------------------------------------------------


async def test_invalid_json_raises_structured_output_error() -> None:
    """Non-JSON content raises StructuredOutputError naming the failure (Req 13.3)."""
    agent = _agent("this is definitely not json")

    with pytest.raises(StructuredOutputError) as exc_info:
        await agent.arun("weather?", output_schema=Weather)

    assert "JSON" in str(exc_info.value)


async def test_dataclass_schema_mismatch_raises() -> None:
    """JSON with unexpected fields for a dataclass raises StructuredOutputError."""
    agent = _agent('{"unexpected": "field"}')

    with pytest.raises(StructuredOutputError):
        await agent.arun("weather?", output_schema=Weather)


# --- no schema (pass-through) ----------------------------------------------


async def test_no_schema_leaves_structured_none() -> None:
    """With no output_schema, structured stays None and output is unchanged (Req 13.4)."""
    provider = JSONProvider("plain text response")
    agent = Agent(model=ModelSpec(provider="json", provider_impl=provider))

    result = await agent.arun("hi")

    assert result.structured is None
    assert result.output.text() == "plain text response"
    # No schema hint appended when no schema requested.
    assert provider.last_request is not None
    assert len(provider.last_request.messages) == 1
