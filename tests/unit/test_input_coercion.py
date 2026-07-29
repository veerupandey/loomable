"""Unit tests for input coercion and schema validation (task 1.1).

Covers:
- _coerce_input accepts str, AgentInput, Pydantic model, dataclass, and dict.
- Strings and AgentInput bypass schema validation (Req 1.3/1.6).
- Pydantic/dataclass/dict inputs are validated against input_schema (Req 1.4).
- Non-conforming inputs raise InputValidationError before any model call (Req 1.5).
- arun/run/astream all route through _coerce_input.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import pydantic
import pytest

from loomable.agent import Agent, ModelSpec, RunResult
from loomable.agent.errors import InputValidationError
from loomable.content import AgentInput, Message, Modality, Text
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _TrackingProvider:
    """Records whether it was called and what text it received."""

    def __init__(self) -> None:
        self.called = False
        self.last_text: str | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.called = True
        for msg in request.messages:
            if msg["role"] == "user":
                for part in msg["content"]:
                    if part.get("type") == "text":
                        self.last_text = part["text"]
                        break
        return ModelResponse(content="ok")


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------


class OrderSchema(pydantic.BaseModel):
    item: str
    quantity: int


@dataclass
class TicketSchema:
    title: str
    priority: int = 3


# ---------------------------------------------------------------------------
# _coerce_input: str and AgentInput bypass schema (Req 1.3, 1.6)
# ---------------------------------------------------------------------------


class TestCoerceInputBypass:
    """Strings and AgentInput bypass schema validation even when input_schema is set."""

    async def test_string_bypasses_schema(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=OrderSchema,
        )
        result = await agent.arun("just a string")
        assert provider.called is True
        assert provider.last_text == "just a string"

    async def test_agent_input_bypasses_schema(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=OrderSchema,
        )
        ai = AgentInput(messages=[Message(role="user", parts=[Text("raw")])])
        result = await agent.arun(ai)
        assert provider.called is True
        assert provider.last_text == "raw"


# ---------------------------------------------------------------------------
# _coerce_input: structured values are validated and serialized (Req 1.1, 1.2, 1.4)
# ---------------------------------------------------------------------------


class TestCoerceInputPydantic:
    """Pydantic models and dicts are validated against the input_schema."""

    async def test_pydantic_model_matching_schema_accepted(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=OrderSchema,
        )
        order = OrderSchema(item="widget", quantity=5)
        result = await agent.arun(order)
        assert provider.called is True
        # The serialized JSON is passed as user message text.
        data = json.loads(provider.last_text)
        assert data == {"item": "widget", "quantity": 5}

    async def test_dict_matching_schema_accepted(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=OrderSchema,
        )
        result = await agent.arun({"item": "gadget", "quantity": 10})
        assert provider.called is True
        data = json.loads(provider.last_text)
        assert data == {"item": "gadget", "quantity": 10}

    async def test_dict_not_matching_schema_raises(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=OrderSchema,
        )
        with pytest.raises(InputValidationError) as exc_info:
            await agent.arun({"item": "gadget"})  # missing quantity
        assert "quantity" in exc_info.value.reason or "field" in exc_info.value.reason.lower()
        # The provider must NOT have been called.
        assert provider.called is False


class TestCoerceInputDataclass:
    """Dataclass instances and dicts are validated against a dataclass schema."""

    async def test_dataclass_matching_schema_accepted(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=TicketSchema,
        )
        ticket = TicketSchema(title="bug fix", priority=1)
        result = await agent.arun(ticket)
        assert provider.called is True
        data = json.loads(provider.last_text)
        assert data == {"title": "bug fix", "priority": 1}

    async def test_dict_matching_dataclass_schema_accepted(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=TicketSchema,
        )
        result = await agent.arun({"title": "new feature", "priority": 2})
        assert provider.called is True
        data = json.loads(provider.last_text)
        assert data == {"title": "new feature", "priority": 2}

    async def test_dict_not_matching_dataclass_schema_raises(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=TicketSchema,
        )
        with pytest.raises(InputValidationError) as exc_info:
            await agent.arun({"priority": 1})  # missing required 'title'
        assert "title" in exc_info.value.reason.lower() or "dataclass" in exc_info.value.reason.lower()
        assert provider.called is False


# ---------------------------------------------------------------------------
# No input_schema: structured values pass without validation (Req 1.1, 1.2)
# ---------------------------------------------------------------------------


class TestCoerceInputNoSchema:
    """Without input_schema, structured values are serialized without validation."""

    async def test_pydantic_model_serialized_without_schema(self):
        provider = _TrackingProvider()
        agent = Agent(model=ModelSpec(provider="t", provider_impl=provider))
        order = OrderSchema(item="widget", quantity=5)
        result = await agent.arun(order)
        assert provider.called is True
        data = json.loads(provider.last_text)
        assert data == {"item": "widget", "quantity": 5}

    async def test_dict_serialized_without_schema(self):
        provider = _TrackingProvider()
        agent = Agent(model=ModelSpec(provider="t", provider_impl=provider))
        result = await agent.arun({"foo": "bar", "count": 42})
        assert provider.called is True
        data = json.loads(provider.last_text)
        assert data == {"foo": "bar", "count": 42}

    async def test_dataclass_serialized_without_schema(self):
        provider = _TrackingProvider()
        agent = Agent(model=ModelSpec(provider="t", provider_impl=provider))
        ticket = TicketSchema(title="test", priority=5)
        result = await agent.arun(ticket)
        assert provider.called is True
        data = json.loads(provider.last_text)
        assert data == {"title": "test", "priority": 5}


# ---------------------------------------------------------------------------
# run (sync) and astream also route through _coerce_input
# ---------------------------------------------------------------------------


class TestRunAndStreamCoercion:
    """Verify that run() and astream() also go through _coerce_input."""

    def test_sync_run_validates_schema(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=OrderSchema,
        )
        with pytest.raises(InputValidationError):
            agent.run({"item": "x"})  # missing quantity
        assert provider.called is False

    def test_sync_run_accepts_valid_input(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=OrderSchema,
        )
        result = agent.run({"item": "x", "quantity": 1})
        assert provider.called is True
        assert isinstance(result, RunResult)

    async def test_astream_validates_schema(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=OrderSchema,
        )
        with pytest.raises(InputValidationError):
            async for _ in agent.astream({"wrong": "data"}):
                pass
        assert provider.called is False

    async def test_astream_accepts_valid_input(self):
        provider = _TrackingProvider()
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            input_schema=OrderSchema,
        )
        chunks = [chunk async for chunk in agent.astream({"item": "y", "quantity": 2})]
        assert provider.called is True
        assert len(chunks) >= 1
