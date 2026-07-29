"""Sanity unit tests for Message, AgentInput, and AgentOutput (task 2.2)."""

from __future__ import annotations

import pytest

from loomable.content import (
    AgentInput,
    AgentOutput,
    Image,
    Message,
    Modality,
    Text,
    Video,
)


class TestMessage:
    def test_role_and_parts(self) -> None:
        msg = Message(role="user", parts=[Text("hi")])
        assert msg.role == "user"
        assert msg.parts[0].data == b"hi"

    def test_default_parts_empty_list(self) -> None:
        msg = Message(role="system")
        assert msg.parts == []


class TestAgentInputConstruction:
    def test_non_empty_messages_ok(self) -> None:
        agent_input = AgentInput(messages=[Message(role="user", parts=[Text("hi")])])
        assert len(agent_input.messages) == 1

    def test_empty_messages_raises(self) -> None:
        with pytest.raises(ValueError):
            AgentInput(messages=[])

    def test_messages_preserve_order(self) -> None:
        m1 = Message(role="user", parts=[Text("first")])
        m2 = Message(role="assistant", parts=[Text("second")])
        agent_input = AgentInput(messages=[m1, m2])
        assert agent_input.messages == [m1, m2]


class TestAgentInputFromText:
    def test_from_text_builds_single_user_message(self) -> None:
        agent_input = AgentInput.from_text("hello world")
        assert len(agent_input.messages) == 1
        message = agent_input.messages[0]
        assert message.role == "user"
        assert len(message.parts) == 1
        part = message.parts[0]
        assert part.modality is Modality.TEXT
        assert part.data == b"hello world"


class TestAgentInputModalities:
    def test_text_only(self) -> None:
        agent_input = AgentInput.from_text("hi")
        assert agent_input.modalities() == {Modality.TEXT}

    def test_union_across_messages_and_parts(self) -> None:
        agent_input = AgentInput(
            messages=[
                Message(role="user", parts=[Text("hi"), Image(data=b"\x89PNG")]),
                Message(role="user", parts=[Video(data=b"\x00\x00")]),
            ]
        )
        assert agent_input.modalities() == {
            Modality.TEXT,
            Modality.IMAGE,
            Modality.VIDEO,
        }

    def test_no_parts_yields_empty_set(self) -> None:
        agent_input = AgentInput(messages=[Message(role="user")])
        assert agent_input.modalities() == set()


class TestAgentOutputConstruction:
    def test_non_empty_parts_ok(self) -> None:
        output = AgentOutput(parts=[Text("done")])
        assert len(output.parts) == 1

    def test_empty_parts_raises(self) -> None:
        with pytest.raises(ValueError):
            AgentOutput(parts=[])

    def test_parts_preserve_order(self) -> None:
        p1 = Text("a")
        p2 = Text("b")
        output = AgentOutput(parts=[p1, p2])
        assert output.parts == [p1, p2]


class TestAgentOutputText:
    def test_concatenates_text_parts(self) -> None:
        output = AgentOutput(parts=[Text("Hello, "), Text("world")])
        assert output.text() == "Hello, world"

    def test_ignores_non_text_parts(self) -> None:
        output = AgentOutput(
            parts=[Text("caption: "), Image(data=b"\x89PNG"), Text("done")]
        )
        assert output.text() == "caption: done"

    def test_uri_only_text_part_contributes_nothing(self) -> None:
        # A text part referenced by uri (no inline data) has no bytes to decode.
        uri_part = Video(uri="https://example.com/v.mp4")
        output = AgentOutput(parts=[Text("x"), uri_part])
        assert output.text() == "x"

    def test_decodes_utf8(self) -> None:
        output = AgentOutput(parts=[Text("héllo \u2603")])
        assert output.text() == "héllo \u2603"


class TestAgentOutputModalities:
    def test_single_text(self) -> None:
        output = AgentOutput(parts=[Text("hi")])
        assert output.modalities() == {Modality.TEXT}

    def test_mixed(self) -> None:
        output = AgentOutput(parts=[Text("hi"), Image(data=b"\x89PNG")])
        assert output.modalities() == {Modality.TEXT, Modality.IMAGE}
