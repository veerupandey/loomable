"""Unit tests for loomable.content kernel bridging (task 2.3).

Covers ModelCapabilities defaults and the to_model_request / from_model_response
mapping between the multimodal content model and the provider-agnostic kernel
ModelRequest / ModelResponse shapes.
"""

from __future__ import annotations

import base64

from loomable.content import (
    AgentInput,
    AgentOutput,
    Image,
    Message,
    Modality,
    ModelCapabilities,
    Text,
    Video,
    from_model_response,
    to_model_request,
)
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# ModelCapabilities (Req 6.1 / 6.2)
# ---------------------------------------------------------------------------


def test_model_capabilities_default_text_only() -> None:
    caps = ModelCapabilities()
    assert caps.input == frozenset({Modality.TEXT})
    assert caps.output == frozenset({Modality.TEXT})


def test_model_capabilities_is_frozen() -> None:
    caps = ModelCapabilities()
    # Frozen dataclass: attribute assignment must fail.
    import dataclasses

    try:
        caps.input = frozenset({Modality.IMAGE})  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("ModelCapabilities should be frozen")


def test_model_capabilities_explicit_multimodal() -> None:
    caps = ModelCapabilities(
        input=frozenset({Modality.TEXT, Modality.IMAGE}),
        output=frozenset({Modality.TEXT, Modality.VIDEO}),
    )
    assert Modality.IMAGE in caps.input
    assert Modality.VIDEO in caps.output


# ---------------------------------------------------------------------------
# to_model_request (Req 4.3 / 4.5)
# ---------------------------------------------------------------------------


def test_to_model_request_text_part() -> None:
    agent_input = AgentInput.from_text("hello world")
    request = to_model_request(agent_input)

    assert isinstance(request, ModelRequest)
    assert request.messages == [
        {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
    ]


def test_to_model_request_preserves_message_and_part_order() -> None:
    agent_input = AgentInput(
        messages=[
            Message(role="system", parts=[Text("sys")]),
            Message(
                role="user",
                parts=[Text("look:"), Image(uri="https://ex.com/a.png")],
            ),
        ]
    )
    request = to_model_request(agent_input)

    assert [m["role"] for m in request.messages] == ["system", "user"]
    user_content = request.messages[1]["content"]
    assert user_content[0] == {"type": "text", "text": "look:"}
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://ex.com/a.png"},
    }


def test_to_model_request_inline_data_becomes_data_uri() -> None:
    raw = b"\x89PNG\r\n"
    agent_input = AgentInput(
        messages=[Message(role="user", parts=[Image(data=raw, media_type="image/png")])]
    )
    request = to_model_request(agent_input)

    url = request.messages[0]["content"][0]["image_url"]["url"]
    expected = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    assert url == expected


def test_to_model_request_video_part() -> None:
    agent_input = AgentInput(
        messages=[Message(role="user", parts=[Video(uri="https://ex.com/v.mp4")])]
    )
    request = to_model_request(agent_input)

    assert request.messages[0]["content"][0] == {
        "type": "video_url",
        "video_url": {"url": "https://ex.com/v.mp4"},
    }


def test_to_model_request_passes_through_optionals() -> None:
    agent_input = AgentInput.from_text("hi")
    tools = [{"name": "search"}]
    request = to_model_request(
        agent_input,
        tools=tools,
        temperature=0.2,
        max_tokens=128,
        metadata={"trace": "abc"},
    )

    assert request.tools == tools
    assert request.temperature == 0.2
    assert request.max_tokens == 128
    assert request.metadata == {"trace": "abc"}
    # Defensive copies, not aliases.
    assert request.tools is not tools


# ---------------------------------------------------------------------------
# from_model_response (Req 5.2 / 5.3 / 5.5)
# ---------------------------------------------------------------------------


def test_from_model_response_text_only_single_part() -> None:
    output = from_model_response(ModelResponse(content="the answer"))

    assert isinstance(output, AgentOutput)
    assert len(output.parts) == 1
    assert output.parts[0].modality is Modality.TEXT
    assert output.text() == "the answer"


def test_from_model_response_empty_yields_single_empty_text_part() -> None:
    output = from_model_response(ModelResponse(content=""))

    assert len(output.parts) == 1
    assert output.parts[0].modality is Modality.TEXT
    assert output.text() == ""


def test_from_model_response_text_then_media_order() -> None:
    img_bytes = b"imgdata"
    response = ModelResponse(
        content="here is your image",
        metadata={
            "media": [
                {
                    "modality": "image",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(img_bytes).decode("ascii"),
                }
            ]
        },
    )
    output = from_model_response(response)

    assert [p.modality for p in output.parts] == [Modality.TEXT, Modality.IMAGE]
    assert output.parts[0].data == b"here is your image"
    assert output.parts[1].media_type == "image/jpeg"
    assert output.parts[1].data == img_bytes


def test_from_model_response_media_by_uri() -> None:
    response = ModelResponse(
        content="",
        metadata={"media": [{"modality": "video", "uri": "https://ex.com/v.mp4"}]},
    )
    output = from_model_response(response)

    assert len(output.parts) == 1
    assert output.parts[0].modality is Modality.VIDEO
    assert output.parts[0].uri == "https://ex.com/v.mp4"


def test_from_model_response_infers_modality_from_media_type() -> None:
    response = ModelResponse(
        content="",
        metadata={"media": [{"media_type": "image/png", "uri": "https://ex.com/a.png"}]},
    )
    output = from_model_response(response)

    assert output.parts[0].modality is Modality.IMAGE


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_output_round_trip_text_and_media() -> None:
    img_bytes = b"\x01\x02\x03"
    response = ModelResponse(
        content="caption",
        metadata={
            "media": [
                {
                    "modality": "image",
                    "media_type": "image/png",
                    "data": base64.b64encode(img_bytes).decode("ascii"),
                },
                {"modality": "video", "uri": "https://ex.com/v.mp4"},
            ]
        },
    )
    output = from_model_response(response)

    assert [p.modality for p in output.parts] == [
        Modality.TEXT,
        Modality.IMAGE,
        Modality.VIDEO,
    ]
    assert output.text() == "caption"
    assert output.parts[1].data == img_bytes
    assert output.parts[2].uri == "https://ex.com/v.mp4"
