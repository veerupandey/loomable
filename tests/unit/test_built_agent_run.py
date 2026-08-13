"""Unit tests for the high-level run flow and capability gating (task 3.2).

Covers:
- String input is wrapped via ``AgentInput.from_text`` (Req 1.4).
- Text round-trips through the run to the ``AgentOutput`` (Req 5.1).
- Input modality gating raises ``UnsupportedModalityError`` *without* invoking the
  provider (Req 4.4, 6.3, 6.4).
- Output modality gating raises ``UnsupportedModalityError`` (Req 5.4).
- ``RunResult`` fields are populated (output, session_id, usage, tool_activity).
- ``astream`` yields chunks ending with ``done=True`` (Req 1.5).
"""

from __future__ import annotations

import base64

import pytest

from loomable.agent import (
    Agent,
    ModelSpec,
    RunChunk,
    RunResult,
    UnsupportedModalityError,
)
from loomable.content import (
    AgentInput,
    Image,
    Message,
    Modality,
    ModelCapabilities,
    Text,
)
from loomable.kernel.models import ModelRequest, ModelResponse


class EchoProvider:
    """A fake provider that echoes the first text part back as content.

    Records the last request it received (and whether it was called) so tests can
    assert the provider is *not* invoked when input gating rejects a run.
    """

    def __init__(self) -> None:
        self.called = False
        self.last_request: ModelRequest | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.called = True
        self.last_request = request
        # Find the first user text part in the content array.
        text = ""
        for message in request.messages:
            if message["role"] == "user":
                for part in message["content"]:
                    if part.get("type") == "text":
                        text = part["text"]
                        break
                if text:
                    break
        return ModelResponse(
            content=f"echo: {text}",
            usage={"input_tokens": 3, "output_tokens": 2},
        )


class ImageEmittingProvider:
    """A fake provider that returns an image media part in its response metadata."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        png = base64.b64encode(b"\x89PNG-bytes").decode("ascii")
        return ModelResponse(
            content="here is an image",
            usage={"input_tokens": 1, "output_tokens": 1},
            metadata={
                "media": [
                    {"modality": "image", "media_type": "image/png", "data": png}
                ]
            },
        )


def _text_agent(provider: EchoProvider | None = None) -> Agent:
    """A text-in / text-out agent over the given (or a fresh) EchoProvider."""
    provider = provider or EchoProvider()
    return Agent(model=ModelSpec(provider="echo", provider_impl=provider))


async def test_string_input_is_wrapped_and_round_trips() -> None:
    """A bare string is wrapped via from_text and the text round-trips (Req 1.4/5.1)."""
    provider = EchoProvider()
    agent = _text_agent(provider)

    result = await agent.arun("hello world")

    assert isinstance(result, RunResult)
    assert provider.called is True
    assert result.output.text() == "echo: hello world"


async def test_run_result_fields_are_populated() -> None:
    """RunResult carries output, session_id, usage, and tool_activity (Req 1.4)."""
    agent = _text_agent()

    result = await agent.arun("hi")

    assert result.output.text() == "echo: hi"
    assert result.session_id  # non-empty session id
    assert result.usage == {"input_tokens": 3, "output_tokens": 2}
    assert result.tool_activity == []
    assert result.structured is None
    assert result.sub_results is None


async def test_agent_input_object_is_accepted() -> None:
    """An AgentInput object (not just a string) is accepted directly (Req 4.1)."""
    agent = _text_agent()
    agent_input = AgentInput(messages=[Message(role="user", parts=[Text("typed")])])

    result = await agent.arun(agent_input)

    assert result.output.text() == "echo: typed"


async def test_instructions_prepended_as_system_message() -> None:
    """When instructions are set, a leading system message is prepended."""
    provider = EchoProvider()
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        instructions="be terse",
    )

    await agent.arun("hey")

    assert provider.last_request is not None
    first = provider.last_request.messages[0]
    assert first["role"] == "system"
    assert first["content"][0]["text"] == "be terse"


async def test_input_modality_gating_raises_without_provider_call() -> None:
    """Unsupported input modality raises before the provider is invoked (Req 4.4/6.3)."""
    provider = EchoProvider()
    # Explicit text-only lock-down so an image input is unsupported.
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        capabilities=ModelCapabilities(
            input=frozenset({Modality.TEXT}),
            output=frozenset({Modality.TEXT}),
        ),
    )
    image_input = AgentInput(
        messages=[Message(role="user", parts=[Image(data=b"\x89PNG")])]
    )

    with pytest.raises(UnsupportedModalityError) as exc_info:
        await agent.arun(image_input)

    assert exc_info.value.modality == Modality.IMAGE.value
    assert exc_info.value.model == "echo"
    # Critical: the provider must NOT have been called (fail fast, no side effects).
    assert provider.called is False


async def test_default_agent_accepts_image_input_at_gate() -> None:
    """Default capabilities include IMAGE; gating does not raise for images=."""
    provider = EchoProvider()
    agent = Agent(model=ModelSpec(provider="echo", provider_impl=provider))
    image_input = AgentInput(
        messages=[Message(role="user", parts=[Image(data=b"\x89PNG")])]
    )

    result = await agent.arun(image_input)

    assert provider.called is True
    assert isinstance(result, RunResult)


async def test_output_modality_gating_raises() -> None:
    """A response with media beyond declared output capabilities raises (Req 5.4)."""
    provider = ImageEmittingProvider()
    # Allow image input so gating happens on OUTPUT, but keep output text-only.
    agent = Agent(
        model=ModelSpec(
            provider="img",
            provider_impl=provider,
            capabilities=ModelCapabilities(
                input=frozenset({Modality.TEXT}),
                output=frozenset({Modality.TEXT}),
            ),
        )
    )

    with pytest.raises(UnsupportedModalityError) as exc_info:
        await agent.arun("make an image")

    assert exc_info.value.modality == Modality.IMAGE.value
    assert exc_info.value.model == "img"


async def test_output_image_allowed_when_declared() -> None:
    """When output image capability is declared, an image output is returned."""
    provider = ImageEmittingProvider()
    agent = Agent(
        model=ModelSpec(
            provider="img",
            provider_impl=provider,
            capabilities=ModelCapabilities(
                input=frozenset({Modality.TEXT}),
                output=frozenset({Modality.TEXT, Modality.IMAGE}),
            ),
        )
    )

    result = await agent.arun("make an image")

    assert Modality.IMAGE in result.output.modalities()
    assert result.output.text() == "here is an image"


async def test_astream_yields_chunks_ending_done() -> None:
    """astream yields RunChunks, the final one marked done (Req 1.5)."""
    agent = _text_agent()

    chunks = [chunk async for chunk in agent.astream("stream me")]

    assert len(chunks) >= 1
    assert all(isinstance(c, RunChunk) for c in chunks)
    assert chunks[-1].done is True
    assert all(c.done is False for c in chunks[:-1])
    # Reassembled text matches the run output.
    text = "".join(
        c.delta.data.decode("utf-8")
        for c in chunks
        if c.delta.modality is Modality.TEXT and c.delta.data is not None
    )
    assert text == "echo: stream me"


def test_sync_run_wrapper() -> None:
    """The synchronous run() wrapper returns a RunResult (Req 1.4)."""
    agent = _text_agent()

    result = agent.run("sync call")

    assert isinstance(result, RunResult)
    assert result.output.text() == "echo: sync call"


async def test_audio_input_gating_raises_without_provider_call() -> None:
    """Unsupported audio modality raises before the provider is invoked (Req 4.4)."""
    provider = EchoProvider()
    # Audio stays opt-in; default multimodal caps exclude AUDIO.
    agent = Agent(model=ModelSpec(provider="echo", provider_impl=provider))

    with pytest.raises(UnsupportedModalityError) as exc_info:
        await agent.arun("transcribe this", audio=["test.wav"])

    assert exc_info.value.modality == "audio"
    assert exc_info.value.model == "echo"
    # Critical: the provider must NOT have been called (fail fast, no side effects).
    assert provider.called is False


async def test_audio_input_allowed_when_capability_declared() -> None:
    """Audio input passes through when model declares audio capability (Req 4.1)."""
    from loomable.content import MediaPart

    provider = EchoProvider()
    agent = Agent(
        model=ModelSpec(
            provider="echo",
            provider_impl=provider,
            capabilities=ModelCapabilities(
                input=frozenset({Modality.TEXT, Modality.AUDIO}),
                output=frozenset({Modality.TEXT}),
            ),
        )
    )

    # Use a MediaPart directly to avoid filesystem read.
    audio_part = MediaPart(modality=Modality.AUDIO, media_type="audio/wav", data=b"RIFF-fake")
    result = await agent.arun("transcribe this", audio=[audio_part])

    assert provider.called is True
    assert result.output.text() == "echo: transcribe this"


async def test_audio_none_does_not_trigger_gating() -> None:
    """When audio=None (default), no gating check fires even on text-only model."""
    provider = EchoProvider()
    agent = Agent(model=ModelSpec(provider="echo", provider_impl=provider))

    # Should not raise; audio is None by default.
    result = await agent.arun("hello")

    assert provider.called is True
    assert result.output.text() == "echo: hello"
