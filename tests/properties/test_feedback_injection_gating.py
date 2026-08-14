# Feature: multimodal-io, Property 11: Feedback Injection Respects Capabilities
"""Property 11: Feedback Injection Respects Capabilities.

For any tool-generated Media_Class of modality M, if M is in the agent's
capabilities.input AND feedback_media=True, the media SHALL be injected into
the conversation messages. If M is NOT in capabilities.input OR
feedback_media=False, the media SHALL NOT be injected (but SHALL still appear
on RunResult properties).

**Validates: Requirements 7.1, 7.2, 7.5**
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent import Agent, ModelSpec
from loomable.agent.tools import FunctionTool
from loomable.content import Modality, ModelCapabilities
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.media import Audio, Image, Video
from loomable.media.types import _MediaBase


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# The modalities that tool-generated media can have (excluding TEXT which is
# always in capabilities.input and is not a media modality for injection).
media_modalities = st.sampled_from([Modality.IMAGE, Modality.AUDIO, Modality.VIDEO])

# Strategy: generate a frozenset of input capabilities (always includes TEXT,
# may or may not include any media modalities).
capabilities_input_sets = st.frozensets(
    st.sampled_from([Modality.IMAGE, Modality.AUDIO, Modality.VIDEO]),
    min_size=0,
    max_size=3,
).map(lambda s: s | frozenset({Modality.TEXT}))

# Strategy: feedback_media flag
feedback_media_flag = st.booleans()

# Strategy: random content bytes for constructing media instances
content_bytes = st.binary(min_size=8, max_size=64)


# ---------------------------------------------------------------------------
# Test double: a provider that returns one tool call then a final answer.
# ---------------------------------------------------------------------------


class _ToolCallProvider:
    """Provider that first returns a tool call, then a final answer."""

    def __init__(self, tool_name: str = "gen_media"):
        self._tool_name = tool_name
        self._call_count = 0
        self._requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self._requests.append(request)
        self._call_count += 1
        if self._call_count == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        tool_name=self._tool_name,
                        args={},
                    )
                ],
            )
        return ModelResponse(content="Done.")


# ---------------------------------------------------------------------------
# Helper: build an agent with the given capabilities and feedback flag,
# and a tool that returns media of the specified modality.
# ---------------------------------------------------------------------------


def _build_agent_for_test(
    *,
    media_modality: Modality,
    capabilities_input: frozenset[Modality],
    feedback_media: bool,
    media_content: bytes,
) -> tuple[Any, _ToolCallProvider]:
    """Construct a minimal agent with a tool returning a specific media modality."""
    capabilities = ModelCapabilities(
        input=capabilities_input,
        output=frozenset({Modality.TEXT}),
    )

    provider = _ToolCallProvider(tool_name="gen_media")
    agent = Agent(
        model=ModelSpec(
            provider="test",
            provider_impl=provider,
            capabilities=capabilities,
        ),
        feedback_media=feedback_media,
    )
    built = agent.build()

    # Create a media instance of the given modality
    if media_modality == Modality.IMAGE:
        media_instance = Image(content=media_content, mime_type="image/png")
    elif media_modality == Modality.AUDIO:
        media_instance = Audio(content=media_content, mime_type="audio/wav")
    elif media_modality == Modality.VIDEO:
        media_instance = Video(content=media_content, mime_type="video/mp4")
    else:
        media_instance = Image(content=media_content, mime_type="image/png")

    # Register a tool that returns this media instance
    captured = {"media": media_instance}

    def gen_media() -> _MediaBase:
        """Generate media."""
        return captured["media"]

    tool = FunctionTool(gen_media, name="gen_media", description="Generate media")
    built.tool_runtime._tools["gen_media"] = tool

    return built, provider


def _count_media_content_entries(msg: dict) -> int:
    """Count non-text content entries in a message (injected media parts)."""
    content = msg.get("content", [])
    if not isinstance(content, list):
        return 0
    return sum(
        1 for entry in content if isinstance(entry, dict) and entry.get("type") != "text"
    )


def _count_feedback_media(request: ModelRequest) -> int:
    """Media feedback is a follow-up user message (not attached to role=tool)."""
    total = 0
    for msg in request.messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        texts = [
            str(entry.get("text") or "")
            for entry in content
            if isinstance(entry, dict) and entry.get("type") == "text"
        ]
        if any("produced media" in text for text in texts):
            total += _count_media_content_entries(msg)
    return total


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestFeedbackInjectionRespectsCapabilities:
    """Property 11: Feedback injection respects capabilities gating."""

    @settings(max_examples=100, deadline=None)
    @given(
        media_modality=media_modalities,
        capabilities_input=capabilities_input_sets,
        media_content=content_bytes,
    )
    def test_media_injected_when_modality_in_capabilities_and_feedback_enabled(
        self,
        media_modality: Modality,
        capabilities_input: frozenset[Modality],
        media_content: bytes,
    ) -> None:
        """When feedback_media=True AND modality in capabilities.input, media is injected."""
        # Only test cases where modality IS in capabilities
        if media_modality not in capabilities_input:
            return  # Skip — this case is covered by the next test

        built, provider = _build_agent_for_test(
            media_modality=media_modality,
            capabilities_input=capabilities_input,
            feedback_media=True,
            media_content=media_content,
        )
        result = asyncio.run(built.arun("Generate media"))

        # The second request to the provider should contain the injected media
        assert len(provider._requests) >= 2
        second_request = provider._requests[1]

        tool_msgs = [m for m in second_request.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        media_entries = _count_feedback_media(second_request)
        assert media_entries >= 1, (
            f"Expected media injection for modality={media_modality} "
            f"with capabilities={capabilities_input} and feedback_media=True, "
            f"but found {media_entries} media entries in feedback user message"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        media_modality=media_modalities,
        capabilities_input=capabilities_input_sets,
        media_content=content_bytes,
    )
    def test_media_not_injected_when_feedback_disabled(
        self,
        media_modality: Modality,
        capabilities_input: frozenset[Modality],
        media_content: bytes,
    ) -> None:
        """When feedback_media=False, media is NEVER injected regardless of capabilities."""
        built, provider = _build_agent_for_test(
            media_modality=media_modality,
            capabilities_input=capabilities_input,
            feedback_media=False,
            media_content=media_content,
        )
        result = asyncio.run(built.arun("Generate media"))

        assert len(provider._requests) >= 2
        second_request = provider._requests[1]

        tool_msgs = [m for m in second_request.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        media_entries = _count_feedback_media(second_request)
        assert media_entries == 0, (
            f"Expected NO media injection when feedback_media=False, "
            f"but found {media_entries} media entries in feedback user message"
        )

        # But media should still appear on RunResult properties
        if media_modality == Modality.IMAGE:
            assert len(result.images) >= 1
        elif media_modality == Modality.AUDIO:
            assert len(result.audio) >= 1
        elif media_modality == Modality.VIDEO:
            assert len(result.videos) >= 1

    @settings(max_examples=100, deadline=None)
    @given(
        media_modality=media_modalities,
        capabilities_input=capabilities_input_sets,
        media_content=content_bytes,
    )
    def test_media_not_injected_when_modality_not_in_capabilities(
        self,
        media_modality: Modality,
        capabilities_input: frozenset[Modality],
        media_content: bytes,
    ) -> None:
        """When modality NOT in capabilities.input, media is NOT injected."""
        # Only test cases where modality is NOT in capabilities
        if media_modality in capabilities_input:
            return  # Skip — this case is covered by the first test

        built, provider = _build_agent_for_test(
            media_modality=media_modality,
            capabilities_input=capabilities_input,
            feedback_media=True,
            media_content=media_content,
        )
        result = asyncio.run(built.arun("Generate media"))

        assert len(provider._requests) >= 2
        second_request = provider._requests[1]

        tool_msgs = [m for m in second_request.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        media_entries = _count_feedback_media(second_request)
        assert media_entries == 0, (
            f"Expected NO media injection for modality={media_modality} "
            f"NOT in capabilities={capabilities_input}, "
            f"but found {media_entries} media entries in feedback user message"
        )

        # But media should still appear on RunResult properties
        if media_modality == Modality.IMAGE:
            assert len(result.images) >= 1
        elif media_modality == Modality.AUDIO:
            assert len(result.audio) >= 1
        elif media_modality == Modality.VIDEO:
            assert len(result.videos) >= 1

    @settings(max_examples=100, deadline=None)
    @given(
        media_modality=media_modalities,
        capabilities_input=capabilities_input_sets,
        feedback_media=feedback_media_flag,
        media_content=content_bytes,
    )
    def test_injection_decision_matches_property(
        self,
        media_modality: Modality,
        capabilities_input: frozenset[Modality],
        feedback_media: bool,
        media_content: bytes,
    ) -> None:
        """Unified property: injection iff modality in capabilities AND feedback_media=True."""
        should_inject = (media_modality in capabilities_input) and feedback_media

        built, provider = _build_agent_for_test(
            media_modality=media_modality,
            capabilities_input=capabilities_input,
            feedback_media=feedback_media,
            media_content=media_content,
        )
        result = asyncio.run(built.arun("Generate media"))

        assert len(provider._requests) >= 2
        second_request = provider._requests[1]

        tool_msgs = [m for m in second_request.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        media_entries = _count_feedback_media(second_request)

        if should_inject:
            assert media_entries >= 1, (
                f"Expected injection (modality={media_modality} in "
                f"capabilities={capabilities_input}, feedback_media={feedback_media}) "
                f"but found {media_entries} media entries"
            )
        else:
            assert media_entries == 0, (
                f"Expected NO injection (modality={media_modality}, "
                f"capabilities={capabilities_input}, feedback_media={feedback_media}) "
                f"but found {media_entries} media entries"
            )

        # Invariant: media ALWAYS appears on RunResult properties regardless of injection
        if media_modality == Modality.IMAGE:
            assert len(result.images) >= 1
        elif media_modality == Modality.AUDIO:
            assert len(result.audio) >= 1
        elif media_modality == Modality.VIDEO:
            assert len(result.videos) >= 1
