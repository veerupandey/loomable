"""Unit tests for tool-generated media feedback injection in _run_tool_loop (Req 7.1–7.5).

Verifies:
- Media is injected into the tool result message when modality is in capabilities.input
  and feedback_media is True.
- Media is NOT injected when feedback_media is False.
- Media is NOT injected when the modality is not in capabilities.input.
- The tool_call_id association in the tool result message is preserved.
- Media still appears on RunResult properties even when not injected.
"""

from __future__ import annotations

import asyncio

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.agent.tools import FunctionTool
from loomable.content import Modality, ModelCapabilities
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.media import Image


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _ToolCallProvider:
    """Provider that first returns a tool call, then a final answer."""

    def __init__(self, tool_name: str = "gen_image", tool_args: dict | None = None):
        self._tool_name = tool_name
        self._tool_args = tool_args or {}
        self._call_count = 0
        self._requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self._requests.append(request)
        self._call_count += 1
        if self._call_count == 1:
            # First call: model requests to call the tool
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        tool_name=self._tool_name,
                        args=self._tool_args,
                    )
                ],
            )
        # Second call: model provides final answer
        return ModelResponse(content="I analyzed the image.")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeedbackInjection:
    """Verify tool-generated media feedback injection in _run_tool_loop."""

    def _build_agent(self, *, feedback_media: bool = True, include_image_input: bool = True):
        """Build an agent with a tool that returns an Image."""
        input_modalities = frozenset({Modality.TEXT})
        if include_image_input:
            input_modalities = frozenset({Modality.TEXT, Modality.IMAGE})

        capabilities = ModelCapabilities(
            input=input_modalities,
            output=frozenset({Modality.TEXT}),
        )

        provider = _ToolCallProvider(tool_name="gen_image", tool_args={})
        agent = Agent(
            model=ModelSpec(
                provider="test",
                provider_impl=provider,
                capabilities=capabilities,
            ),
            feedback_media=feedback_media,
        )
        built = agent.build()

        # Add a tool that returns an Image
        def gen_image() -> Image:
            """Generate an image."""
            return Image(content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, mime_type="image/png")

        tool = FunctionTool(gen_image, name="gen_image", description="Generate an image")
        built.tool_runtime._tools["gen_image"] = tool

        return built, provider

    def test_media_injected_when_capable_and_enabled(self):
        """Media is injected into tool result message when modality is supported and feedback enabled."""
        built, provider = self._build_agent(feedback_media=True, include_image_input=True)
        result = asyncio.run(built.arun("Generate an image"))

        # Check the second request (after tool call) contains an image in the tool message
        assert len(provider._requests) == 2
        second_request = provider._requests[1]

        # Find the tool result message
        tool_msgs = [m for m in second_request.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        tool_msg = tool_msgs[0]
        # Should have text content + injected image_url content
        assert tool_msg["tool_call_id"] == "call_1"
        assert len(tool_msg["content"]) == 2  # text + image
        assert tool_msg["content"][0]["type"] == "text"
        assert tool_msg["content"][1]["type"] == "image_url"

    def test_media_not_injected_when_feedback_disabled(self):
        """Media is NOT injected when feedback_media=False."""
        built, provider = self._build_agent(feedback_media=False, include_image_input=True)
        result = asyncio.run(built.arun("Generate an image"))

        assert len(provider._requests) == 2
        second_request = provider._requests[1]

        tool_msgs = [m for m in second_request.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        tool_msg = tool_msgs[0]
        # Should only have text content, no injected image
        assert len(tool_msg["content"]) == 1
        assert tool_msg["content"][0]["type"] == "text"

    def test_media_not_injected_when_modality_not_in_capabilities(self):
        """Media is NOT injected when the modality is not in capabilities.input."""
        built, provider = self._build_agent(feedback_media=True, include_image_input=False)
        result = asyncio.run(built.arun("Generate an image"))

        assert len(provider._requests) == 2
        second_request = provider._requests[1]

        tool_msgs = [m for m in second_request.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        tool_msg = tool_msgs[0]
        # Should only have text content, no injected image
        assert len(tool_msg["content"]) == 1
        assert tool_msg["content"][0]["type"] == "text"

    def test_tool_call_id_preserved(self):
        """The tool_call_id association is preserved after injection."""
        built, provider = self._build_agent(feedback_media=True, include_image_input=True)
        result = asyncio.run(built.arun("Generate an image"))

        second_request = provider._requests[1]
        tool_msgs = [m for m in second_request.messages if m.get("role") == "tool"]
        assert tool_msgs[0]["tool_call_id"] == "call_1"

    def test_media_still_on_run_result_when_not_injected(self):
        """Even when media is NOT injected, it still appears on RunResult properties."""
        built, provider = self._build_agent(feedback_media=False, include_image_input=True)
        result = asyncio.run(built.arun("Generate an image"))

        # Media should still be accessible via RunResult.images
        assert len(result.images) == 1
        assert isinstance(result.images[0], Image)
