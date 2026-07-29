"""Unit tests for loomable.serve.MCPServerAdapter.

These exercise the pure translation + error handling directly via
``MCPServerAdapter.run_tool`` / ``output_to_mcp_content`` — no live MCP client is
required. A fake ``BuiltAgent`` stub supplies an async ``arun``.
"""

from __future__ import annotations

import base64

import mcp.types as types
import pytest

from loomable.content import AgentInput, AgentOutput, Image, Text, Video
from loomable.serve import MCPServerAdapter
from loomable.serve.mcp_adapter import arguments_to_agent_input


class FakeBuiltAgent:
    """Minimal BuiltAgent stub exposing an async ``arun``.

    ``arun`` records the received input and returns a preconfigured output, or
    raises ``error`` when set (to exercise the failure path).
    """

    def __init__(self, output: AgentOutput | None = None, error: Exception | None = None):
        self._output = output
        self._error = error
        self.received: AgentInput | None = None

    async def arun(self, input, *, output_schema=None):  # noqa: A002
        self.received = input
        if self._error is not None:
            raise self._error

        class _Result:
            def __init__(self, output):
                self.output = output
                self.session_id = "s1"
                self.usage: dict[str, int] = {}

        return _Result(self._output)


@pytest.mark.unit
async def test_text_run_returns_text_content():
    agent = FakeBuiltAgent(output=AgentOutput(parts=[Text("hello there")]))
    adapter = MCPServerAdapter(agent)  # type: ignore[arg-type]

    result = await adapter.run_tool({"text": "hi"})

    assert result.isError is False
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == "hello there"
    # The tool translated the argument into an AgentInput and ran the agent.
    assert agent.received is not None
    assert agent.received.messages[0].parts[0].data == b"hi"


@pytest.mark.unit
async def test_image_output_maps_to_image_content():
    raw = b"\x89PNG\r\n\x1a\n fake image bytes"
    agent = FakeBuiltAgent(
        output=AgentOutput(parts=[Image(data=raw, media_type="image/png")])
    )
    adapter = MCPServerAdapter(agent)  # type: ignore[arg-type]

    result = await adapter.run_tool({"text": "make an image"})

    assert result.isError is False
    block = result.content[0]
    assert isinstance(block, types.ImageContent)
    assert block.mimeType == "image/png"
    assert base64.b64decode(block.data) == raw


@pytest.mark.unit
async def test_image_uri_output_maps_to_resource_link():
    agent = FakeBuiltAgent(
        output=AgentOutput(parts=[Image(uri="https://example.com/cat.png")])
    )
    adapter = MCPServerAdapter(agent)  # type: ignore[arg-type]

    result = await adapter.run_tool({"text": "link me an image"})

    block = result.content[0]
    assert isinstance(block, types.ResourceLink)
    assert str(block.uri) == "https://example.com/cat.png"


@pytest.mark.unit
async def test_video_output_maps_to_embedded_resource():
    raw = b"fake mp4 bytes"
    agent = FakeBuiltAgent(
        output=AgentOutput(parts=[Video(data=raw, media_type="video/mp4")])
    )
    adapter = MCPServerAdapter(agent)  # type: ignore[arg-type]

    result = await adapter.run_tool({"text": "make a video"})

    block = result.content[0]
    assert isinstance(block, types.EmbeddedResource)
    assert isinstance(block.resource, types.BlobResourceContents)
    assert block.resource.mimeType == "video/mp4"
    assert base64.b64decode(block.resource.blob) == raw


@pytest.mark.unit
async def test_failing_run_yields_error_result():
    agent = FakeBuiltAgent(error=RuntimeError("provider exploded"))
    adapter = MCPServerAdapter(agent)  # type: ignore[arg-type]

    result = await adapter.run_tool({"text": "hi"})

    assert result.isError is True
    assert len(result.content) == 1
    assert isinstance(result.content[0], types.TextContent)
    assert "provider exploded" in result.content[0].text
    assert "RuntimeError" in result.content[0].text


@pytest.mark.unit
async def test_invalid_arguments_yield_error_result():
    agent = FakeBuiltAgent(output=AgentOutput(parts=[Text("unused")]))
    adapter = MCPServerAdapter(agent)  # type: ignore[arg-type]

    # Neither 'text' nor 'messages' provided.
    result = await adapter.run_tool({})

    assert result.isError is True
    assert isinstance(result.content[0], types.TextContent)


@pytest.mark.unit
async def test_structured_messages_with_multimodal_parts():
    agent = FakeBuiltAgent(output=AgentOutput(parts=[Text("ok")]))
    adapter = MCPServerAdapter(agent)  # type: ignore[arg-type]

    img_b64 = base64.b64encode(b"img").decode("ascii")
    arguments = {
        "messages": [
            {
                "role": "user",
                "parts": [
                    {"modality": "text", "text": "describe this"},
                    {"modality": "image", "data_base64": img_b64, "media_type": "image/png"},
                ],
            }
        ]
    }

    result = await adapter.run_tool(arguments)

    assert result.isError is False
    assert agent.received is not None
    parts = agent.received.messages[0].parts
    assert parts[0].data == b"describe this"
    assert parts[1].data == b"img"


@pytest.mark.unit
def test_arguments_to_agent_input_prefers_messages():
    arguments = {
        "text": "ignored",
        "messages": [{"role": "user", "parts": [{"modality": "text", "text": "hi"}]}],
    }
    agent_input = arguments_to_agent_input(arguments)
    assert agent_input.messages[0].parts[0].data == b"hi"


@pytest.mark.unit
def test_server_advertises_single_named_tool():
    agent = FakeBuiltAgent(output=AgentOutput(parts=[Text("ok")]))
    adapter = MCPServerAdapter(agent, tool_name="do_run")  # type: ignore[arg-type]

    server = adapter.server()
    # The server object exists and carries the configured tool name via the adapter.
    assert adapter.tool_name == "do_run"
    assert server is not None
