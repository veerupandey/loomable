"""Unit tests for the API Tool runtime."""

from __future__ import annotations

import pytest
import httpx

from loomable.kernel.api_tool import APITool
from loomable.kernel.errors import APIToolError, APIToolTimeoutError
from loomable.kernel.models import ToolResult


def _make_spec(
    *,
    method: str = "GET",
    url: str = "https://example.com/api",
    headers: dict | None = None,
    timeout: float = 30.0,
    name: str = "test-api-tool",
    description: str = "A test API tool",
) -> dict:
    spec: dict = {
        "name": name,
        "method": method,
        "url": url,
        "timeout": timeout,
        "description": description,
    }
    if headers is not None:
        spec["headers"] = headers
    return spec


class TestAPIToolConstruction:
    """Tests for APITool initialization from spec."""

    def test_name_from_spec(self):
        spec = _make_spec(name="my-tool")
        tool = APITool(spec)
        assert tool.name == "my-tool"

    def test_description_from_spec(self):
        spec = _make_spec(description="Does something")
        tool = APITool(spec)
        assert tool.description == "Does something"

    def test_default_description_when_missing(self):
        spec = {"name": "t", "method": "GET", "url": "http://x", "timeout": 5.0}
        tool = APITool(spec)
        assert tool.description == ""


class TestAPIToolInvokeSuccess:
    """Tests for successful HTTP requests."""

    @pytest.mark.asyncio
    async def test_get_json_response(self, httpx_mock):
        spec = _make_spec(method="GET", url="https://api.example.com/data")
        tool = APITool(spec)

        httpx_mock.add_response(
            url="https://api.example.com/data",
            json={"key": "value"},
            status_code=200,
        )

        result = await tool.invoke({})

        assert isinstance(result, ToolResult)
        assert result.content == {"key": "value"}
        assert result.metadata["status_code"] == 200
        assert result.error is None

    @pytest.mark.asyncio
    async def test_post_with_body(self, httpx_mock):
        spec = _make_spec(method="POST", url="https://api.example.com/items")
        tool = APITool(spec)

        httpx_mock.add_response(
            url="https://api.example.com/items",
            json={"id": 42},
            status_code=201,
        )

        result = await tool.invoke({"body": {"name": "widget"}})

        assert result.content == {"id": 42}
        assert result.metadata["status_code"] == 201

        # Verify the body was sent
        request = httpx_mock.get_request()
        assert request.method == "POST"

    @pytest.mark.asyncio
    async def test_text_response(self, httpx_mock):
        spec = _make_spec(url="https://api.example.com/text")
        tool = APITool(spec)

        httpx_mock.add_response(
            url="https://api.example.com/text",
            text="Hello, world!",
            status_code=200,
            headers={"content-type": "text/plain"},
        )

        result = await tool.invoke({})

        assert result.content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_custom_headers_from_spec(self, httpx_mock):
        spec = _make_spec(
            url="https://api.example.com/auth",
            headers={"Authorization": "Bearer token123"},
        )
        tool = APITool(spec)

        httpx_mock.add_response(url="https://api.example.com/auth", json={})

        await tool.invoke({})

        request = httpx_mock.get_request()
        assert request.headers["authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_per_request_header_override(self, httpx_mock):
        spec = _make_spec(
            url="https://api.example.com/auth",
            headers={"Authorization": "Bearer original"},
        )
        tool = APITool(spec)

        httpx_mock.add_response(url="https://api.example.com/auth", json={})

        await tool.invoke({"headers": {"Authorization": "Bearer override"}})

        request = httpx_mock.get_request()
        assert request.headers["authorization"] == "Bearer override"

    @pytest.mark.asyncio
    async def test_query_params(self, httpx_mock):
        spec = _make_spec(url="https://api.example.com/search")
        tool = APITool(spec)

        httpx_mock.add_response(
            url=httpx.URL("https://api.example.com/search", params={"q": "test"}),
            json={"results": []},
        )

        result = await tool.invoke({"params": {"q": "test"}})

        assert result.content == {"results": []}


class TestAPIToolInvokeErrors:
    """Tests for HTTP error responses."""

    @pytest.mark.asyncio
    async def test_404_raises_api_tool_error(self, httpx_mock):
        spec = _make_spec(url="https://api.example.com/missing")
        tool = APITool(spec)

        httpx_mock.add_response(
            url="https://api.example.com/missing",
            status_code=404,
        )

        with pytest.raises(APIToolError) as exc_info:
            await tool.invoke({})

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_500_raises_api_tool_error(self, httpx_mock):
        spec = _make_spec(url="https://api.example.com/error")
        tool = APITool(spec)

        httpx_mock.add_response(
            url="https://api.example.com/error",
            status_code=500,
        )

        with pytest.raises(APIToolError) as exc_info:
            await tool.invoke({})

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_403_raises_api_tool_error(self, httpx_mock):
        spec = _make_spec(url="https://api.example.com/forbidden")
        tool = APITool(spec)

        httpx_mock.add_response(
            url="https://api.example.com/forbidden",
            status_code=403,
        )

        with pytest.raises(APIToolError) as exc_info:
            await tool.invoke({})

        assert exc_info.value.status_code == 403


class TestAPIToolTimeout:
    """Tests for request timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_raises_api_tool_timeout_error(self, httpx_mock):
        spec = _make_spec(
            url="https://api.example.com/slow",
            timeout=2.0,
            name="slow-tool",
        )
        tool = APITool(spec)

        httpx_mock.add_exception(
            httpx.ReadTimeout("timed out"),
            url="https://api.example.com/slow",
        )

        with pytest.raises(APIToolTimeoutError) as exc_info:
            await tool.invoke({})

        assert exc_info.value.tool_name == "slow-tool"
        assert exc_info.value.timeout == 2.0

    @pytest.mark.asyncio
    async def test_connect_timeout_raises_api_tool_timeout_error(self, httpx_mock):
        spec = _make_spec(
            url="https://api.example.com/unreachable",
            timeout=1.0,
            name="unreachable-tool",
        )
        tool = APITool(spec)

        httpx_mock.add_exception(
            httpx.ConnectTimeout("connect timed out"),
            url="https://api.example.com/unreachable",
        )

        with pytest.raises(APIToolTimeoutError) as exc_info:
            await tool.invoke({})

        assert exc_info.value.tool_name == "unreachable-tool"
        assert exc_info.value.timeout == 1.0
