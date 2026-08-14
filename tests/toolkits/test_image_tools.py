"""Unit tests for ImageTools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from loomable.agent import ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.toolkits.image_tools import ImageTools


def _content(result) -> str:  # noqa: ANN001
    if getattr(result, "error", None):
        raise AssertionError(result.error)
    return str(result.content)


def _mock_httpx_client(status_code: int, content: bytes, content_type: str = "image/png"):
    response = MagicMock()
    response.status_code = status_code
    response.content = content
    response.headers = {"content-type": content_type}
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_fetch_image_writes_workspace(tmp_path) -> None:
    tools = ImageTools(workspace=tmp_path, model=None)
    by_name = {t.name: t for t in tools.tools()}
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    with patch("httpx.AsyncClient", return_value=_mock_httpx_client(200, png)):
        out = json.loads(
            _content(
                await by_name["fetch_image"].invoke(
                    {"url": "https://cdn.example.com/cat.png"}
                )
            )
        )
    assert out["ok"] is True
    assert out["path"].startswith("images/")
    assert (tmp_path / out["path"]).is_file()
    assert (tmp_path / out["path"]).read_bytes() == png

    listed = json.loads(_content(await by_name["list_images"].invoke({})))
    assert out["path"] in listed["entries"]


@pytest.mark.asyncio
async def test_fetch_image_http_error(tmp_path) -> None:
    tools = ImageTools(workspace=tmp_path)
    fetch = next(t for t in tools.tools() if t.name == "fetch_image")
    with patch("httpx.AsyncClient", return_value=_mock_httpx_client(404, b"")):
        out = _content(await fetch.invoke({"url": "https://example.com/x.png"}))
    assert "404" in out


@pytest.mark.asyncio
async def test_analyze_image_with_scripted_model(tmp_path) -> None:
    class _Vision:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            # Ensure multimodal content reached the provider
            msgs = request.messages or []
            assert msgs
            return ModelResponse(content="A red square chart with rising trend.")

    img_path = tmp_path / "images" / "chart.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    tools = ImageTools(
        workspace=tmp_path,
        model=ModelSpec(provider="scripted", provider_impl=_Vision()),
    )
    analyze = next(t for t in tools.tools() if t.name == "analyze_image")
    out = json.loads(
        _content(
            await analyze.invoke(
                {"path": "images/chart.png", "question": "What does the chart show?"}
            )
        )
    )
    assert out["ok"] is True
    assert "rising trend" in out["analysis"]
    assert (tmp_path / out["note"]).is_file()
