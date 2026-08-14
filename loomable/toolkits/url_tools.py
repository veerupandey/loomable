"""loomable.toolkits.url_tools - URL fetching and text extraction toolkit.

Provides fetch_url and extract_text tools using httpx for async HTTP and
BeautifulSoup for HTML parsing. Requires: pip install loomable[url]
"""

from __future__ import annotations

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit


class URLTools(Toolkit):
    """URL fetching and text extraction toolkit. Requires: loomable[url]

    Fetches web pages asynchronously via httpx and extracts clean readable
    text using BeautifulSoup. Supports configurable timeout and max_length
    truncation.

    Usage::

        from loomable.toolkits import URLTools

        tools = URLTools(timeout=30, max_length=5000)
        # Or with defaults (30s timeout, no length limit):
        tools = URLTools()
    """

    def __init__(
        self,
        *,
        timeout: int = 30,
        max_length: int | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        try:
            import httpx  # noqa: F401
            import bs4  # noqa: F401
        except ImportError:
            raise ImportError(
                "URLTools requires 'httpx' and 'beautifulsoup4'. "
                "Install with: pip install loomable[url]"
            )
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._timeout = timeout
        self._max_length = max_length

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._fetch_url, name="fetch_url"),
            FunctionTool(self._extract_text, name="extract_text"),
        ]

    async def _fetch_url(self, url: str) -> str:
        """Fetch raw HTML content from a URL."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url)
                if response.status_code < 200 or response.status_code >= 300:
                    return (
                        f"Error: HTTP {response.status_code} for URL: {url}"
                    )
                text = response.text
                if self._max_length is not None and len(text) > self._max_length:
                    text = text[: self._max_length]
                return text
        except httpx.TimeoutException:
            return f"Error: Request timed out after {self._timeout} seconds: {url}"
        except httpx.HTTPError as exc:
            return f"Error: Request failed: {exc}"
        except Exception as exc:
            return f"Error: Failed to fetch URL: {exc}"

    async def _extract_text(self, url: str) -> str:
        """Fetch a URL and return clean readable text (HTML tags stripped)."""
        import httpx
        from bs4 import BeautifulSoup

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url)
                if response.status_code < 200 or response.status_code >= 300:
                    return (
                        f"Error: HTTP {response.status_code} for URL: {url}"
                    )
                html = response.text
        except httpx.TimeoutException:
            return f"Error: Request timed out after {self._timeout} seconds: {url}"
        except httpx.HTTPError as exc:
            return f"Error: Request failed: {exc}"
        except Exception as exc:
            return f"Error: Failed to fetch URL: {exc}"

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        if self._max_length is not None and len(text) > self._max_length:
            text = text[: self._max_length]

        return text
