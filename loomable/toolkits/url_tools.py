"""loomable.toolkits.url_tools - URL fetching and text extraction toolkit.

Provides fetch_url and extract_text tools using httpx for async HTTP and
BeautifulSoup for HTML parsing. Requires: pip install loomable[url]
"""

from __future__ import annotations

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit
from loomable.toolkits.net_safety import validate_http_url, validate_redirect_target

_DEFAULT_UA = "loomable-research-agent/0.1 (+https://github.com/veerupandey/loomable)"
_MAX_REDIRECTS = 8


def _truncate(text: str, max_length: int | None) -> str:
    if max_length is None or len(text) <= max_length:
        return text
    omitted = len(text) - max_length
    marker = f"\n\n...[truncated {omitted} chars by URLTools]"
    keep = max_length - len(marker)
    if keep <= 0:
        return text[:max_length]
    return text[:keep] + marker


class URLTools(Toolkit):
    """URL fetching and text extraction toolkit. Requires: loomable[url]

    Fetches web pages asynchronously via httpx and extracts clean readable
    text using BeautifulSoup. Supports configurable timeout and max_length
    truncation. Private/loopback hosts are blocked (SSRF); redirects are
    re-validated hop-by-hop.
    """

    def __init__(
        self,
        *,
        timeout: int = 30,
        max_length: int | None = None,
        user_agent: str = _DEFAULT_UA,
        block_private_hosts: bool = True,
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
        self._user_agent = user_agent
        self._block_private_hosts = block_private_hosts

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._fetch_url, name="fetch_url"),
            FunctionTool(self._extract_text, name="extract_text"),
        ]

    def _validate_url(self, url: str) -> str | None:
        return validate_http_url(url, block_private_hosts=self._block_private_hosts)

    async def _get(self, url: str):
        """GET with hop-by-hop SSRF checks on redirects. Returns (error, response)."""
        import httpx

        err = self._validate_url(url)
        if err:
            return err, None
        current = url
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                headers={"User-Agent": self._user_agent},
            ) as client:
                for _ in range(_MAX_REDIRECTS):
                    err = self._validate_url(current)
                    if err:
                        return err, None
                    response = await client.get(current)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        loc = (response.headers or {}).get("location", "")
                        if not loc:
                            return (
                                f"Error: HTTP {response.status_code} for URL: {url}",
                                None,
                            )
                        hop_err, next_url = validate_redirect_target(
                            current,
                            loc,
                            block_private_hosts=self._block_private_hosts,
                        )
                        if hop_err:
                            return hop_err, None
                        current = next_url or current
                        continue
                    return None, response
                return "Error: too many redirects", None
        except httpx.TimeoutException:
            return f"Error: Request timed out after {self._timeout} seconds: {url}", None
        except httpx.HTTPError as exc:
            return f"Error: Request failed: {exc}", None
        except Exception as exc:  # noqa: BLE001
            return f"Error: Failed to fetch URL: {exc}", None

    async def _fetch_url(self, url: str) -> str:
        """Fetch raw HTML content from a URL."""
        err, response = await self._get(url)
        if err:
            return err
        assert response is not None
        if response.status_code < 200 or response.status_code >= 300:
            return f"Error: HTTP {response.status_code} for URL: {url}"
        return _truncate(response.text, self._max_length)

    async def _extract_text(self, url: str) -> str:
        """Fetch a URL and return clean readable text (HTML tags stripped)."""
        from bs4 import BeautifulSoup

        err, response = await self._get(url)
        if err:
            return err
        assert response is not None
        if response.status_code < 200 or response.status_code >= 300:
            return f"Error: HTTP {response.status_code} for URL: {url}"
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return _truncate(text, self._max_length)
