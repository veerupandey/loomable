"""loomable.toolkits.url_tools - URL fetching and text extraction toolkit.

Provides fetch_url and extract_text tools using httpx for async HTTP and
BeautifulSoup for HTML parsing. Requires: pip install loomable[url]
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit

_DEFAULT_UA = "loomable-research-agent/0.1 (+https://github.com/veerupandey/loomable)"


def _is_blocked_host(hostname: str) -> bool:
    """Best-effort SSRF guard for localhost / private / link-local targets."""
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return True
    if host in {"localhost", "metadata.google.internal"} or host.endswith(".localhost"):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return True
    return False


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
        raw = (url or "").strip()
        if not raw:
            return "Error: url is required"
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            return f"Error: unsupported URL scheme: {parsed.scheme or '(none)'}"
        if self._block_private_hosts and _is_blocked_host(parsed.hostname or ""):
            return f"Error: blocked host (SSRF guard): {parsed.hostname}"
        return None

    async def _fetch_url(self, url: str) -> str:
        """Fetch raw HTML content from a URL."""
        import httpx

        err = self._validate_url(url)
        if err:
            return err
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": self._user_agent},
            ) as client:
                response = await client.get(url)
                if response.status_code < 200 or response.status_code >= 300:
                    return (
                        f"Error: HTTP {response.status_code} for URL: {url}"
                    )
                return _truncate(response.text, self._max_length)
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

        err = self._validate_url(url)
        if err:
            return err
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": self._user_agent},
            ) as client:
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
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return _truncate(text, self._max_length)
