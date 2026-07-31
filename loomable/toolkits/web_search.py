"""loomable.toolkits.web_search - Web search toolkit supporting multiple providers.

Provides a ``web_search`` tool that queries the web via DuckDuckGo (default,
no API key required) or Tavily (requires an API key). Results are returned as
formatted text with title, URL, and snippet for each result.

The ``duckduckgo-search`` package is lazily imported at call time since it is
an optional dependency (install via ``pip install loomable[web]``).
"""

from __future__ import annotations

import asyncio

from loomable.agent.errors import AgentConfigError
from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit


class WebSearchTools(Toolkit):
    """Web search toolkit supporting DuckDuckGo (default) and Tavily.

    DuckDuckGo requires no configuration. Tavily requires an ``api_key``.

    Usage::

        from loomable.toolkits import WebSearchTools

        # Zero-config DuckDuckGo (default):
        tools = WebSearchTools()

        # Tavily with API key:
        tools = WebSearchTools(provider="tavily", api_key="tvly-...")
    """

    def __init__(
        self,
        *,
        provider: str = "duckduckgo",
        api_key: str | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        if provider == "tavily" and not api_key:
            raise AgentConfigError("api_key (required for Tavily provider)")
        self._provider = provider
        self._api_key = api_key

    def _register_tools(self) -> list[FunctionTool]:
        return [FunctionTool(self._web_search, name="web_search")]

    async def _web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web and return results with title, URL, and snippet."""
        if self._provider == "tavily":
            return await self._search_tavily(query, max_results)
        return await self._search_duckduckgo(query, max_results)

    async def _search_duckduckgo(self, query: str, max_results: int) -> str:
        """Execute a search using the DuckDuckGo backend."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return (
                "Error: Web search failed: duckduckgo-search package not installed. "
                "Install with: pip install loomable[web]"
            )

        try:
            ddgs = DDGS()
            results = await asyncio.to_thread(
                ddgs.text, query, max_results=max_results
            )
        except Exception as exc:  # noqa: BLE001
            return f"Error: Web search failed: {exc}"

        return self._format_results(results, keys=("title", "href", "body"))

    async def _search_tavily(self, query: str, max_results: int) -> str:
        """Execute a search using the Tavily API."""
        try:
            import httpx
        except ImportError:
            return (
                "Error: Web search failed: httpx package not installed. "
                "Install with: pip install httpx"
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self._api_key,
                        "query": query,
                        "max_results": max_results,
                    },
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
        except Exception as exc:  # noqa: BLE001
            return f"Error: Web search failed: {exc}"

        return self._format_results(results, keys=("title", "url", "content"))

    @staticmethod
    def _format_results(
        results: list[dict],
        *,
        keys: tuple[str, str, str],
    ) -> str:
        """Format search results as a numbered list with title, URL, and snippet."""
        if not results:
            return "No results found."

        title_key, url_key, snippet_key = keys
        lines: list[str] = []
        for i, result in enumerate(results, start=1):
            title = result.get(title_key, "No title")
            url = result.get(url_key, "No URL")
            snippet = result.get(snippet_key, "No snippet")
            lines.append(f"{i}. {title}")
            lines.append(f"   URL: {url}")
            lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines).rstrip()
