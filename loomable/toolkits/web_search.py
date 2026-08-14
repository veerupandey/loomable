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
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results, 20))
        if self._provider == "tavily":
            return await self._search_tavily(query, max_results)
        return await self._search_duckduckgo(query, max_results)

    async def _search_duckduckgo(self, query: str, max_results: int) -> str:
        """Execute a search using DuckDuckGo, with resilient public fallbacks."""
        import warnings

        errors: list[str] = []

        # Prefer the renamed ``ddgs`` package; suppress the legacy rename warning
        # when falling back to ``duckduckgo_search``.
        ddgs_cls = None
        try:
            from ddgs import DDGS as ddgs_cls  # type: ignore[assignment]
        except ImportError:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*renamed to.*ddgs.*",
                        category=RuntimeWarning,
                    )
                    from duckduckgo_search import DDGS as ddgs_cls  # type: ignore[assignment]
            except ImportError:
                errors.append("ddgs/duckduckgo-search not installed")

        for candidate in self._query_variants(query):
            if ddgs_cls is not None:
                try:
                    ddgs = ddgs_cls()
                    results = await asyncio.to_thread(
                        ddgs.text, candidate, max_results=max_results
                    )
                    if results:
                        return self._format_results(
                            results, keys=("title", "href", "body")
                        )
                    errors.append(f"DuckDuckGo text empty for {candidate!r}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"DuckDuckGo text failed for {candidate!r}: {exc}")

            # Fallback 1: DuckDuckGo Instant Answer API (no key)
            ia = await self._search_duckduckgo_instant(candidate, max_results)
            if ia:
                return ia

            # Fallback 2: Wikipedia OpenSearch (stable for research demos)
            wiki = await self._search_wikipedia(candidate, max_results)
            if wiki:
                return wiki

        detail = "; ".join(errors[-6:]) if errors else "unknown"
        return (
            "No results found. "
            f"Tried DuckDuckGo + Instant Answer + Wikipedia ({detail}). "
            "Pass explicit URLs to extract_text, or configure Tavily."
        )

    @staticmethod
    def _query_variants(query: str) -> list[str]:
        """Broaden sparse queries so Instant Answer / Wikipedia can still hit."""
        q = " ".join((query or "").split())
        if not q:
            return [""]
        words = q.split()
        variants = [q]
        if len(words) > 4:
            variants.append(" ".join(words[:4]))
        if len(words) > 3:
            variants.append(" ".join(words[:3]))
        if len(words) > 2:
            variants.append(" ".join(words[:2]))
        if len(words) > 1:
            variants.append(words[0])
        # Preserve order, drop dupes
        return list(dict.fromkeys(v for v in variants if v))

    async def _search_duckduckgo_instant(self, query: str, max_results: int) -> str | None:
        try:
            import httpx
        except ImportError:
            return None
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_redirect": "1",
                        "no_html": "1",
                    },
                )
                if response.status_code >= 300:
                    return None
                data = response.json()
        except Exception:  # noqa: BLE001
            return None

        results: list[dict] = []
        abstract = (data.get("AbstractText") or "").strip()
        abs_url = (data.get("AbstractURL") or "").strip()
        heading = (data.get("Heading") or query).strip()
        if abstract and abs_url:
            results.append({"title": heading, "url": abs_url, "content": abstract})

        for topic in data.get("RelatedTopics") or []:
            if len(results) >= max_results:
                break
            if isinstance(topic, dict) and topic.get("FirstURL") and topic.get("Text"):
                results.append(
                    {
                        "title": str(topic.get("Text") or "")[:80],
                        "url": topic["FirstURL"],
                        "content": topic.get("Text") or "",
                    }
                )
            elif isinstance(topic, dict) and "Topics" in topic:
                for nested in topic.get("Topics") or []:
                    if len(results) >= max_results:
                        break
                    if isinstance(nested, dict) and nested.get("FirstURL"):
                        results.append(
                            {
                                "title": str(nested.get("Text") or "")[:80],
                                "url": nested["FirstURL"],
                                "content": nested.get("Text") or "",
                            }
                        )

        if not results:
            return None
        return self._format_results(results[:max_results], keys=("title", "url", "content"))

    async def _search_wikipedia(self, query: str, max_results: int) -> str | None:
        try:
            import httpx
        except ImportError:
            return None
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "opensearch",
                        "search": query,
                        "limit": max_results,
                        "namespace": 0,
                        "format": "json",
                    },
                    headers={"User-Agent": "loomable-research-agent/0.1"},
                )
                if response.status_code >= 300:
                    return None
                data = response.json()
        except Exception:  # noqa: BLE001
            return None

        # OpenSearch: [query, titles[], descriptions[], urls[]]
        if not isinstance(data, list) or len(data) < 4:
            return None
        titles, descs, urls = data[1], data[2], data[3]
        results = []
        for title, desc, url in zip(titles, descs, urls):
            results.append({"title": title, "url": url, "content": desc or title})
        if not results:
            return None
        return self._format_results(results, keys=("title", "url", "content"))

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
