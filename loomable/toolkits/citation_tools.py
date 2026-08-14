"""Citation / source registry for research deep agents.

Persists sources under ``{workspace}/sources.json`` so long-horizon agents can
cite URLs without stuffing the full page into the final answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit
from loomable.toolkits.net_safety import validate_http_url

__all__ = ["CitationStore", "CitationTools"]


class CitationStore:
    """JSON-backed list of research sources and optional claim→source links."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._sources: list[dict[str, Any]] = []
        self._claims: list[dict[str, Any]] = []
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._sources = [x for x in data if isinstance(x, dict)]
                elif isinstance(data, dict):
                    src = data.get("sources")
                    claims = data.get("claims")
                    if isinstance(src, list):
                        self._sources = [x for x in src if isinstance(x, dict)]
                    if isinstance(claims, list):
                        self._claims = [x for x in claims if isinstance(x, dict)]
            except (OSError, json.JSONDecodeError, TypeError):
                self._sources = []
                self._claims = []

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: Any
        if self._claims:
            payload = {"sources": self._sources, "claims": self._claims}
        else:
            # Keep backward-compatible list shape when no claims exist.
            payload = self._sources
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def register(
        self,
        *,
        url: str,
        title: str = "",
        summary: str = "",
        quote: str = "",
        verified: bool = False,
    ) -> dict[str, Any]:
        url = (url or "").strip()
        if not url:
            raise ValueError("url is required")
        err = validate_http_url(url, block_private_hosts=True)
        if err:
            raise ValueError(err.removeprefix("Error: ").strip() or err)
        # Upsert by URL
        for existing in self._sources:
            if str(existing.get("url") or "") == url:
                if title:
                    existing["title"] = title
                if summary:
                    existing["summary"] = summary
                if quote:
                    existing["quote"] = quote
                if verified:
                    existing["verified"] = True
                self._persist()
                return dict(existing)

        host = urlparse(url).netloc or ""
        entry = {
            "id": f"S{len(self._sources) + 1}",
            "url": url,
            "title": (title or host or url).strip(),
            "summary": (summary or "").strip(),
            "quote": (quote or "").strip(),
            "host": host,
            "verified": bool(verified),
        }
        self._sources.append(entry)
        self._persist()
        return dict(entry)

    def register_claim(
        self,
        *,
        claim: str,
        source_id: str,
        quote: str = "",
    ) -> dict[str, Any]:
        claim = (claim or "").strip()
        source_id = (source_id or "").strip()
        if not claim:
            raise ValueError("claim is required")
        if not source_id:
            raise ValueError("source_id is required")
        ids = {str(s.get("id") or "") for s in self._sources}
        if source_id not in ids:
            raise ValueError(f"unknown source_id: {source_id}")
        entry = {
            "id": f"C{len(self._claims) + 1}",
            "claim": claim,
            "source_id": source_id,
            "quote": (quote or "").strip(),
        }
        self._claims.append(entry)
        self._persist()
        return dict(entry)

    def list(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self._sources]

    def list_claims(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self._claims]

    def bibliography_markdown(self) -> str:
        if not self._sources:
            return "_No sources registered._"
        lines = ["## Sources"]
        for src in self._sources:
            sid = src.get("id") or "?"
            title = src.get("title") or src.get("url")
            url = src.get("url") or ""
            summary = src.get("summary") or ""
            quote = src.get("quote") or ""
            verified = " ✓" if src.get("verified") else ""
            block = f"- **[{sid}]**{verified} [{title}]({url})"
            if summary:
                block += f" — {summary}"
            lines.append(block)
            if quote:
                lines.append(f"  > {quote}")
        if self._claims:
            lines.append("")
            lines.append("## Claims")
            for c in self._claims:
                cid = c.get("id") or "?"
                sid = c.get("source_id") or "?"
                claim = c.get("claim") or ""
                quote = c.get("quote") or ""
                lines.append(f"- **[{cid}]** {claim} (basis: [{sid}])")
                if quote:
                    lines.append(f"  > {quote}")
        return "\n".join(lines) + "\n"


class CitationTools(Toolkit):
    """Research citations: register, verify, claim-link, bibliography."""

    def __init__(
        self,
        workspace: str | Path = "./.deep_workspace",
        *,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        root = Path(workspace)
        root.mkdir(parents=True, exist_ok=True)
        self._store = CitationStore(root / "sources.json")

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._register_source, name="register_source"),
            FunctionTool(self._verify_source, name="verify_source"),
            FunctionTool(self._register_claim, name="register_claim"),
            FunctionTool(self._list_sources, name="list_sources"),
            FunctionTool(self._format_bibliography, name="format_bibliography"),
        ]

    async def _register_source(
        self,
        url: str,
        title: str = "",
        summary: str = "",
        quote: str = "",
    ) -> str:
        """Register a web source for the final brief citations."""
        try:
            entry = self._store.register(
                url=url, title=title, summary=summary, quote=quote
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return json.dumps({"ok": True, "source": entry}, ensure_ascii=False)

    async def _verify_source(self, url: str = "", source_id: str = "") -> str:
        """HTTP-check a registered (or new) URL and mark it verified when reachable."""
        import httpx

        target = (url or "").strip()
        sid = (source_id or "").strip()
        if not target and sid:
            for src in self._store.list():
                if str(src.get("id") or "") == sid:
                    target = str(src.get("url") or "")
                    break
        if not target:
            return "Error: url or source_id is required"
        err = validate_http_url(target, block_private_hosts=True)
        if err:
            return err
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
                response = await client.head(target)
                if response.is_redirect:
                    # Accept redirects as reachable; registration still SSRF-guards URL.
                    status = response.status_code
                elif response.status_code >= 400:
                    # Some hosts reject HEAD — try GET.
                    response = await client.get(target)
                    status = response.status_code
                else:
                    status = response.status_code
        except Exception as exc:  # noqa: BLE001
            return f"Error: verify failed: {exc}"
        if status >= 400:
            return f"Error: HTTP {status} verifying {target}"
        entry = self._store.register(url=target, verified=True)
        return json.dumps(
            {"ok": True, "verified": True, "http_status": status, "source": entry},
            ensure_ascii=False,
        )

    async def _register_claim(
        self,
        claim: str,
        source_id: str,
        quote: str = "",
    ) -> str:
        """Link a concrete claim to a registered source id (e.g. S1)."""
        try:
            entry = self._store.register_claim(
                claim=claim, source_id=source_id, quote=quote
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return json.dumps({"ok": True, "claim": entry}, ensure_ascii=False)

    async def _list_sources(self) -> str:
        """List all registered research sources (and claims when present)."""
        return json.dumps(
            {"sources": self._store.list(), "claims": self._store.list_claims()},
            indent=2,
            ensure_ascii=False,
        )

    async def _format_bibliography(self) -> str:
        """Return a Markdown bibliography block for the deliverable."""
        return self._store.bibliography_markdown()
