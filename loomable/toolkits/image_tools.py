"""Image toolkit for multimodal research deep agents.

Downloads images into a workspace directory and analyzes them with a vision-capable
model. Complements ``URLTools`` / ``WebSearchTools`` for real research loops.
"""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit
from loomable.toolkits.net_safety import validate_http_url, validate_redirect_target

__all__ = ["ImageTools"]

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")
_MAX_REDIRECTS = 8
_DEFAULT_MAX_BYTES = 8_000_000


def _guess_ext(url: str, content_type: str | None) -> str:
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}:
        return suffix
    return ".img"


def _safe_filename(url: str, content_type: str | None) -> str:
    path = urlparse(url).path
    base = Path(path).name or "image"
    base = _SAFE_NAME.sub("_", base).strip("._") or "image"
    if "." not in base:
        base = base + _guess_ext(url, content_type)
    return base[:120]


class ImageTools(Toolkit):
    """Multimodal research tools: ``fetch_image``, ``analyze_image``, ``list_images``.

    Usage::

        from loomable.toolkits import ImageTools

        tools = ImageTools(workspace=\"./.deep_workspace\", model=provider)
    """

    def __init__(
        self,
        workspace: str | Path = "./.deep_workspace",
        *,
        model: Any = None,
        timeout: int = 30,
        images_dir: str = "images",
        max_bytes: int = _DEFAULT_MAX_BYTES,
        block_private_hosts: bool = True,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ImageTools requires 'httpx'. Install with: pip install httpx"
            ) from exc
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._root = Path(workspace)
        self._root.mkdir(parents=True, exist_ok=True)
        self._images = self._root / images_dir
        self._images.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._timeout = timeout
        self._max_bytes = max(1024, int(max_bytes))
        self._block_private_hosts = block_private_hosts

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._fetch_image, name="fetch_image"),
            FunctionTool(self._analyze_image, name="analyze_image"),
            FunctionTool(self._list_images, name="list_images"),
            FunctionTool(self._discover_images, name="discover_images"),
        ]

    def _resolve_path(self, path: str) -> Path | None:
        raw = (path or "").strip().replace("\\", "/")
        if not raw or ".." in raw.split("/"):
            return None
        raw = raw.lstrip("/")
        candidate = (self._root / raw).resolve()
        try:
            candidate.relative_to(self._root.resolve())
        except ValueError:
            return None
        return candidate

    async def _http_get_bytes(self, url: str) -> tuple[str | None, bytes | None, str]:
        """GET bytes with hop-by-hop SSRF checks. Returns (error, data, content_type)."""
        import httpx

        err = validate_http_url(url, block_private_hosts=self._block_private_hosts)
        if err:
            return err, None, ""
        current = url
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=False
            ) as client:
                for _ in range(_MAX_REDIRECTS):
                    err = validate_http_url(
                        current, block_private_hosts=self._block_private_hosts
                    )
                    if err:
                        return err, None, ""
                    response = await client.get(current)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        loc = (response.headers or {}).get("location", "")
                        if not loc:
                            return (
                                f"Error: HTTP {response.status_code} for URL: {url}",
                                None,
                                "",
                            )
                        hop_err, next_url = validate_redirect_target(
                            current,
                            loc,
                            block_private_hosts=self._block_private_hosts,
                        )
                        if hop_err:
                            return hop_err, None, ""
                        current = next_url or current
                        continue
                    if response.status_code < 200 or response.status_code >= 300:
                        return (
                            f"Error: HTTP {response.status_code} for URL: {url}",
                            None,
                            "",
                        )
                    data = response.content
                    if len(data) > self._max_bytes:
                        return (
                            f"Error: image exceeds max_bytes={self._max_bytes}",
                            None,
                            "",
                        )
                    content_type = response.headers.get("content-type", "")
                    return None, data, content_type
                return "Error: too many redirects", None, ""
        except httpx.TimeoutException:
            return (
                f"Error: Request timed out after {self._timeout} seconds: {url}",
                None,
                "",
            )
        except httpx.HTTPError as exc:
            return f"Error: Request failed: {exc}", None, ""
        except Exception as exc:  # noqa: BLE001
            return f"Error: Failed to fetch image: {exc}", None, ""

    async def _fetch_image(self, url: str, path: str = "") -> str:
        """Download an image URL into the workspace and return its path + metadata."""
        url = (url or "").strip()
        if not url:
            return "Error: url is required"
        err, data, content_type = await self._http_get_bytes(url)
        if err:
            return err
        assert data is not None

        if path.strip():
            dest = self._resolve_path(path.strip())
            if dest is None:
                return "Error: invalid path (must stay inside workspace)"
            dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            name = _safe_filename(url, content_type)
            dest = self._images / name
            # Avoid clobbering: suffix if exists
            if dest.exists():
                stem, suffix = dest.stem, dest.suffix
                n = 2
                while dest.exists():
                    dest = self._images / f"{stem}_{n}{suffix}"
                    n += 1

        dest.write_bytes(data)
        rel = str(dest.relative_to(self._root)).replace("\\", "/")
        meta = {
            "ok": True,
            "path": rel,
            "bytes": len(data),
            "content_type": content_type.split(";")[0].strip() if content_type else "",
            "url": url,
        }
        return json.dumps(meta, ensure_ascii=False)

    async def _analyze_image(self, path: str, question: str = "") -> str:
        """Analyze a workspace image with the vision model and save a short note."""
        if self._model is None:
            return "Error: ImageTools was created without a model; cannot analyze_image"
        resolved = self._resolve_path(path)
        if resolved is None or not resolved.is_file():
            return f"Error: image not found in workspace: {path}"

        question = (question or "Describe this image in detail for research notes.").strip()
        from loomable.agent.builder import Agent
        from loomable.agent.media import image as image_part

        agent = Agent(
            model=self._model,
            name="image-analyst",
            role="Image Analyst",
            goal="Describe images accurately for research",
            instructions=(
                "You analyze images for research. Be concrete: objects, text in the "
                "image, charts, and anything relevant to the question. Avoid fluff."
            ),
            modalities="text+image",
            think_tool=False,
            max_tool_iterations=1,
        )
        try:
            result = await agent.arun(question, images=[image_part(path=resolved)])
            text = (result.output.text() or "").strip() or "(no analysis)"
        except Exception as exc:  # noqa: BLE001 — isolate vision failures
            return f"Error: image analysis failed: {exc}"

        note_rel = f"images/analysis_{resolved.stem}.md"
        note_path = self._root / note_rel
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            f"# Image analysis: {path}\n\n**Question:** {question}\n\n{text}\n",
            encoding="utf-8",
        )
        return json.dumps(
            {"ok": True, "path": path, "note": note_rel, "analysis": text},
            ensure_ascii=False,
        )

    async def _list_images(self, dir: str = "images") -> str:  # noqa: A002
        """List image files under a workspace subdirectory (default: images/)."""
        target = self._resolve_path(dir or "images")
        if target is None:
            return json.dumps({"error": "invalid path", "entries": []})
        if not target.exists():
            return json.dumps({"entries": [], "dir": dir})
        entries: list[str] = []
        if target.is_file():
            entries.append(str(target.relative_to(self._root)).replace("\\", "/"))
        else:
            for p in sorted(target.rglob("*")):
                if p.is_file() and p.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".webp",
                    ".bmp",
                    ".svg",
                    ".img",
                }:
                    entries.append(str(p.relative_to(self._root)).replace("\\", "/"))
        return json.dumps({"dir": dir, "entries": entries}, ensure_ascii=False)

    async def _discover_images(self, url: str, limit: int = 8) -> str:
        """Fetch a page and return candidate image URLs (for fetch_image follow-up)."""
        from urllib.parse import urljoin

        url = (url or "").strip()
        if not url:
            return "Error: url is required"
        err = validate_http_url(url, block_private_hosts=self._block_private_hosts)
        if err:
            return err
        limit = max(1, min(int(limit) or 8, 20))
        # Reuse SSRF-safe GET; decode as text for scraping.
        get_err, data, _ctype = await self._http_get_bytes(url)
        if get_err:
            return f"Error: Failed to discover images: {get_err}"
        assert data is not None
        try:
            html = data.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return f"Error: Failed to discover images: {exc}"

        # Lightweight scrape — prefer BeautifulSoup when available
        found: list[str] = []
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all("img"):
                src = tag.get("src") or tag.get("data-src") or ""
                if not src or src.startswith("data:"):
                    continue
                found.append(urljoin(url, src))
                if len(found) >= limit:
                    break
        except ImportError:
            for match in re.finditer(
                r"""(?:src|data-src)\s*=\s*["']([^"']+\.(?:png|jpe?g|gif|webp|svg))["']""",
                html,
                flags=re.I,
            ):
                found.append(urljoin(url, match.group(1)))
                if len(found) >= limit:
                    break

        # De-dupe preserving order
        uniq: list[str] = []
        seen: set[str] = set()
        for u in found:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return json.dumps({"url": url, "images": uniq[:limit]}, ensure_ascii=False)
