"""loomable.toolkits.pdf_tools - PDF reading and search toolkit.

Provides tools for extracting text from PDF files and searching PDF content.
Requires the ``pypdf`` package (install via ``pip install loomable[pdf]``).
"""

from __future__ import annotations

import asyncio

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit


class PDFTools(Toolkit):
    """PDF reading and search toolkit. Requires: loomable[pdf]"""

    def __init__(
        self,
        *,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        try:
            import pypdf  # noqa: F401
        except ImportError:
            raise ImportError(
                "PDFTools requires 'pypdf'. Install with: pip install loomable[pdf]"
            )
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)

    def _register_tools(self) -> list[FunctionTool]:
        """Return all FunctionTool instances this toolkit provides."""
        return [
            FunctionTool(self._read_pdf, name="read_pdf"),
            FunctionTool(self._search_pdf, name="search_pdf"),
        ]

    def _parse_pages(self, pages: str, total_pages: int) -> list[int]:
        """Parse a pages specification string into a list of 0-indexed page numbers.

        Supported formats:
        - "3" → [2] (single page, 1-indexed input → 0-indexed output)
        - "1-5" → [0, 1, 2, 3, 4]
        - "1,3,5" → [0, 2, 4]

        Returns None-equivalent (all pages) on invalid format by returning
        the full range.
        """
        pages = pages.strip()
        result: list[int] = []

        try:
            if "," in pages:
                # Comma-separated: "1,3,5"
                for part in pages.split(","):
                    p = int(part.strip())
                    idx = p - 1
                    if 0 <= idx < total_pages:
                        result.append(idx)
            elif "-" in pages:
                # Range: "1-5"
                parts = pages.split("-", 1)
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                for p in range(start, end + 1):
                    idx = p - 1
                    if 0 <= idx < total_pages:
                        result.append(idx)
            else:
                # Single page: "3"
                p = int(pages)
                idx = p - 1
                if 0 <= idx < total_pages:
                    result.append(idx)
        except (ValueError, IndexError):
            # Invalid format → return all pages with a note
            return list(range(total_pages))

        if not result:
            # No valid pages matched → return all pages
            return list(range(total_pages))

        return result

    async def _read_pdf(self, path: str, pages: str | None = None) -> str:
        """Extract text from a PDF file. Optionally specify page range (e.g. '1-5', '3', '1,3,5')."""
        return await asyncio.to_thread(self._read_pdf_sync, path, pages)

    def _read_pdf_sync(self, path: str, pages: str | None = None) -> str:
        """Synchronous implementation of PDF text extraction."""
        import pypdf

        try:
            reader = pypdf.PdfReader(path)
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as exc:
            return f"Error: Invalid PDF file: {path}: {exc}"

        total_pages = len(reader.pages)

        if pages is not None:
            page_indices = self._parse_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))

        parts: list[str] = []
        for idx in page_indices:
            page_num = idx + 1  # 1-indexed for display
            text = reader.pages[idx].extract_text() or ""
            parts.append(f"--- Page {page_num} ---\n{text}")

        return "\n\n".join(parts)

    async def _search_pdf(self, path: str, query: str) -> str:
        """Search a PDF for text matching the query. Returns matching page numbers and excerpts."""
        return await asyncio.to_thread(self._search_pdf_sync, path, query)

    def _search_pdf_sync(self, path: str, query: str) -> str:
        """Synchronous implementation of PDF text search."""
        import pypdf

        try:
            reader = pypdf.PdfReader(path)
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as exc:
            return f"Error: Invalid PDF file: {path}: {exc}"

        query_lower = query.lower()
        matches: list[str] = []

        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text_lower = text.lower()

            pos = text_lower.find(query_lower)
            if pos != -1:
                # Extract excerpt: 50 chars before + match + 50 chars after
                start = max(0, pos - 50)
                end = min(len(text), pos + len(query) + 50)
                excerpt = text[start:end].replace("\n", " ")
                page_num = idx + 1
                matches.append(f"Page {page_num}: ...{excerpt}...")

        if not matches:
            return f"No matches found for '{query}' in {path}"

        return "\n".join(matches)
