"""Build extractable multi-page PDFs for ingest / retrieval tests."""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(text: str, width: int = 90) -> list[str]:
    words = text.split()
    lines: list[str] = []
    buf = ""
    for w in words:
        cand = f"{buf} {w}".strip()
        if len(cand) <= width:
            buf = cand
        else:
            if buf:
                lines.append(buf)
            buf = w
    if buf:
        lines.append(buf)
    return lines or [""]


def write_pdf(path: Path, pages: list[str], *, max_lines: int = 400) -> Path:
    """Write a PDF where each page's text is extractable via pypdf."""
    writer = PdfWriter()
    font_dict = DictionaryObject()
    font_dict[NameObject("/Type")] = NameObject("/Font")
    font_dict[NameObject("/Subtype")] = NameObject("/Type1")
    font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")

    for page_text in pages:
        page = writer.add_blank_page(width=612, height=792)
        lines = _wrap(page_text)[:max_lines]
        cmds = ["BT /F1 10 Tf"]
        for i, line in enumerate(lines):
            esc = _escape(line)
            if i == 0:
                cmds.append(f"50 760 Td ({esc}) Tj")
            else:
                cmds.append(f"0 -13 Td ({esc}) Tj")
        cmds.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(cmds).encode("latin-1", errors="replace"))
        resources = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_dict
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = stream

    buf = io.BytesIO()
    writer.write(buf)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return path


_FILLER = (
    "The operations handbook describes routine capacity planning, on-call rotation, "
    "change management windows, and service-level objectives for regional clusters. "
    "Weather notes and cafeteria menus are mixed in as distractors. Bake sourdough at "
    "230C. Lisbon is mild in spring. This paragraph is repeated to simulate a dense "
    "enterprise PDF with little signal per page. "
)


def large_handbook_pages(*, n_pages: int = 80) -> list[str]:
    """~80-page handbook with planted facts at known pages (1-indexed)."""
    pages: list[str] = []
    for i in range(1, n_pages + 1):
        body = f"Chapter filler page {i}. " + (_FILLER * 3)
        if i == 3:
            body += (
                " PLANTED AUTH: Clients authenticate with OAuth2 bearer tokens. "
                "Registered client id AUTH-PLANT-A1 must never appear in query strings."
            )
        elif i == 12:
            # Intentionally huge so naive 8k truncation would drop the plant.
            body = (_FILLER * 40) + (
                " PLANTED SKU: Enterprise volume pricing uses SKU-PDF-8821 "
                "for the audit-log add-on. Quote this SKU on invoices."
            )
        elif i == 41:
            body += (
                " PLANTED INCIDENT: After a webhook leak rotate KEY-WHSEC-9901 "
                "within one hour and page the security on-call."
            )
        elif i == 70:
            body += (
                " PLANTED LATENCY: The ap-south region p99 is 118 milliseconds "
                "on the gold tier."
            )
        pages.append(body)
    return pages
