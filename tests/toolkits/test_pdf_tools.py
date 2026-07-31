"""Unit tests for PDFTools.

Tests cover:
- ImportError when pypdf is missing
- read_pdf with a sample PDF file
- search_pdf returning correct pages and excerpts
- Error handling for invalid PDF files
- Pages parameter (single page, range, comma-separated)

**Validates: Requirements 4.3, 4.4**
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from loomable.toolkits.pdf_tools import PDFTools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_pdf_page_content(text: str) -> bytes:
    """Create PDF content stream bytes for a page with the given text."""
    # Escape parentheses in text for PDF string literals
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"BT /F1 12 Tf 100 700 Td ({escaped}) Tj ET".encode()


def _make_pdf_with_pages(page_texts: list[str]) -> bytes:
    """Create a multi-page PDF in memory with extractable text on each page.

    Args:
        page_texts: List of text strings, one per page.

    Returns:
        PDF file content as bytes.
    """
    writer = PdfWriter()

    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)

        # Create content stream with text
        stream = DecodedStreamObject()
        stream.set_data(_create_pdf_page_content(text))

        # Create font dictionary
        font_dict = DictionaryObject()
        font_dict[NameObject("/Type")] = NameObject("/Font")
        font_dict[NameObject("/Subtype")] = NameObject("/Type1")
        font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")

        # Create resources dict
        resources = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_dict
        resources[NameObject("/Font")] = fonts

        # Assign to page
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = stream

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pdf_tools() -> PDFTools:
    """Create a PDFTools instance."""
    return PDFTools()


@pytest.fixture
def single_page_pdf(tmp_path: Path) -> Path:
    """Create a single-page PDF with known text content."""
    pdf_bytes = _make_pdf_with_pages(["Hello World from loomable PDF tools"])
    pdf_path = tmp_path / "single.pdf"
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    """Create a 3-page PDF with distinct content on each page."""
    pdf_bytes = _make_pdf_with_pages([
        "First page content about testing",
        "Second page discusses integration",
        "Third page covers deployment strategies",
    ])
    pdf_path = tmp_path / "multi.pdf"
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path


# ---------------------------------------------------------------------------
# Test: ImportError when pypdf is missing
# ---------------------------------------------------------------------------


class TestImportError:
    """Test that PDFTools raises ImportError when pypdf is not installed."""

    def test_raises_import_error_when_pypdf_missing(self) -> None:
        """PDFTools() raises ImportError with install instruction when pypdf is absent."""
        with patch.dict(sys.modules, {"pypdf": None}):
            with pytest.raises(ImportError, match="pip install loomable\\[pdf\\]"):
                PDFTools()

    def test_import_error_message_contains_package_name(self) -> None:
        """The ImportError message mentions 'pypdf' as the missing package."""
        with patch.dict(sys.modules, {"pypdf": None}):
            with pytest.raises(ImportError, match="pypdf"):
                PDFTools()


# ---------------------------------------------------------------------------
# Test: read_pdf with sample PDF
# ---------------------------------------------------------------------------


class TestReadPdf:
    """Test read_pdf extracts text correctly from PDF files."""

    async def test_read_single_page_pdf(
        self, pdf_tools: PDFTools, single_page_pdf: Path
    ) -> None:
        """read_pdf returns extracted text from a single-page PDF."""
        result = await pdf_tools._read_pdf(str(single_page_pdf))

        assert "Hello World" in result
        assert "loomable" in result
        assert "Page 1" in result

    async def test_read_multi_page_pdf_all_pages(
        self, pdf_tools: PDFTools, multi_page_pdf: Path
    ) -> None:
        """read_pdf with no pages param returns text from all pages."""
        result = await pdf_tools._read_pdf(str(multi_page_pdf))

        assert "First page" in result
        assert "Second page" in result
        assert "Third page" in result
        assert "Page 1" in result
        assert "Page 2" in result
        assert "Page 3" in result

    async def test_read_pdf_single_page_param(
        self, pdf_tools: PDFTools, multi_page_pdf: Path
    ) -> None:
        """read_pdf with pages='2' returns only page 2 content."""
        result = await pdf_tools._read_pdf(str(multi_page_pdf), pages="2")

        assert "Second page" in result
        assert "Page 2" in result
        # Should NOT include other pages
        assert "First page" not in result
        assert "Third page" not in result

    async def test_read_pdf_page_range(
        self, pdf_tools: PDFTools, multi_page_pdf: Path
    ) -> None:
        """read_pdf with pages='1-2' returns pages 1 and 2."""
        result = await pdf_tools._read_pdf(str(multi_page_pdf), pages="1-2")

        assert "First page" in result
        assert "Second page" in result
        assert "Third page" not in result

    async def test_read_pdf_comma_separated_pages(
        self, pdf_tools: PDFTools, multi_page_pdf: Path
    ) -> None:
        """read_pdf with pages='1,3' returns pages 1 and 3 but not 2."""
        result = await pdf_tools._read_pdf(str(multi_page_pdf), pages="1,3")

        assert "First page" in result
        assert "Third page" in result
        assert "Second page" not in result


# ---------------------------------------------------------------------------
# Test: search_pdf
# ---------------------------------------------------------------------------


class TestSearchPdf:
    """Test search_pdf returns correct pages and excerpts."""

    async def test_search_finds_matching_text(
        self, pdf_tools: PDFTools, multi_page_pdf: Path
    ) -> None:
        """search_pdf returns results when query matches page content."""
        result = await pdf_tools._search_pdf(str(multi_page_pdf), "integration")

        assert "Page 2" in result
        assert "integration" in result

    async def test_search_returns_excerpt(
        self, pdf_tools: PDFTools, multi_page_pdf: Path
    ) -> None:
        """search_pdf includes a text excerpt around the match."""
        result = await pdf_tools._search_pdf(str(multi_page_pdf), "deployment")

        assert "Page 3" in result
        assert "deployment" in result

    async def test_search_case_insensitive(
        self, pdf_tools: PDFTools, multi_page_pdf: Path
    ) -> None:
        """search_pdf is case-insensitive."""
        result = await pdf_tools._search_pdf(str(multi_page_pdf), "TESTING")

        assert "Page 1" in result

    async def test_search_no_matches(
        self, pdf_tools: PDFTools, multi_page_pdf: Path
    ) -> None:
        """search_pdf returns a 'no matches' message when query isn't found."""
        result = await pdf_tools._search_pdf(
            str(multi_page_pdf), "xyznonexistent"
        )

        assert "No matches found" in result

    async def test_search_multiple_pages_match(
        self, pdf_tools: PDFTools, tmp_path: Path
    ) -> None:
        """search_pdf returns all pages that contain the query."""
        pdf_bytes = _make_pdf_with_pages([
            "The quick brown fox",
            "A lazy dog sleeps",
            "Another quick rabbit",
        ])
        pdf_path = tmp_path / "search_multi.pdf"
        pdf_path.write_bytes(pdf_bytes)

        result = await pdf_tools._search_pdf(str(pdf_path), "quick")

        assert "Page 1" in result
        assert "Page 3" in result
        # Page 2 doesn't contain "quick"
        assert "Page 2" not in result


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling for invalid PDF files and missing files."""

    async def test_read_nonexistent_file(self, pdf_tools: PDFTools) -> None:
        """read_pdf returns error for a file that doesn't exist."""
        result = await pdf_tools._read_pdf("/nonexistent/path/fake.pdf")

        assert "Error" in result
        assert "not found" in result.lower() or "File not found" in result

    async def test_search_nonexistent_file(self, pdf_tools: PDFTools) -> None:
        """search_pdf returns error for a file that doesn't exist."""
        result = await pdf_tools._search_pdf(
            "/nonexistent/path/fake.pdf", "query"
        )

        assert "Error" in result
        assert "not found" in result.lower() or "File not found" in result

    async def test_read_invalid_pdf_file(
        self, pdf_tools: PDFTools, tmp_path: Path
    ) -> None:
        """read_pdf returns error for a file that isn't a valid PDF."""
        invalid_pdf = tmp_path / "not_a_pdf.pdf"
        invalid_pdf.write_text("This is just a text file, not a PDF")

        result = await pdf_tools._read_pdf(str(invalid_pdf))

        assert "Error" in result
        assert "Invalid PDF" in result or "invalid" in result.lower()

    async def test_search_invalid_pdf_file(
        self, pdf_tools: PDFTools, tmp_path: Path
    ) -> None:
        """search_pdf returns error for a file that isn't a valid PDF."""
        invalid_pdf = tmp_path / "not_a_pdf.pdf"
        invalid_pdf.write_text("This is just plain text, not a PDF")

        result = await pdf_tools._search_pdf(str(invalid_pdf), "text")

        assert "Error" in result
        assert "Invalid PDF" in result or "invalid" in result.lower()

    async def test_read_empty_file_as_pdf(
        self, pdf_tools: PDFTools, tmp_path: Path
    ) -> None:
        """read_pdf returns error for an empty file with .pdf extension."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")

        result = await pdf_tools._read_pdf(str(empty_pdf))

        assert "Error" in result
