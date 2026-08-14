"""Unit tests for CitationTools."""

from __future__ import annotations

import json

import pytest

from loomable.toolkits.citation_tools import CitationStore, CitationTools


def _content(result) -> str:  # noqa: ANN001
    if getattr(result, "error", None):
        raise AssertionError(result.error)
    return str(result.content)


@pytest.mark.asyncio
async def test_citation_register_list_bibliography(tmp_path) -> None:
    tools = CitationTools(workspace=tmp_path)
    by_name = {t.name: t for t in tools.tools()}

    out = json.loads(
        _content(
            await by_name["register_source"].invoke(
                {
                    "url": "https://example.com/a",
                    "title": "Alpha",
                    "summary": "First source",
                    "quote": "quote-a",
                }
            )
        )
    )
    assert out["ok"] is True
    assert out["source"]["id"] == "S1"

    # Upsert same URL
    await by_name["register_source"].invoke(
        {"url": "https://example.com/a", "summary": "Updated summary"}
    )
    listed = json.loads(_content(await by_name["list_sources"].invoke({})))
    assert len(listed["sources"]) == 1
    assert listed["sources"][0]["summary"] == "Updated summary"

    await by_name["register_source"].invoke(
        {"url": "https://example.com/b", "title": "Beta"}
    )
    bib = _content(await by_name["format_bibliography"].invoke({}))
    assert "## Sources" in bib
    assert "Alpha" in bib
    assert "Beta" in bib
    assert (tmp_path / "sources.json").is_file()


def test_citation_store_rejects_empty_url(tmp_path) -> None:
    store = CitationStore(tmp_path / "sources.json")
    with pytest.raises(ValueError):
        store.register(url="")
