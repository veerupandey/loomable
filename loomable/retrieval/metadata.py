"""Citation metadata + retrieve filters.

Hits always carry source/path/page/filename/media_type/document_id/corpus plus
any caller metadata attached at ingest. ``filters=`` is equality / membership
on those fields (post-filter; works on every vector engine).
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "CITATION_KEYS",
    "apply_document_metadata",
    "matches_filters",
    "shape_hit",
]

CITATION_KEYS = (
    "id",
    "content",
    "score",
    "document_id",
    "source",
    "path",
    "filename",
    "media_type",
    "page",
    "start_line",
    "end_line",
    "kind",
    "name",
    "corpus",
    "url",
    "title",
    "author",
    "tags",
)


def apply_document_metadata(doc: Any, extra: Mapping[str, Any] | None) -> None:
    """Merge extra metadata onto a Document (ingest-time)."""
    if not extra:
        return
    meta = dict(getattr(doc, "metadata", None) or {})
    for key, value in extra.items():
        if value is None:
            continue
        meta.setdefault(key, value)
    doc.metadata = meta


def matches_filters(hit: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    """True when *hit* satisfies equality / membership filters."""
    if not filters:
        return True
    for key, expected in filters.items():
        actual = _maybe_json(hit.get(key))
        expected_n = _maybe_json(expected)
        if isinstance(expected_n, (list, tuple, set)):
            exp = list(expected_n)
            if actual in exp:
                continue
            if isinstance(actual, (list, tuple, set)) and set(map(_norm, actual)) & set(
                map(_norm, exp)
            ):
                continue
            if _norm(actual) in set(map(_norm, exp)):
                continue
            return False
        if isinstance(actual, (list, tuple, set)):
            if expected_n in actual or _norm(expected_n) in set(map(_norm, actual)):
                continue
            return False
        if actual != expected_n and _norm(actual) != _norm(expected_n):
            return False
    return True


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in {"[", "{"}:
        try:
            import json

            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def shape_hit(row: Mapping[str, Any], *, score: float | None = None) -> dict[str, Any]:
    """Normalize a store row / chunk result into a citation hit (keep extras)."""
    hit = {k: v for k, v in dict(row).items() if k != "embedding"}
    if score is not None:
        hit["score"] = float(score)
    hit.setdefault("content", hit.get("text") or "")
    hit.setdefault("source", hit.get("path") or hit.get("url") or "")
    hit.setdefault("path", hit.get("source") or "")
    if "score" in hit:
        try:
            hit["score"] = float(hit["score"])
        except (TypeError, ValueError):
            hit["score"] = 0.0
    return hit


def _norm(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""
