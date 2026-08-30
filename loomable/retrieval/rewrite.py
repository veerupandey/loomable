"""Built-in pluggable query rewriters."""

from __future__ import annotations

from typing import Any

__all__ = [
    "IdentityRewriter",
    "MultiQueryRewriter",
    "HyDERewriter",
    "AgenticDecompositionRewriter",
    "resolve_rewriter",
]

# Token Jaccard threshold above which two queries are treated as near-duplicates.
_DEDUP_OVERLAP = 0.70
# Containment threshold: if one query's content words are mostly a subset of the
# other's, treat it as a restatement (not an orthogonal sub-question).
_DEDUP_CONTAINMENT = 0.80

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "is",
    "are", "was", "were", "be", "been", "being", "which", "what", "that", "this",
    "these", "those", "how", "why", "when", "where", "who", "does", "do", "did",
    "across", "all", "single", "method", "methods", "report", "reports",
    "reported", "number", "numbers", "compare", "comparing", "comparison",
})


def _tokenize(text: str) -> set[str]:
    import re

    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())}


def _content_tokens(text: str) -> set[str]:
    return _tokenize(text) - _STOPWORDS


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _containment(a: str, b: str) -> float:
    """Fraction of ``a``'s content words that also appear in ``b``."""
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _clean_line(line: str) -> str:
    """Strip numbering, bullets, and markdown emphasis from an LLM-emitted line."""
    import re

    s = (line or "").strip()
    # Leading "1.", "1)", "-", "*", "**", "•", "#", ">" markers.
    s = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*•#>]+\s*|\*\*)+", "", s)
    # Strip emphasis asterisks anywhere (e.g. "**bold** foo" -> "bold foo").
    s = re.sub(r"\*+", "", s)
    s = s.strip().strip('"').strip()
    return s


def _dedup_queries(
    queries: list[str],
    *,
    threshold: float = _DEDUP_OVERLAP,
    containment: float = _DEDUP_CONTAINMENT,
) -> list[str]:
    """Greedily drop near-duplicate queries by token overlap and containment."""
    kept: list[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        dup = False
        for k in kept:
            if _overlap(q, k) >= threshold:
                dup = True
                break
            if _containment(q, k) >= containment or _containment(k, q) >= containment:
                dup = True
                break
        if not dup:
            kept.append(q)
    return kept


class IdentityRewriter:
    """Pass-through (no LLM)."""

    name = "off"

    async def rewrite(self, query: str) -> list[str]:
        q = (query or "").strip()
        return [q] if q else []


class MultiQueryRewriter:
    """LLM generates alternate phrasings (LangChain MultiQuery pattern).

    ``llm`` must expose ``async complete(prompt: str) -> str`` **or** be an
    object with ``async complete(request) -> response`` compatible with
    Loomable providers. Prefer a simple callable::

        async def llm(prompt: str) -> str: ...
    """

    name = "multi_query"

    def __init__(
        self,
        llm: Any,
        *,
        n: int = 3,
        include_original: bool = True,
    ) -> None:
        self.llm = llm
        self.n = max(1, int(n))
        self.include_original = include_original

    async def _call_llm(self, prompt: str) -> str:
        llm = self.llm
        if callable(llm) and not hasattr(llm, "complete"):
            out = llm(prompt)
            if hasattr(out, "__await__"):
                out = await out
            return str(out or "")
        if hasattr(llm, "complete"):
            from loomable.kernel.models import ModelRequest

            resp = await llm.complete(
                ModelRequest(messages=[{"role": "user", "content": prompt}])
            )
            return str(getattr(resp, "content", None) or resp or "")
        raise TypeError("llm must be async callable(prompt)->str or have .complete()")

    async def rewrite(self, query: str) -> list[str]:
        q = (query or "").strip()
        if not q:
            return []
        prompt = (
            f"Generate {self.n} diverse search queries for retrieving documents "
            f"relevant to the user question. One query per line. No numbering.\n\n"
            f"Question: {q}\n"
        )
        raw = await self._call_llm(prompt)
        alts: list[str] = []
        for line in (raw or "").splitlines():
            line = _clean_line(line)
            if line and line.lower() != q.lower():
                alts.append(line)
        out: list[str] = []
        if self.include_original:
            out.append(q)
        out.extend(alts)
        out = _dedup_queries(out)
        limit = self.n + (1 if self.include_original else 0)
        return out[:limit] or [q]


class HyDERewriter:
    """Hypothetical Document Embedding — LLM writes a fake answer, search that."""

    name = "hyde"

    def __init__(self, llm: Any, *, include_original: bool = True) -> None:
        self.llm = llm
        self.include_original = include_original
        self._mq = MultiQueryRewriter(llm, n=1, include_original=False)

    async def rewrite(self, query: str) -> list[str]:
        q = (query or "").strip()
        if not q:
            return []
        prompt = (
            "Write a short hypothetical passage that would answer the question. "
            "Do not mention that it is hypothetical.\n\n"
            f"Question: {q}\n"
        )
        hypo = (await self._mq._call_llm(prompt)).strip()
        out: list[str] = []
        if self.include_original:
            out.append(q)
        if hypo:
            out.append(hypo)
        return out or [q]


class AgenticDecompositionRewriter:
    """Agentic query planner — LLM dynamically decides whether to decompose into sub-questions.

    For simple questions, preserves a single targeted query. For complex, multi-part,
    or comparative questions, decomposes into 2-5 orthogonal sub-queries for parallel fan-out.
    """

    name = "agentic"

    def __init__(
        self,
        llm: Any,
        *,
        max_subqueries: int = 5,
        include_original: bool = True,
    ) -> None:
        self.llm = llm
        self.max_subqueries = max(1, int(max_subqueries))
        self.include_original = include_original
        self._mq = MultiQueryRewriter(llm, n=self.max_subqueries, include_original=False)

    async def rewrite(self, query: str) -> list[str]:
        q = (query or "").strip()
        if not q:
            return []
        prompt = (
            "You are an expert retrieval planner. Analyze the user question and determine how to search knowledge effectively.\n"
            "- If the question is simple, atomic, or focused on a single topic, output 1 direct search query.\n"
            f"- If the question is complex, multi-part, or comparative, decompose it into 2 to {self.max_subqueries} targeted, orthogonal sub-queries.\n"
            "Output ONLY the search queries, one per line. Do not number them or include explanations.\n\n"
            f"Question: {q}\n"
        )
        raw = await self._mq._call_llm(prompt)
        subqueries: list[str] = []
        for line in (raw or "").splitlines():
            line = _clean_line(line)
            if line and line.lower() != q.lower():
                subqueries.append(line)
        if not subqueries:
            return [q]
        out: list[str] = []
        if self.include_original:
            out.append(q)
        out.extend(subqueries)
        # Drop near-duplicates (exact + token-overlap) so parallel fan-out isn't
        # spent on redundant searches of the same ground.
        out = _dedup_queries(out)
        limit = self.max_subqueries + (1 if self.include_original else 0)
        return out[:limit] or [q]


def resolve_rewriter(
    spec: str | Any | None,
    *,
    llm: Any | None = None,
) -> Any:
    """Resolve ``off`` / ``multi_query`` / ``hyde`` / ``agentic`` / custom rewriter object."""
    if spec is None or spec is False or spec == "off":
        return IdentityRewriter()
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in {"off", "identity", "none", ""}:
            return IdentityRewriter()
        if key in {"multi_query", "multiquery", "mq"}:
            if llm is None:
                raise ValueError("rewrite='multi_query' requires llm=")
            return MultiQueryRewriter(llm)
        if key == "hyde":
            if llm is None:
                raise ValueError("rewrite='hyde' requires llm=")
            return HyDERewriter(llm)
        if key in {"agentic", "decompose", "adaptive", "subquery", "subqueries"}:
            if llm is None:
                raise ValueError("rewrite='agentic' requires llm=")
            return AgenticDecompositionRewriter(llm)
        raise ValueError(
            f"unknown rewrite={spec!r}; use off|multi_query|hyde|agentic|custom"
        )
    return spec
