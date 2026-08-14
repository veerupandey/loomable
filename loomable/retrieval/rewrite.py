"""Built-in pluggable query rewriters."""

from __future__ import annotations

from typing import Any

__all__ = [
    "IdentityRewriter",
    "MultiQueryRewriter",
    "HyDERewriter",
    "resolve_rewriter",
]


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
            line = line.strip().lstrip("0123456789.-) ").strip()
            if line and line.lower() != q.lower():
                alts.append(line)
            if len(alts) >= self.n:
                break
        out: list[str] = []
        if self.include_original:
            out.append(q)
        out.extend(alts)
        return out or [q]


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


def resolve_rewriter(
    spec: str | Any | None,
    *,
    llm: Any | None = None,
) -> Any:
    """Resolve ``off`` / ``multi_query`` / ``hyde`` / custom rewriter object."""
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
        raise ValueError(f"unknown rewrite={spec!r}; use off|multi_query|hyde|custom")
    return spec
