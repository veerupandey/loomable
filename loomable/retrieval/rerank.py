"""Built-in pluggable rerankers and compressors."""

from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "IdentityReranker",
    "ScoreReranker",
    "MMRReranker",
    "LLMReranker",
    "IdentityCompressor",
    "LLMCompressor",
    "resolve_reranker",
    "resolve_compressor",
]


class IdentityReranker:
    """Keep retrieval order; truncate to ``top_n``."""

    name = "off"

    async def rerank(
        self,
        query: str,
        hits: Sequence[dict[str, Any]],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        return list(hits)[: max(0, int(top_n))]


class ScoreReranker:
    """Re-sort by existing ``score`` field (no model)."""

    name = "score"

    async def rerank(
        self,
        query: str,
        hits: Sequence[dict[str, Any]],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            hits, key=lambda h: float(h.get("score") or 0.0), reverse=True
        )
        return ordered[: max(0, int(top_n))]


class MMRReranker:
    """Maximal Marginal Relevance — balance relevance vs diversity (industry RAG).

    Uses token Jaccard as a cheap diversity signal when embeddings are absent.
    ``lambda_mult`` closer to 1.0 favors relevance; closer to 0 favors diversity.
    """

    name = "mmr"

    def __init__(self, *, lambda_mult: float = 0.7) -> None:
        self.lambda_mult = min(1.0, max(0.0, float(lambda_mult)))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        import re

        return {t.lower() for t in re.findall(r"[a-z0-9_]+", text or "", flags=re.I)}

    def _sim(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    async def rerank(
        self,
        query: str,
        hits: Sequence[dict[str, Any]],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        top_n = max(0, int(top_n))
        if not hits or top_n == 0:
            return []
        q_tok = self._tokens(query)
        candidates = list(hits)
        # relevance proxy: existing score + query overlap
        scored: list[tuple[float, dict[str, Any], set[str]]] = []
        for h in candidates:
            toks = self._tokens(str(h.get("content") or ""))
            rel = float(h.get("score") or 0.0) + 0.15 * self._sim(q_tok, toks)
            scored.append((rel, h, toks))
        selected: list[dict[str, Any]] = []
        selected_toks: list[set[str]] = []
        while scored and len(selected) < top_n:
            best_i = -1
            best_val = float("-inf")
            for i, (rel, _h, toks) in enumerate(scored):
                div = 0.0
                if selected_toks:
                    div = max(self._sim(toks, s) for s in selected_toks)
                mmr = self.lambda_mult * rel - (1.0 - self.lambda_mult) * div
                if mmr > best_val:
                    best_val = mmr
                    best_i = i
            rel, hit, toks = scored.pop(best_i)
            row = dict(hit)
            row["score"] = float(best_val)
            selected.append(row)
            selected_toks.append(toks)
        return selected


class LLMReranker:
    """Ask an LLM to pick/order hit ids (simple pluggable cross-encoder stand-in).

    ``llm``: ``async (prompt: str) -> str`` or provider with ``.complete``.
    Custom cross-encoders: implement the :class:`~loomable.retrieval.plugins.Reranker`
    protocol instead.
    """

    name = "llm"

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        from loomable.retrieval.rewrite import MultiQueryRewriter

        self._call = MultiQueryRewriter(llm)._call_llm

    async def rerank(
        self,
        query: str,
        hits: Sequence[dict[str, Any]],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        if not hits:
            return []
        top_n = max(1, int(top_n))
        lines = []
        by_id: dict[str, dict[str, Any]] = {}
        for i, hit in enumerate(hits):
            hid = str(hit.get("id") or f"hit-{i}")
            by_id[hid] = hit
            preview = str(hit.get("content") or "")[:400].replace("\n", " ")
            lines.append(f"{hid}: {preview}")
        prompt = (
            f"Rank these passages for the query. Return the best {top_n} ids, "
            f"one per line, most relevant first.\n\nQuery: {query}\n\n"
            + "\n".join(lines)
        )
        raw = await self._call(prompt)
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line in (raw or "").splitlines():
            tok = line.strip().split()[0] if line.strip() else ""
            tok = tok.strip(":-)")
            if tok in by_id and tok not in seen:
                row = dict(by_id[tok])
                row["score"] = float(len(hits) - len(ordered))
                ordered.append(row)
                seen.add(tok)
            if len(ordered) >= top_n:
                break
        if not ordered:
            return list(hits)[:top_n]
        return ordered


class IdentityCompressor:
    name = "off"

    async def compress(
        self, query: str, hits: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return list(hits)


class LLMCompressor:
    """Extract query-relevant sentences from each hit (LC ContextualCompression)."""

    name = "llm"

    def __init__(self, llm: Any) -> None:
        from loomable.retrieval.rewrite import MultiQueryRewriter

        self._call = MultiQueryRewriter(llm)._call_llm

    async def compress(
        self, query: str, hits: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for hit in hits:
            text = str(hit.get("content") or "")
            if not text.strip():
                continue
            prompt = (
                "Extract only the sentences relevant to the query. "
                "If nothing is relevant, reply with EMPTY.\n\n"
                f"Query: {query}\n\nPassage:\n{text[:3000]}\n"
            )
            extracted = (await self._call(prompt)).strip()
            if not extracted or extracted.upper() == "EMPTY":
                continue
            row = dict(hit)
            row["content"] = extracted
            row["compressed"] = True
            out.append(row)
        return out


def resolve_reranker(spec: str | bool | Any | None, *, llm: Any | None = None) -> Any:
    if spec is None or spec is False or spec == "off":
        return IdentityReranker()
    if spec is True:
        return ScoreReranker()
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in {"off", "none", "identity", ""}:
            return IdentityReranker()
        if key in {"score", "true", "on"}:
            return ScoreReranker()
        if key == "mmr":
            return MMRReranker()
        if key == "llm":
            if llm is None:
                raise ValueError("rerank='llm' requires llm=")
            return LLMReranker(llm)
        raise ValueError(f"unknown rerank={spec!r}; use off|score|mmr|llm|custom")
    return spec


def resolve_compressor(spec: str | bool | Any | None, *, llm: Any | None = None) -> Any:
    if spec is None or spec is False or spec == "off":
        return IdentityCompressor()
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in {"off", "none", "identity", ""}:
            return IdentityCompressor()
        if key == "llm":
            if llm is None:
                raise ValueError("compress='llm' requires llm=")
            return LLMCompressor(llm)
        raise ValueError(f"unknown compress={spec!r}; use off|llm|custom")
    if spec is True:
        if llm is None:
            raise ValueError("compress=True requires llm=")
        return LLMCompressor(llm)
    return spec
