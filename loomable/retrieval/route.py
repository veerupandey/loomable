"""Built-in pluggable routers (retrieval mode + multi-corpus)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Sequence

__all__ = [
    "FixedModeRouter",
    "HeuristicModeRouter",
    "LLMModeRouter",
    "AllCorporaRouter",
    "DescriptionCorpusRouter",
    "resolve_mode_router",
    "resolve_corpus_router",
]

_FILE_HINT = re.compile(
    r"\b[\w./-]+\.(?:md|mdx|txt|pdf|py|ts|tsx|js|jsx|json|ya?ml|toml|html?|rst)\b",
    re.I,
)
_FILE_WORDS = re.compile(
    r"\b(file|filename|document|doc|readme|path|in\s+[\w./-]+\.\w+)\b",
    re.I,
)


class FixedModeRouter:
    """Always return the same mode."""

    name = "fixed"

    def __init__(self, mode: Literal["chunks", "file"] = "chunks") -> None:
        if mode not in ("chunks", "file"):
            raise ValueError("mode must be chunks|file")
        self.mode = mode

    async def choose_mode(self, query: str) -> Literal["chunks", "file"]:
        return self.mode


class HeuristicModeRouter:
    """Pick ``file`` when the query names a path/filename; else ``chunks``."""

    name = "heuristic"

    async def choose_mode(self, query: str) -> Literal["chunks", "file"]:
        q = query or ""
        if _FILE_HINT.search(q) or _FILE_WORDS.search(q):
            return "file"
        return "chunks"


class LLMModeRouter:
    """LLM chooses chunks vs file (LlamaIndex auto_routed lite)."""

    name = "llm"

    def __init__(self, llm: Any) -> None:
        from loomable.retrieval.rewrite import MultiQueryRewriter

        self._call = MultiQueryRewriter(llm)._call_llm

    async def choose_mode(self, query: str) -> Literal["chunks", "file"]:
        prompt = (
            "Classify the retrieval mode for this question. "
            "Reply with exactly one word: chunks OR file.\n"
            "- file: user names a specific file/path or wants a whole document\n"
            "- chunks: factual / semantic lookup over passages\n\n"
            f"Question: {query}\n"
        )
        raw = (await self._call(prompt)).strip().lower()
        if "file" in raw.split()[0:1] or raw.startswith("file"):
            return "file"
        return "chunks"


class AllCorporaRouter:
    """Query every corpus (fan-out)."""

    name = "all"

    async def choose_corpora(
        self, query: str, corpora: Sequence[dict[str, str]]
    ) -> list[str]:
        return [c["name"] for c in corpora if c.get("name")]


class DescriptionCorpusRouter:
    """LLM routes to corpora by name/description (LlamaIndex composite ROUTED)."""

    name = "description"

    def __init__(self, llm: Any, *, max_corpora: int = 3) -> None:
        from loomable.retrieval.rewrite import MultiQueryRewriter

        self._call = MultiQueryRewriter(llm)._call_llm
        self.max_corpora = max(1, int(max_corpora))

    async def choose_corpora(
        self, query: str, corpora: Sequence[dict[str, str]]
    ) -> list[str]:
        if not corpora:
            return []
        if len(corpora) == 1:
            return [corpora[0]["name"]]
        listing = "\n".join(
            f"- {c['name']}: {c.get('description') or '(no description)'}"
            for c in corpora
        )
        prompt = (
            f"Pick up to {self.max_corpora} corpora for this question. "
            "Return corpus names only, one per line.\n\n"
            f"Corpora:\n{listing}\n\nQuestion: {query}\n"
        )
        raw = await self._call(prompt)
        names = {c["name"] for c in corpora}
        chosen: list[str] = []
        for line in (raw or "").splitlines():
            tok = line.strip().lstrip("-* ").split()[0] if line.strip() else ""
            if tok in names and tok not in chosen:
                chosen.append(tok)
            if len(chosen) >= self.max_corpora:
                break
        return chosen or [corpora[0]["name"]]


def resolve_mode_router(
    mode: str | Any | None,
    *,
    llm: Any | None = None,
) -> Any:
    """``chunks``|``file``|``auto``|``heuristic``|``llm``|custom ModeRouter."""
    if mode is None:
        return HeuristicModeRouter()
    if not isinstance(mode, str):
        return mode
    key = mode.strip().lower()
    if key == "chunks":
        return FixedModeRouter("chunks")
    if key == "file":
        return FixedModeRouter("file")
    if key in {"auto", "heuristic"}:
        return HeuristicModeRouter()
    if key == "llm":
        if llm is None:
            raise ValueError("mode='llm' requires llm=")
        return LLMModeRouter(llm)
    raise ValueError(f"unknown mode={mode!r}; use chunks|file|auto|llm|custom")


def resolve_corpus_router(
    spec: str | Any | None,
    *,
    llm: Any | None = None,
) -> Any:
    if spec is None or spec == "all":
        return AllCorporaRouter()
    if not isinstance(spec, str):
        return spec
    key = spec.strip().lower()
    if key == "all":
        return AllCorporaRouter()
    if key in {"description", "routed", "llm"}:
        if llm is None:
            raise ValueError("corpus_router='description' requires llm=")
        return DescriptionCorpusRouter(llm)
    raise ValueError(f"unknown corpus_router={spec!r}; use all|description|custom")


def match_file_sources(query: str, sources: Sequence[str]) -> list[str]:
    """Return sources whose basename/path appears in the query."""
    q = (query or "").lower()
    hinted = {m.group(0).lower() for m in _FILE_HINT.finditer(query or "")}
    hits: list[str] = []
    for src in sources:
        if not src:
            continue
        s = src.lower()
        base = Path(s).name.lower()
        if base in q or s in q or base in hinted or s in hinted:
            hits.append(src)
            continue
        # stem match e.g. "readme"
        stem = Path(s).stem.lower()
        if len(stem) >= 4 and re.search(rf"\b{re.escape(stem)}\b", q):
            hits.append(src)
    return hits
