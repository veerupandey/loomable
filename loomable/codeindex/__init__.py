"""Codebase indexing for agents — pluggable vector store (zvec by default).

Build once, search / map / find symbols from any :class:`~loomable.agent.Agent`
or :func:`~loomable.agent.deep.create_deep_agent` profile::

    from loomable.codeindex import CodeIndex
    from loomable.toolkits import CodeTools

    index = await CodeIndex.build("./my-app")  # HashingEmbedder + zvec
    agent = Agent(model=..., tools=[CodeTools(index)], skills=["coding"])
"""

from __future__ import annotations

from loomable.codeindex.chunking import CodeChunk, iter_code_chunks
from loomable.codeindex.embedders import HashingEmbedder
from loomable.codeindex.index import CodeHit, CodeIndex

__all__ = [
    "CodeChunk",
    "CodeHit",
    "CodeIndex",
    "HashingEmbedder",
    "iter_code_chunks",
]
