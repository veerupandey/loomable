"""Abstract Kernel contracts for loomable.

This module declares the stable abstract interfaces that form the Kernel boundary.
Concrete implementations live at the extension edge and are loaded via configuration.
No concrete/example module is imported here.

Contracts use:
- typing.Protocol for pluggable backends (structural subtyping): ModelProvider,
  MemoryBackend, VectorBackend
- abc.ABC + abc.abstractmethod for extension types with concrete fields: Tool,
  Retriever, Skill
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from loomable.kernel.models import ModelRequest, ModelResponse, ToolResult


# ---------------------------------------------------------------------------
# Protocol-based contracts (structural subtyping for pluggable backends)
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelProvider(Protocol):
    """Provider-agnostic model backend.

    Any class that implements `complete` with the correct signature satisfies
    this protocol — no explicit inheritance required.
    """

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send a completion request and return the model response."""
        ...


@runtime_checkable
class MemoryBackend(Protocol):
    """Pluggable short-term memory backend (RDBMS-style).

    Structural protocol so that SQLite, Postgres, DynamoDB, or any
    conforming store can be used without agent changes.
    """

    async def read(self, key: str) -> Any:
        """Read a value by key from the backend."""
        ...

    async def write(self, key: str, value: Any) -> None:
        """Write a value by key to the backend."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a value by key from the backend."""
        ...

    async def exists(self, key: str) -> bool:
        """Check whether a key exists in the backend."""
        ...


@runtime_checkable
class VectorBackend(Protocol):
    """Pluggable long-term vector memory backend.

    Structural protocol so that Alibaba zvec, FAISS (CPU/GPU), Postgres
    (`PgVectorBackend`), Pinecone, or any conforming vector store can be
    used without agent changes.
    """

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Index a vector with associated metadata."""
        ...

    async def query(
        self, vector: list[float], k: int
    ) -> list[dict[str, Any]]:
        """Query for the top-k most similar vectors. Returns results ranked
        by non-increasing similarity."""
        ...

    async def delete(self, id: str) -> None:
        """Remove an indexed vector by id."""
        ...


# ---------------------------------------------------------------------------
# ABC-based contracts (extension types with concrete fields)
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Abstract base for all invocable tools.

    Subclasses must provide `name`, `description`, and implement `invoke`.
    """

    name: str
    description: str

    @abstractmethod
    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Invoke the tool with the given arguments and return a result."""
        ...


class Retriever(ABC):
    """Abstract base for retriever components.

    Ship any concrete Retriever (or duck-typed object with ``name`` +
    ``async retrieve``) via ``Agent(retrievers=[...])``. At build time each
    retriever becomes a tool named ``retriever.name`` (prefer ``search_*``).

    Optional ``description`` is advertised to the model as the tool description.
    """

    name: str
    description: str = ""

    @abstractmethod
    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        """Retrieve the top-k items relevant to the query.

        Returns a list of result dicts, each containing at minimum a
        'content' key with the retrieved text/data.
        """
        ...


class Skill(ABC):
    """Abstract base for Anthropic-style Skills.

    A Skill is a capability package with a name, description, instruction
    body, and optional bundled script tools. Skills are loaded at the
    extension edge and never require Kernel modification.
    """

    name: str
    description: str
    body: str
    script_tools: list[str]

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return the Tool instances bundled with this Skill."""
        ...
