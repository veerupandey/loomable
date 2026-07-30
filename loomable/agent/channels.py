"""loomable.agent.channels - Decoupled message-passing between agents.

Provides a Channel Protocol and InMemoryChannel default for inter-agent
communication, iterative refinement, and mid-plan HITL.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ChannelMessage:
    """A message on a channel."""
    sender: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class Channel(Protocol):
    """A named async message channel between agents."""

    @property
    def name(self) -> str: ...

    async def send(self, msg: ChannelMessage) -> None: ...

    async def receive(self, timeout: float | None = None) -> ChannelMessage | None: ...

    async def peek(self) -> list[ChannelMessage]: ...

    async def clear(self) -> None: ...


class InMemoryChannel:
    """Zero-dependency default: asyncio.Queue-backed channel."""

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._queue: asyncio.Queue[ChannelMessage] = asyncio.Queue()
        self._history: list[ChannelMessage] = []

    @property
    def name(self) -> str:
        return self._name

    async def send(self, msg: ChannelMessage) -> None:
        self._history.append(msg)
        await self._queue.put(msg)

    async def receive(self, timeout: float | None = None) -> ChannelMessage | None:
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    async def peek(self) -> list[ChannelMessage]:
        return list(self._history)

    async def clear(self) -> None:
        self._history.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
