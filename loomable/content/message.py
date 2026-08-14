"""loomable.content.message - Message, AgentInput, and AgentOutput models.

These types compose ``MediaPart`` values (from ``loomable.content.parts``) into the
conversational units exchanged with an agent:

- ``Message`` pairs a ``role`` with an ordered list of ``MediaPart`` parts.
- ``AgentInput`` is an ordered, non-empty sequence of ``Message`` values supplied to
  an agent for a run (Req 3.3).
- ``AgentOutput`` is an ordered, non-empty sequence of ``MediaPart`` values produced
  by an agent for a run (Req 3.4).

Depends only on the standard library and ``loomable.content.parts``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

from .parts import MediaPart, Modality, Text


@dataclass
class Message:
    """A single conversational message.

    ``role`` is one of ``"user"`` | ``"assistant"`` | ``"system"``. ``parts`` is the
    ordered list of multimodal content parts that make up the message.
    """

    role: str
    parts: list[MediaPart] = field(default_factory=list)


@dataclass
class AgentInput:
    """The ordered, non-empty message(s) supplied to an agent for a run (Req 3.3)."""

    messages: list[Message]

    def __post_init__(self) -> None:
        # Req 3.3: an AgentInput is composed of one or more messages.
        if not self.messages:
            raise ValueError("AgentInput.messages must contain at least one Message.")

    @classmethod
    def from_text(cls, text: str) -> "AgentInput":
        """Build an ``AgentInput`` of a single user message with one text part."""
        return cls(messages=[Message(role="user", parts=[Text(text)])])

    def modalities(self) -> set[Modality]:
        """Return the union of modalities across all parts in all messages."""
        return {
            part.modality
            for message in self.messages
            for part in message.parts
        }


@dataclass
class AgentOutput:
    """The ordered, non-empty content parts produced by an agent for a run (Req 3.4)."""

    parts: list[MediaPart]

    def __post_init__(self) -> None:
        # Req 3.4: an AgentOutput is composed of one or more parts.
        if not self.parts:
            raise ValueError("AgentOutput.parts must contain at least one MediaPart.")

    def text(self) -> str:
        """Return the concatenation of the decoded text of all TEXT parts.

        ``data`` bytes for text parts are decoded as UTF-8. Parts referenced only by
        ``uri`` (no inline ``data``) contribute nothing to the concatenation.
        """
        pieces: list[str] = []
        for part in self.parts:
            if part.modality is Modality.TEXT and part.data is not None:
                pieces.append(part.data.decode("utf-8"))
        return "".join(pieces)

    def modalities(self) -> set[Modality]:
        """Return the set of modalities across parts."""
        return {part.modality for part in self.parts}


def to_agent_input(value: Any) -> AgentInput:
    """Coerce a supported input value into an :class:`AgentInput`.

    Accepts flexible input: an agent may be given a plain string, an
    already-built :class:`AgentInput`, an :class:`AgentOutput`, a ``RunResult``,
    a Pydantic model, a dataclass instance, or a plain ``dict``.

    - :class:`AgentInput` -> returned unchanged.
    - ``str`` -> a single user text message.
    - :class:`AgentOutput` -> its ``.text()`` as a user message.
    - ``RunResult`` -> its ``.output.text()`` as a user message.
    - Pydantic ``BaseModel`` -> its ``model_dump_json()`` as user text.
    - dataclass instance -> ``json.dumps(asdict(...))`` as user text.
    - ``dict`` -> ``json.dumps(...)`` as user text.
    - anything else -> ``str(value)`` as user text.

    ``pydantic`` is imported lazily so this module works without it installed.
    """
    if isinstance(value, AgentInput):
        return value
    if isinstance(value, str):
        return AgentInput.from_text(value)

    # AgentOutput: extract the text content (framework's own output type).
    if isinstance(value, AgentOutput):
        return AgentInput.from_text(value.text())

    # RunResult: unwrap to AgentOutput, then extract text.
    # Import lazily to avoid circular dependency.
    try:
        from loomable.agent.run import RunResult
        if isinstance(value, RunResult):
            return AgentInput.from_text(value.output.text())
    except ImportError:
        pass

    # Pydantic model instance (lazy import so pydantic stays optional).
    try:
        import pydantic

        if isinstance(value, pydantic.BaseModel):
            return AgentInput.from_text(value.model_dump_json())
    except ImportError:
        pass

    # Dataclass instance (not the class itself).
    # Skip our own dataclasses (AgentOutput is handled above).
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return AgentInput.from_text(json.dumps(dataclasses.asdict(value)))
        except (TypeError, ValueError):
            # Fallback: if asdict/json fails (e.g. contains non-serializable fields),
            # use str() representation.
            return AgentInput.from_text(str(value))

    if isinstance(value, dict):
        return AgentInput.from_text(json.dumps(value))

    return AgentInput.from_text(str(value))
