"""loomable.content - Low-level typed multimodal content model.

This package holds the provider-agnostic multimodal content primitives used
across in-process calls, the FastAPI transport, and the MCP transport:

- ``MediaPart`` / ``Modality`` and the ``Text`` / ``Image`` / ``Video`` constructors
- ``Message``, ``AgentInput``, ``AgentOutput``
- ``ModelCapabilities``
- kernel bridging helpers (``to_model_request`` / ``from_model_response``)

It depends only on the standard library and ``loomable.kernel`` models. It must
not depend on ``loomable.agent`` or ``loomable.serve``.
"""

from .capabilities import (
    ModelCapabilities,
    capabilities_for,
    from_model_response,
    to_model_request,
)
from .errors import MediaPartError
from .message import AgentInput, AgentOutput, Message, to_agent_input
from .parts import Image, MediaPart, Modality, Text, Video

__all__ = [
    "MediaPartError",
    "Modality",
    "MediaPart",
    "Text",
    "Image",
    "Video",
    "Message",
    "AgentInput",
    "AgentOutput",
    "to_agent_input",
    "ModelCapabilities",
    "capabilities_for",
    "to_model_request",
    "from_model_response",
]
