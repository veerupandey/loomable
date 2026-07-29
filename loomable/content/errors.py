"""loomable.content.errors - Errors for the low-level multimodal content model.

``MediaPartError`` is a ``ValueError`` subclass (per design Error Taxonomy) raised
when a ``MediaPart`` is constructed with both/neither payload, or with a modality
that is inconsistent with its media type.
"""

from __future__ import annotations


class MediaPartError(ValueError):
    """Raised when a ``MediaPart`` is constructed invalidly.

    Invalid constructions include:
    - neither ``data`` nor ``uri`` provided (Req 3.5)
    - both ``data`` and ``uri`` provided (Req 3.5)
    - ``media_type`` prefix inconsistent with the declared ``modality`` (Req 3.6)
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
