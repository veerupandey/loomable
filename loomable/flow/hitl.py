"""Flow-level human-in-the-loop: pause and resume.

Provides flow-level pause/resume over the existing durable pending-action
primitives so long-running workflows can gate risky steps without holding a
process open.

A node marked ``require_confirmation=True`` causes the engine to pause before
executing it, record a ``PendingAction``, checkpoint, and raise ``FlowPaused``.
On resume, the pending action's status ("approved"/"rejected") determines
whether the node runs or is skipped. Already-completed nodes are never
re-executed (Req 16.3).
"""

from __future__ import annotations

__all__ = ["FlowPaused"]

from loomable.kernel.errors import LoomableError
from loomable.persist.checkpoint import PendingAction


class FlowPaused(LoomableError):
    """Raised when a Flow pauses before a node requiring confirmation (Req 16.1).

    The caller catches this, inspects ``pending`` / ``node_id`` to present an
    approval UI, then resumes the flow with the decision (approve or reject).
    On resume the checkpointer restores state and the engine either executes
    or skips the node without re-running completed nodes.

    Attributes
    ----------
    pending:
        The durable ``PendingAction`` describing the node awaiting approval.
    thread_id:
        The checkpoint thread/session identifier so the flow can be resumed.
    node_id:
        Convenience alias for the paused Workflow step / node name
        (``pending.tool_name``).
    """

    def __init__(self, pending: PendingAction, thread_id: str) -> None:
        self.pending = pending
        self.thread_id = thread_id
        super().__init__(
            f"Flow paused before node {pending.tool_name!r} "
            f"(thread_id={thread_id!r}) — awaiting confirmation."
        )

    @property
    def node_id(self) -> str:
        """Paused step / node name (same as ``pending.tool_name``)."""
        return self.pending.tool_name
