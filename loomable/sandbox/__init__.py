"""Execution sandbox backends for agent toolkits.

Soft isolation by default (:class:`SubprocessSandbox`): separate process,
timeout, working-directory root, and optional env scrubbing. Not a hard
security boundary (no gVisor/Firecracker). Prefer Docker/microVM backends
or an external gateway for untrusted multi-tenant code.
"""

from __future__ import annotations

from loomable.sandbox.subprocess_backend import SubprocessSandbox
from loomable.sandbox.types import ExecResult, Sandbox

__all__ = [
    "ExecResult",
    "Sandbox",
    "SubprocessSandbox",
    "make_sandbox",
]


def make_sandbox(
    root: str | None = None,
    *,
    timeout: float = 30.0,
    backend: str = "subprocess",
) -> Sandbox:
    """Factory for the default sandbox backends.

    ``backend="subprocess"`` is always available. ``backend="docker"`` is
    optional and raises if the Docker CLI is unavailable.
    """
    key = (backend or "subprocess").strip().lower()
    if key == "subprocess":
        return SubprocessSandbox(root=root, timeout=timeout)
    if key == "docker":
        from loomable.sandbox.docker_backend import DockerSandbox

        return DockerSandbox(root=root, timeout=timeout)
    raise ValueError(f"unknown sandbox backend: {backend!r}")
