"""Optional typed configuration for advanced agent execution."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionOptions:
    token_budget: int = 8192
    max_run_tokens: int | None = None
    max_tool_iterations: int = 12
    tool_timeout: float | None = None
    tool_concurrency: int | None = None
    loop_repeat_threshold: int = 3

    def __post_init__(self):
        for name in ("token_budget", "max_tool_iterations", "loop_repeat_threshold"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("tool_timeout", "tool_concurrency"):
            if getattr(self, name) is not None and getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class DurabilityOptions:
    """Opt-in JSON execution journal. Requires an explicit session_id.

    The checkpointer coordinates one writer per session.
    Dynamic discovery, planning, delegation, and multimodal tools are excluded.
    """
    checkpointer: Any

    def __post_init__(self):
        if not all(callable(getattr(self.checkpointer, name, None)) for name in ("get", "put", "claim")):
            raise ValueError("durability requires a Checkpointer with get/put/claim")
