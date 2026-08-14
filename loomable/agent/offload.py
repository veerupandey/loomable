"""Workspace-aware tool-result offload for deep agents.

Unlike :meth:`~loomable.agent.context_policy.ContextPolicy.trim_tool_payloads`
(which truncates and loses content), this writes the full tool body to the
workspace and returns a short preview + path the agent can ``read_file`` / ``grep``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from loomable.kernel.models import ToolCall, ToolOutcome, ToolResult

__all__ = ["make_workspace_offload_hook", "offload_tool_text"]

DEFAULT_THRESHOLD = 12_000
DEFAULT_PREVIEW = 800


def offload_tool_text(
    workspace: str | Path,
    tool_name: str,
    content: str,
    *,
    preview_chars: int = DEFAULT_PREVIEW,
) -> tuple[str, str]:
    """Write ``content`` under ``workspace/.offload/`` and return (rel_path, preview_msg)."""
    root = Path(workspace)
    offload_dir = root / ".offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()[:12]
    safe_tool = "".join(c if c.isalnum() or c in "-_" else "_" for c in (tool_name or "tool"))[:40]
    rel = f".offload/{safe_tool}_{digest}.txt"
    dest = root / rel
    dest.write_text(content, encoding="utf-8")
    preview = content[:preview_chars]
    if len(content) > preview_chars:
        preview += "\n..."
    msg = (
        f"[offloaded {len(content)} chars to workspace:{rel}]\n"
        f"Use read_file or grep on '{rel}' to retrieve slices.\n"
        f"--- preview ---\n{preview}"
    )
    return rel, msg


def make_workspace_offload_hook(
    workspace: str | Path,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    preview_chars: int = DEFAULT_PREVIEW,
    skip_tools: frozenset[str] | None = None,
) -> Callable[[str, ToolCall | None, ToolOutcome], Any]:
    """Return a post-tool hook that offloads oversized string results to disk.

    Attach via ``Agent(tool_hooks=[hook])`` — the hook sets ``phase = \"post\"``.
    """
    skip = skip_tools or frozenset(
        {
            "read_file",
            "write_file",
            "edit_file",
            "ls",
            "glob",
            "grep",
            "write_todos",
            "read_todos",
            "update_todo",
            "think",
            "list_sources",
            "format_bibliography",
            "list_images",
        }
    )
    root = Path(workspace)

    def hook(
        tool_name: str,
        call: ToolCall | None,
        outcome: ToolOutcome,
    ) -> ToolOutcome | None:
        if outcome.result is None or outcome.error is not None:
            return None
        if tool_name in skip:
            return None
        content = outcome.result.content
        if not isinstance(content, str) or len(content) <= threshold:
            return None
        _rel, msg = offload_tool_text(
            root, tool_name, content, preview_chars=preview_chars
        )
        meta = dict(outcome.result.metadata or {})
        meta["offloaded"] = True
        meta["offload_path"] = _rel
        return ToolOutcome(
            call_id=outcome.call_id,
            result=ToolResult(content=msg, metadata=meta),
        )

    hook.phase = "post"  # type: ignore[attr-defined]
    return hook
