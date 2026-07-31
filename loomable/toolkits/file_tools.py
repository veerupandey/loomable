"""loomable.toolkits.file_tools - File I/O toolkit with optional sandboxing.

Provides read_file, write_file, and list_directory tools that resolve paths
relative to a configurable base directory and reject path traversal escapes.
Uses only the Python standard library.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
from pathlib import Path

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit


class FileTools(Toolkit):
    """File I/O toolkit with optional sandboxing via base_dir.

    All paths are resolved relative to ``base_dir``. Any path that escapes the
    base directory via traversal (e.g. ``../../etc/passwd``) is rejected with a
    descriptive error string.

    Usage::

        from loomable.toolkits import FileTools

        tools = FileTools(base_dir="/workspace")
        # Or with defaults (current working directory):
        tools = FileTools()
    """

    def __init__(
        self,
        *,
        base_dir: str | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._read_file, name="read_file"),
            FunctionTool(self._write_file, name="write_file", idempotent=False),
            FunctionTool(self._list_directory, name="list_directory"),
        ]

    def _resolve_safe_path(self, path: str) -> Path | str:
        """Resolve path relative to base_dir and check for traversal.

        Returns the resolved Path on success, or an error string if the path
        escapes the base directory.
        """
        resolved = (self._base_dir / path).resolve()
        if not (resolved == self._base_dir or str(resolved).startswith(str(self._base_dir) + os.sep)):
            return f"Error: Path traversal not allowed: {path}"
        return resolved

    async def _read_file(self, path: str) -> str:
        """Read a file and return its contents with format auto-detection."""
        safe = self._resolve_safe_path(path)
        if isinstance(safe, str):
            return safe

        try:
            content = await asyncio.to_thread(safe.read_text, encoding="utf-8")
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"

        return self._format_content(content, safe.suffix.lower())

    async def _write_file(self, path: str, content: str) -> str:
        """Write content to a file, creating parent directories as needed."""
        safe = self._resolve_safe_path(path)
        if isinstance(safe, str):
            return safe

        try:
            await asyncio.to_thread(self._do_write, safe, content)
        except PermissionError:
            return f"Error: Permission denied: {path}"

        return f"Successfully wrote {len(content.encode('utf-8'))} bytes to {path}"

    async def _list_directory(self, path: str = ".") -> str:
        """List files and directories at the given path."""
        safe = self._resolve_safe_path(path)
        if isinstance(safe, str):
            return safe

        try:
            entries = await asyncio.to_thread(self._do_list, safe)
        except FileNotFoundError:
            return f"Error: Directory not found: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"

        return "\n".join(entries)

    @staticmethod
    def _do_write(resolved_path: Path, content: str) -> None:
        """Synchronous write helper for use with asyncio.to_thread."""
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(content, encoding="utf-8")

    @staticmethod
    def _do_list(resolved_path: Path) -> list[str]:
        """Synchronous directory listing helper for use with asyncio.to_thread."""
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"Not a directory: {resolved_path}")
        entries = []
        for item in sorted(resolved_path.iterdir()):
            prefix = "[DIR] " if item.is_dir() else ""
            entries.append(f"{prefix}{item.name}")
        return entries

    @staticmethod
    def _format_content(content: str, extension: str) -> str:
        """Format file content based on detected file type."""
        if extension == ".json":
            try:
                data = json.loads(content)
                return json.dumps(data, indent=2)
            except (json.JSONDecodeError, ValueError):
                return content

        if extension == ".csv":
            try:
                reader = csv.reader(io.StringIO(content))
                rows = list(reader)
                if not rows:
                    return content
                # Calculate column widths for table formatting
                col_widths = [
                    max(len(row[i]) if i < len(row) else 0 for row in rows)
                    for i in range(max(len(row) for row in rows))
                ]
                lines = []
                for row in rows:
                    padded = [
                        (row[i] if i < len(row) else "").ljust(col_widths[i])
                        for i in range(len(col_widths))
                    ]
                    lines.append(" | ".join(padded))
                    # Add separator after header row
                    if row is rows[0] and len(rows) > 1:
                        lines.append("-+-".join("-" * w for w in col_widths))
                return "\n".join(lines)
            except csv.Error:
                return content

        # For txt, markdown, code, yaml, and all other formats: return raw content
        return content
