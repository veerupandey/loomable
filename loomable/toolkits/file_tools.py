"""loomable.toolkits.file_tools - File I/O toolkit with optional sandboxing.

Provides read_file, write_file, write_json, and list_directory tools that resolve
paths relative to a configurable base directory and reject path traversal escapes.
Uses only the Python standard library (plus optional Pydantic for schema checks).
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit


class FileTools(Toolkit):
    """File I/O toolkit with optional sandboxing via base_dir.

    All paths are resolved relative to ``base_dir``. Any path that escapes the
    base directory via traversal (e.g. ``../../etc/passwd``) is rejected with a
    descriptive error string.

    Pass ``json_schema`` (a Pydantic model type) to validate objects written via
    :meth:`write_json` before they hit disk.

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
        json_schema: type | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        self._json_schema = json_schema

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._read_file, name="read_file"),
            FunctionTool(self._write_file, name="write_file", idempotent=False),
            FunctionTool(self._write_json, name="write_json", idempotent=False),
            FunctionTool(self._edit_file, name="edit_file", idempotent=False),
            FunctionTool(self._list_directory, name="list_directory"),
            FunctionTool(self._glob_files, name="glob_files"),
            FunctionTool(self._grep_files, name="grep_files"),
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

    async def _write_json(self, path: str, content: str) -> str:
        """Parse JSON, optionally validate against json_schema, and write pretty JSON."""
        safe = self._resolve_safe_path(path)
        if isinstance(safe, str):
            return safe

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return f"Error: Invalid JSON: {exc}"

        if self._json_schema is not None:
            validated = self._validate_json_schema(data)
            if isinstance(validated, str):
                return validated
            data = validated

        pretty = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        try:
            await asyncio.to_thread(self._do_write, safe, pretty)
        except PermissionError:
            return f"Error: Permission denied: {path}"

        return f"Successfully wrote validated JSON ({len(pretty.encode('utf-8'))} bytes) to {path}"

    def _validate_json_schema(self, data: Any) -> Any | str:
        """Validate ``data`` with the configured Pydantic model; return data or error."""
        schema = self._json_schema
        assert schema is not None
        try:
            if hasattr(schema, "model_validate"):
                model = schema.model_validate(data)
                return model.model_dump(mode="json")
            if hasattr(schema, "parse_obj"):
                model = schema.parse_obj(data)
                if hasattr(model, "dict"):
                    return model.dict()
                return data
            # Callable constructor fallback (e.g. dataclass / TypedDict-like).
            schema(data)  # type: ignore[misc]
            return data
        except Exception as exc:  # noqa: BLE001 - surface as tool result string
            name = getattr(schema, "__name__", "schema")
            return f"Error: JSON failed {name} validation: {exc}"

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

    async def _edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace ``old_string`` with ``new_string`` inside an existing file."""
        safe = self._resolve_safe_path(path)
        if isinstance(safe, str):
            return safe
        try:
            content = await asyncio.to_thread(safe.read_text, encoding="utf-8")
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        if old_string not in content:
            return f"Error: old_string not found in {path}"
        count = content.count(old_string)
        if count > 1 and not replace_all:
            return (
                f"Error: old_string matched {count} times in {path}; "
                "set replace_all=true or make old_string unique"
            )
        updated = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        try:
            await asyncio.to_thread(self._do_write, safe, updated)
        except PermissionError:
            return f"Error: Permission denied: {path}"
        return f"Successfully edited {path}"

    async def _glob_files(self, pattern: str = "**/*") -> str:
        """Find files under base_dir matching a glob pattern."""
        import fnmatch

        matches: list[str] = []
        base = self._base_dir
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(base)).replace("\\", "/")
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern):
                matches.append(rel)
        return "\n".join(sorted(matches)) if matches else "(no matches)"

    async def _grep_files(self, query: str, path: str = ".", max_hits: int = 40) -> str:
        """Search file contents under ``path`` (substring match)."""
        safe = self._resolve_safe_path(path)
        if isinstance(safe, str):
            return safe
        hits: list[str] = []
        root = safe if safe.is_dir() else safe.parent
        files = [safe] if safe.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    rel = str(fp.relative_to(self._base_dir)).replace("\\", "/")
                    hits.append(f"{rel}:{i}:{line[:240]}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "(no matches)"

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
