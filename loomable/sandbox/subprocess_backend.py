"""Subprocess-backed soft sandbox (default)."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Mapping

from loomable.sandbox.types import ExecResult

# Deny obviously destructive / privilege-escalating shell patterns.
_SHELL_DENY = (
    re.compile(r"(^|\s)rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)*(?:/|~|/etc\b|/usr\b|/var\b|/home\b)"),
    re.compile(r"\b(sudo|doas|su)\b"),
    re.compile(r"\bmkfs\b|\bdd\s+if=", re.I),
    re.compile(r">\s*/dev/sd"),
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh\b", re.I),
    re.compile(r"\bwget\b.*\|\s*(ba)?sh\b", re.I),
    re.compile(r"\bchmod\s+(-R\s+)?777\b"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;"),  # fork bomb
)


def shell_command_allowed(command: str) -> str | None:
    """Return an error string if ``command`` is blocked, else None."""
    text = (command or "").strip()
    if not text:
        return "empty command"
    for pat in _SHELL_DENY:
        if pat.search(text):
            return f"command blocked by sandbox policy: {text[:120]}"
    return None


class SubprocessSandbox:
    """Run Python/shell in a child process rooted at ``root``.

    Soft isolation only: no network/namespace jail. Paths for Python files are
    constrained under ``root`` when set.
    """

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        timeout: float = 30.0,
        scrub_env: bool = True,
        shell_bin: str | None = None,
    ) -> None:
        self._root = Path(root).resolve() if root else None
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
        self._timeout = float(timeout)
        self._scrub_env = scrub_env
        self._shell_bin = shell_bin or os.environ.get("SHELL") or "/bin/sh"

    @property
    def root(self) -> str | None:
        return str(self._root) if self._root is not None else None

    @property
    def timeout(self) -> float:
        return self._timeout

    def _env(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        if self._scrub_env:
            base = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(self._root) if self._root else os.environ.get("HOME", "/tmp"),
                "LANG": os.environ.get("LANG", "C.UTF-8"),
                "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
                "PYTHONUNBUFFERED": "1",
            }
            # Keep VIRTUAL_ENV / PYTHONPATH out unless caller opts in via extra.
        else:
            base = dict(os.environ)
        if extra:
            base.update({str(k): str(v) for k, v in extra.items()})
        return base

    def _resolve_under_root(self, path: str) -> Path | ExecResult:
        raw = Path(path)
        if self._root is None:
            return raw.expanduser().resolve()
        candidate = (self._root / path).resolve() if not raw.is_absolute() else raw.resolve()
        root_s = str(self._root)
        if not (str(candidate) == root_s or str(candidate).startswith(root_s + os.sep)):
            return ExecResult(error=f"path escapes sandbox root: {path}", returncode=1)
        return candidate

    async def _run(
        self,
        argv: list[str],
        *,
        timeout: float | None,
        env: Mapping[str, str] | None,
    ) -> ExecResult:
        limit = self._timeout if timeout is None else float(timeout)
        cwd = str(self._root) if self._root is not None else None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=self._env(env),
            )
        except OSError as exc:
            return ExecResult(error=str(exc), returncode=127)
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=limit)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(timed_out=True, returncode=-1, error=f"timed out after {limit} seconds")
        return ExecResult(
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            returncode=int(proc.returncode or 0),
        )

    async def run_python(
        self,
        code: str,
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        return await self._run([sys.executable, "-c", code], timeout=timeout, env=env)

    async def run_python_file(
        self,
        path: str,
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        resolved = self._resolve_under_root(path)
        if isinstance(resolved, ExecResult):
            return resolved
        if not resolved.is_file():
            return ExecResult(error=f"file not found: {path}", returncode=1)
        return await self._run([sys.executable, str(resolved)], timeout=timeout, env=env)

    async def run_shell(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        blocked = shell_command_allowed(command)
        if blocked:
            return ExecResult(error=blocked, returncode=126)
        # Prefer -c with the configured shell so users get bash features when SHELL=bash.
        return await self._run(
            [self._shell_bin, "-c", command],
            timeout=timeout,
            env=env,
        )


def format_argv_preview(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)
