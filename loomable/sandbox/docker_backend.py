"""Optional Docker-backed sandbox (experimental).

Requires a working ``docker`` CLI. Mounts the sandbox root read-write at
``/workspace`` inside a short-lived container.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Mapping

from loomable.sandbox.subprocess_backend import shell_command_allowed
from loomable.sandbox.types import ExecResult


class DockerSandbox:
    """Run Python/shell inside ``docker run --rm`` with a bind-mounted root.

    Experimental: image must include Python. Default image is ``python:3.12-slim``.
    """

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        timeout: float = 60.0,
        image: str = "python:3.12-slim",
        network: str = "none",
    ) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("docker CLI not found; use SubprocessSandbox instead")
        self._root = Path(root or ".").resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._timeout = float(timeout)
        self._image = image
        self._network = network

    @property
    def root(self) -> str | None:
        return str(self._root)

    @property
    def timeout(self) -> float:
        return self._timeout

    def _base_argv(self) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            f"--network={self._network}",
            "-v",
            f"{self._root}:/workspace",
            "-w",
            "/workspace",
            self._image,
        ]

    async def _run(self, argv: list[str], *, timeout: float | None) -> ExecResult:
        limit = self._timeout if timeout is None else float(timeout)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
        del env  # container env left minimal for now
        return await self._run([*self._base_argv(), "python", "-c", code], timeout=timeout)

    async def run_python_file(
        self,
        path: str,
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        del env
        # Paths are interpreted relative to /workspace
        return await self._run([*self._base_argv(), "python", path], timeout=timeout)

    async def run_shell(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        del env
        blocked = shell_command_allowed(command)
        if blocked:
            return ExecResult(error=blocked, returncode=126)
        return await self._run([*self._base_argv(), "sh", "-c", command], timeout=timeout)
