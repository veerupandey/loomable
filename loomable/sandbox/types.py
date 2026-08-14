"""Sandbox protocol and shared result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecResult:
    """Outcome of a sandboxed command or Python snippet."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False
    error: str | None = None

    def format(self) -> str:
        """Human-readable tool output for the model."""
        if self.error:
            return f"Error: {self.error}"
        if self.timed_out:
            return "Error: Execution timed out"
        if self.returncode != 0:
            return (
                f"Execution failed (return code {self.returncode}):\n"
                f"{self.stderr}\n\nOutput:\n{self.stdout}"
            )
        if self.stdout:
            return self.stdout
        return "Code executed successfully (no output)"


@runtime_checkable
class Sandbox(Protocol):
    """Minimal execution sandbox used by Python/Shell toolkits."""

    @property
    def root(self) -> str | None:
        """Working directory root (None = process cwd)."""
        ...

    @property
    def timeout(self) -> float:
        """Default per-call timeout in seconds."""
        ...

    async def run_python(
        self,
        code: str,
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        """Run inline Python in the sandbox."""
        ...

    async def run_python_file(
        self,
        path: str,
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        """Run a Python file path (relative to root when rooted)."""
        ...

    async def run_shell(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        """Run a shell command string in the sandbox."""
        ...
