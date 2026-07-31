"""loomable.toolkits.python_tools - Python code execution toolkit.

Executes Python code in sandboxed subprocesses with configurable timeout
and working directory. Uses only the Python standard library.
"""

from __future__ import annotations

import asyncio
import sys

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit


class PythonTools(Toolkit):
    """Python code execution toolkit using sandboxed subprocesses.

    Executes Python code or files in a separate subprocess isolated from the
    host agent process. Supports configurable timeout and working directory.

    Usage::

        from loomable.toolkits import PythonTools

        tools = PythonTools(timeout=60, working_dir="/tmp/sandbox")
        agent = Agent(model=..., tools=[tools])
    """

    def __init__(
        self,
        *,
        timeout: int = 30,
        working_dir: str | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._timeout = timeout
        self._working_dir = working_dir

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._run_python, name="run_python", idempotent=False),
            FunctionTool(self._run_python_file, name="run_python_file", idempotent=False),
        ]

    async def _run_python(self, code: str) -> str:
        """Execute Python code in a sandboxed subprocess and return output."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._working_dir,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: Execution timed out after {self._timeout} seconds"

        stdout_text = stdout.decode()
        stderr_text = stderr.decode()

        if proc.returncode != 0:
            return (
                f"Execution failed (return code {proc.returncode}):\n"
                f"{stderr_text}\n\nOutput:\n{stdout_text}"
            )

        return stdout_text if stdout_text else "Code executed successfully (no output)"

    async def _run_python_file(self, path: str) -> str:
        """Execute a Python file in a sandboxed subprocess and return output."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._working_dir,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: Execution timed out after {self._timeout} seconds"

        stdout_text = stdout.decode()
        stderr_text = stderr.decode()

        if proc.returncode != 0:
            return (
                f"Execution failed (return code {proc.returncode}):\n"
                f"{stderr_text}\n\nOutput:\n{stdout_text}"
            )

        return stdout_text if stdout_text else "Code executed successfully (no output)"
