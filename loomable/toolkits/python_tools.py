"""loomable.toolkits.python_tools - Python code execution toolkit.

Executes Python via a :class:`~loomable.sandbox.types.Sandbox` (default:
:class:`~loomable.sandbox.SubprocessSandbox`) with timeout and working directory.
"""

from __future__ import annotations

from loomable.agent.tools import FunctionTool
from loomable.sandbox import SubprocessSandbox
from loomable.sandbox.types import Sandbox
from loomable.toolkits._base import Toolkit


class PythonTools(Toolkit):
    """Python code execution toolkit using a sandbox backend.

    Executes Python code or files isolated from the host agent process
    (subprocess by default). Supports configurable timeout and working directory.

    Usage::

        from loomable.toolkits import PythonTools
        from loomable.sandbox import SubprocessSandbox

        tools = PythonTools(sandbox=SubprocessSandbox(root="/tmp/sandbox", timeout=60))
        agent = Agent(model=..., tools=[tools])
    """

    def __init__(
        self,
        *,
        sandbox: Sandbox | None = None,
        timeout: int | float = 30,
        working_dir: str | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._sandbox: Sandbox = sandbox or SubprocessSandbox(
            root=working_dir, timeout=float(timeout)
        )
        # Keep attributes for older tests / callers.
        self._timeout = float(getattr(self._sandbox, "timeout", timeout))
        self._working_dir = getattr(self._sandbox, "root", working_dir)

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._run_python, name="run_python", idempotent=False),
            FunctionTool(self._run_python_file, name="run_python_file", idempotent=False),
        ]

    async def _run_python(self, code: str) -> str:
        """Execute Python code in a sandboxed subprocess and return output."""
        result = await self._sandbox.run_python(code)
        return result.format()

    async def _run_python_file(self, path: str) -> str:
        """Execute a Python file in a sandboxed subprocess and return output."""
        result = await self._sandbox.run_python_file(path)
        return result.format()
