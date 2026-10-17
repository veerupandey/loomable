"""Approve one known Python script for execution in the subprocess sandbox."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from examples._provider import make_provider
from loomable import Agent
from loomable.toolkits import PythonTools


def approve_fixture(call):
    return call.tool_name == "run_python_file" and call.args.get("path") == "totals.py"


async def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "totals.py").write_text("print(sum([2700, 1200, 9000, 1000, 2400]))\n", encoding="utf-8")
        async with make_provider() as model:
            agent = Agent(
                model,
                tools=[PythonTools(working_dir=directory, include_tools=["run_python_file"])],
                require_tools=["run_python_file"], strict_require_tools=True,
                require_confirmation=["run_python_file"], approver=approve_fixture,
                instructions="Run the provided script and report its output.",
            )
            print((await agent.arun("Run totals.py and report the order total in cents.")).text)


if __name__ == "__main__":
    asyncio.run(main())
