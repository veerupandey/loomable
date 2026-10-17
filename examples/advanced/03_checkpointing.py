"""Pause before an external action, then approve and resume from SQLite."""

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from examples._provider import make_provider
from loomable import Agent, FlowPaused, SQLiteCheckpointer, Workflow


async def main(approve: bool):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        store = SQLiteCheckpointer(str(root / "release.sqlite"))
        destination = root / "release-note.txt"

        def publish(draft):
            destination.write_text(draft.text(), encoding="utf-8")
            return "Release note saved."

        async with make_provider() as model:
            def workflow():
                drafter = Agent(model, instructions="Write a one-sentence release note.")
                return (Workflow("release", session_id="release-42", checkpointer=store)
                        .step("draft", drafter)
                        .step("publish", publish, confirm=True))

            try:
                await workflow().arun("Checkout now retries a failed payment once.")
            except FlowPaused as pause:
                print(f"Paused before {pause.node_id}. Draft saved in the checkpoint.")
                if approve:
                    reopened = workflow()
                    await reopened.approve(pause.node_id)
                    print((await reopened.arun(resume=True)).text)
                    print(destination.read_text(encoding="utf-8"))
                else:
                    print("Run with --approve to allow the demo to write the release note.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve", action="store_true")
    asyncio.run(main(parser.parse_args().approve))
