"""Let an agent query a disposable SQLite database with a read-only toolkit."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from examples._provider import make_provider
from loomable import Agent
from loomable.toolkits import SQLTools


async def main():
    with TemporaryDirectory() as directory:
        database = Path(directory) / "orders.sqlite"
        with closing(sqlite3.connect(database)) as conn:
            conn.execute("CREATE TABLE orders (sku TEXT, units INTEGER)")
            conn.executemany("INSERT INTO orders VALUES (?, ?)", [("WIDGET", 12), ("WIDGET", 3), ("BOLT", 4)])
            conn.commit()

        async with make_provider() as model:
            agent = Agent(
                model, tools=[SQLTools(read_only=True)], require_tools=["run_sql"], strict_require_tools=True,
                instructions="Use SQL to calculate the answer from the orders table.",
            )
            print((await agent.arun(f"How many WIDGET units were ordered? Database path: {database}")).text)


if __name__ == "__main__":
    asyncio.run(main())
