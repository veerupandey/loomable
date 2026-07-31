"""26 — Built-in Toolkits with Agent (High-Level Usage)

Shows how an Agent autonomously uses built-in toolkits to accomplish tasks.
The LLM decides which tools to call — you just pass toolkits in the tools= list.

Demonstrates:
- FileTools: agent reads/writes files on its own
- SQLTools: agent queries a real SQLite database
- PythonTools: agent runs Python code to solve problems
- Toolkits combined: agent uses multiple toolkits in one conversation
"""

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider
from loomable.toolkits import FileTools, SQLTools, PythonTools


# ============================================================================
# Setup: Create a workspace with files and a database for the agent to use
# ============================================================================

WORKSPACE = Path(tempfile.mkdtemp())

# Create some files the agent can read
(WORKSPACE / "readme.md").write_text(
    "# Project Alpha\n\nA machine learning pipeline for customer churn prediction.\n"
    "## Stack\n- Python 3.11\n- scikit-learn\n- pandas\n- FastAPI\n\n"
    "## Status\nIn development. Target launch: Q3 2025.\n"
)
(WORKSPACE / "config.json").write_text(
    '{"model": "random_forest", "n_estimators": 100, "max_depth": 10, '
    '"features": ["tenure", "monthly_charges", "contract_type", "payment_method"]}'
)

# Create a SQLite database with real data
DB_PATH = str(WORKSPACE / "analytics.db")
conn = sqlite3.connect(DB_PATH)
conn.executescript("""
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        plan TEXT NOT NULL,
        monthly_spend REAL,
        tenure_months INTEGER,
        churned INTEGER DEFAULT 0
    );
    INSERT INTO customers VALUES (1, 'Acme Corp', 'enterprise', 2500.00, 36, 0);
    INSERT INTO customers VALUES (2, 'StartupXYZ', 'pro', 150.00, 6, 1);
    INSERT INTO customers VALUES (3, 'BigRetail', 'enterprise', 4200.00, 48, 0);
    INSERT INTO customers VALUES (4, 'DevShop', 'pro', 250.00, 12, 0);
    INSERT INTO customers VALUES (5, 'TinyAgency', 'starter', 50.00, 3, 1);
    INSERT INTO customers VALUES (6, 'MegaBank', 'enterprise', 8500.00, 60, 0);
    INSERT INTO customers VALUES (7, 'LocalCafe', 'starter', 30.00, 2, 1);
    INSERT INTO customers VALUES (8, 'TechGiant', 'enterprise', 12000.00, 72, 0);
    INSERT INTO customers VALUES (9, 'FreelanceJo', 'starter', 50.00, 1, 1);
    INSERT INTO customers VALUES (10, 'MidMarket', 'pro', 450.00, 24, 0);

    CREATE TABLE revenue (
        month TEXT PRIMARY KEY,
        mrr REAL,
        new_customers INTEGER,
        churned_customers INTEGER
    );
    INSERT INTO revenue VALUES ('2025-01', 28200, 3, 1);
    INSERT INTO revenue VALUES ('2025-02', 29500, 4, 2);
    INSERT INTO revenue VALUES ('2025-03', 31000, 5, 1);
    INSERT INTO revenue VALUES ('2025-04', 30200, 2, 3);
    INSERT INTO revenue VALUES ('2025-05', 32500, 6, 1);
""")
conn.close()


# ============================================================================
# Build the Agent with Toolkits
# ============================================================================

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "You are a data analyst assistant. You have access to:\n"
        f"- A project workspace at: use relative paths (the base is set for you)\n"
        f"- A SQLite database at: {DB_PATH}\n"
        "- Python execution for computations\n\n"
        "Use your tools to answer questions. Be concise and data-driven."
    ),
    tools=[
        FileTools(base_dir=str(WORKSPACE)),
        SQLTools(read_only=True),
        PythonTools(timeout=15),
    ],
)


# ============================================================================
# Run: Let the agent use toolkits autonomously
# ============================================================================

async def main():
    print("=" * 60)
    print("  Built-in Toolkits — Agent-Driven Demo")
    print("=" * 60)
    print(f"\n  Workspace: {WORKSPACE}")
    print(f"  Database:  {DB_PATH}\n")

    # One big question that requires all toolkits at once
    big_question = (
        "I need a full project status report. Please do ALL of the following:\n"
        "1. Read the readme.md file and summarize what Project Alpha is about\n"
        "2. Read the config.json to see what ML model settings we're using\n"
        f"3. Query the database at {DB_PATH} to get: total customers, customers per plan, and total MRR from the revenue table\n"
        "4. Use Python to calculate the churn rate (churned/total from the customers table) "
        f"— connect to the database at {DB_PATH} with sqlite3\n"
        "\nGive me a concise summary combining all findings."
    )

    print(f"{'─' * 60}")
    print("  Question (requires FileTools + SQLTools + PythonTools):")
    print(f"{'─' * 60}")
    for line in big_question.splitlines():
        print(f"  {line}")
    print(f"{'─' * 60}\n")

    result = await agent.arun(big_question)

    print(f"  Agent Response:\n")
    for line in result.output.text().splitlines():
        print(f"    {line}")
    print()
    if result.tool_activity:
        print(f"  [Tools called: {len(result.tool_activity)} time(s) in a single run]")
    print()

    print("=" * 60)
    print("  Done! The agent used multiple toolkits in one go.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
