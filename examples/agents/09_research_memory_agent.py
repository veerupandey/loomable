#!/usr/bin/env python3
"""Entry point — see ``research_memory_agent.py`` for the factory."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from research_memory_agent import demo

    asyncio.run(demo())
