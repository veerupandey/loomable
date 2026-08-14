"""Shared provider — re-exports ``examples/_provider`` for this folder."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import has_live_provider, make_provider, require_provider  # noqa: F401

__all__ = ["make_provider", "require_provider", "has_live_provider"]
