"""Shared pytest configuration and fixtures for loomable tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_default_zvec(tmp_path, monkeypatch):
    """Do not share ``.loomable/memory_zvec`` across tests (zvec exclusive lock)."""
    path = tmp_path / "memory_zvec"
    monkeypatch.setattr("loomable.kernel.long_term.DEFAULT_ZVEC_PATH", path)
    monkeypatch.setattr("loomable.providers.vector_store.DEFAULT_ZVEC_PATH", path)
