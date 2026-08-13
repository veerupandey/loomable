"""Unit tests for FileTools.write_json schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from loomable.toolkits.file_tools import FileTools


class Packet(BaseModel):
    incident_id: str
    severity: str = Field(pattern=r"^SEV-[1-4]$")


@pytest.mark.asyncio
async def test_write_json_pretty_writes_without_schema(tmp_path: Path) -> None:
    tools = FileTools(base_dir=str(tmp_path))
    result = await tools._write_json("out/packet.json", '{"a": 1, "b": [2]}')
    assert result.startswith("Successfully wrote")
    written = json.loads((tmp_path / "out" / "packet.json").read_text(encoding="utf-8"))
    assert written == {"a": 1, "b": [2]}


@pytest.mark.asyncio
async def test_write_json_validates_pydantic_schema(tmp_path: Path) -> None:
    tools = FileTools(base_dir=str(tmp_path), json_schema=Packet)
    ok = await tools._write_json(
        "packet.json",
        '{"incident_id": "INC-1", "severity": "SEV-1"}',
    )
    assert ok.startswith("Successfully wrote validated JSON")
    data = json.loads((tmp_path / "packet.json").read_text(encoding="utf-8"))
    assert data["severity"] == "SEV-1"


@pytest.mark.asyncio
async def test_write_json_returns_validation_error_string(tmp_path: Path) -> None:
    tools = FileTools(base_dir=str(tmp_path), json_schema=Packet)
    err = await tools._write_json(
        "packet.json",
        '{"incident_id": "INC-1", "severity": "P1"}',
    )
    assert err.startswith("Error: JSON failed Packet validation")
    assert not (tmp_path / "packet.json").exists()


@pytest.mark.asyncio
async def test_write_json_invalid_json_string(tmp_path: Path) -> None:
    tools = FileTools(base_dir=str(tmp_path))
    err = await tools._write_json("x.json", "{not json")
    assert err.startswith("Error: Invalid JSON")
