"""Unit tests for loomable.toolkits.sql_tools."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile

import pytest

from loomable.toolkits.sql_tools import SQLTools


@pytest.fixture
def sample_db(tmp_path):
    """Create a sample SQLite database for testing."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)"
    )
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@example.com')")
    conn.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)"
    )
    conn.execute("INSERT INTO orders VALUES (1, 1, 99.99)")
    conn.commit()
    conn.close()
    return db_path


class TestSQLToolsInit:
    def test_default_read_only(self):
        tools = SQLTools()
        assert tools._read_only is True

    def test_read_only_false(self):
        tools = SQLTools(read_only=False)
        assert tools._read_only is False

    def test_register_tools_returns_three(self):
        tools = SQLTools()
        registered = tools._register_tools()
        assert len(registered) == 3
        names = {t.name for t in registered}
        assert names == {"run_sql", "list_tables", "describe_table"}

    def test_tools_method_with_include(self):
        tools = SQLTools(include_tools=["run_sql"])
        result = tools.tools()
        assert len(result) == 1
        assert result[0].name == "run_sql"

    def test_tools_method_with_exclude(self):
        tools = SQLTools(exclude_tools=["describe_table"])
        result = tools.tools()
        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"run_sql", "list_tables"}


class TestRunSQL:
    @pytest.mark.asyncio
    async def test_select_query(self, sample_db):
        tools = SQLTools()
        result = await tools._run_sql(sample_db, "SELECT * FROM users")
        assert "Alice" in result
        assert "Bob" in result
        assert "id" in result
        assert "name" in result

    @pytest.mark.asyncio
    async def test_select_no_rows(self, sample_db):
        tools = SQLTools()
        result = await tools._run_sql(sample_db, "SELECT * FROM users WHERE id = 999")
        assert "(no rows)" in result

    @pytest.mark.asyncio
    async def test_write_blocked_in_read_only(self, sample_db):
        tools = SQLTools(read_only=True)
        result = await tools._run_sql(
            sample_db, "INSERT INTO users VALUES (3, 'Charlie', 'charlie@example.com')"
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_write_allowed_when_not_read_only(self, sample_db):
        tools = SQLTools(read_only=False)
        result = await tools._run_sql(
            sample_db, "INSERT INTO users VALUES (3, 'Charlie', 'charlie@example.com')"
        )
        assert "Rows affected" in result

    @pytest.mark.asyncio
    async def test_missing_database(self, tmp_path):
        tools = SQLTools()
        db_path = str(tmp_path / "nonexistent.db")
        result = await tools._run_sql(db_path, "SELECT 1")
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_malformed_sql(self, sample_db):
        tools = SQLTools()
        result = await tools._run_sql(sample_db, "SELCT * FORM users")
        assert "Error" in result


class TestListTables:
    @pytest.mark.asyncio
    async def test_lists_all_tables(self, sample_db):
        tools = SQLTools()
        result = await tools._list_tables(sample_db)
        assert "users" in result
        assert "orders" in result

    @pytest.mark.asyncio
    async def test_empty_database(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.close()
        tools = SQLTools()
        result = await tools._list_tables(db_path)
        assert "(no tables)" in result

    @pytest.mark.asyncio
    async def test_missing_database(self, tmp_path):
        tools = SQLTools()
        db_path = str(tmp_path / "nonexistent.db")
        result = await tools._list_tables(db_path)
        assert "Error" in result
        assert "not found" in result


class TestDescribeTable:
    @pytest.mark.asyncio
    async def test_describes_columns(self, sample_db):
        tools = SQLTools()
        result = await tools._describe_table(sample_db, "users")
        assert "id" in result
        assert "name" in result
        assert "email" in result
        assert "INTEGER" in result
        assert "TEXT" in result

    @pytest.mark.asyncio
    async def test_shows_nullable_info(self, sample_db):
        tools = SQLTools()
        result = await tools._describe_table(sample_db, "users")
        # 'name' column is NOT NULL
        assert "NO" in result

    @pytest.mark.asyncio
    async def test_nonexistent_table(self, sample_db):
        tools = SQLTools()
        result = await tools._describe_table(sample_db, "nonexistent")
        assert "not found" in result or "no columns" in result

    @pytest.mark.asyncio
    async def test_missing_database(self, tmp_path):
        tools = SQLTools()
        db_path = str(tmp_path / "nonexistent.db")
        result = await tools._describe_table(db_path, "users")
        assert "Error" in result
        assert "not found" in result
