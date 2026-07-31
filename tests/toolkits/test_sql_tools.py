# Feature: built-in-toolkits, Property 13: SQL query returns correct rows
# Feature: built-in-toolkits, Property 14: list_tables returns all table names
# Feature: built-in-toolkits, Property 15: describe_table returns column schema
# Feature: built-in-toolkits, Property 16: SQL read-only mode blocks write operations
# Feature: built-in-toolkits, Property 17: Missing database returns error
# Feature: built-in-toolkits, Property 18: Malformed SQL returns SQLite error
"""Property-based tests for loomable.toolkits.sql_tools.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.7, 7.8**
"""

from __future__ import annotations

import os
import sqlite3
import string
import tempfile

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.toolkits.sql_tools import SQLTools


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: valid SQL identifier (table/column names)
# Must start with a letter or underscore, followed by alphanumerics/underscores.
# Keep them short to avoid performance issues.
sql_identifiers = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True).filter(
    lambda s: s.lower() not in (
        "table", "select", "from", "where", "insert", "update", "delete",
        "create", "drop", "alter", "index", "into", "values", "set",
        "order", "group", "having", "join", "on", "and", "or", "not",
        "null", "true", "false", "integer", "text", "real", "blob",
        "primary", "key", "autoincrement", "unique", "default", "check",
        "foreign", "references", "constraint", "if", "exists", "in",
        "between", "like", "is", "as", "by", "asc", "desc", "limit",
        "offset", "union", "except", "intersect", "case", "when", "then",
        "else", "end", "cast", "distinct", "all", "any", "some",
    )
)

# Strategy: valid column types for SQLite
sqlite_types = st.sampled_from(["INTEGER", "TEXT", "REAL", "BLOB", "NUMERIC"])

# Strategy: generate random text values safe for SQL
safe_text_values = st.text(
    alphabet=string.ascii_letters + string.digits + " ",
    min_size=1,
    max_size=30,
)

# Strategy: generate integer values
int_values = st.integers(min_value=-1000000, max_value=1000000)

# Strategy: generate float values
float_values = st.floats(
    min_value=-1000000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Property 13: SQL query returns correct rows
# ---------------------------------------------------------------------------


class TestSQLQueryReturnsCorrectRows:
    """For any SQLite database with known rows and a valid SELECT query,
    run_sql SHALL return the matching result rows.

    **Validates: Requirements 7.1**
    """

    @settings(max_examples=20, deadline=None)
    @given(
        names=st.lists(safe_text_values, min_size=1, max_size=10, unique=True),
        ages=st.lists(int_values, min_size=1, max_size=10),
    )
    async def test_select_returns_all_inserted_rows(
        self, names: list[str], ages: list[int]
    ) -> None:
        # Align lists to same length
        n = min(len(names), len(ages))
        names = names[:n]
        ages = ages[:n]

        # Create database with known rows
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        for i, (name, age) in enumerate(zip(names, ages), start=1):
            conn.execute("INSERT INTO people VALUES (?, ?, ?)", (i, name, age))
        conn.commit()
        conn.close()

        # Query all rows
        tools = SQLTools()
        result = await tools._run_sql(db_path, "SELECT * FROM people")

        # Verify all names appear in the result
        for name in names:
            assert name in result, f"Expected name '{name}' in result"

        # Verify all ages appear in the result
        for age in ages:
            assert str(age) in result, f"Expected age '{age}' in result"


# ---------------------------------------------------------------------------
# Property 14: list_tables returns all table names
# ---------------------------------------------------------------------------


class TestListTablesReturnsAllTableNames:
    """For any SQLite database with a set of created tables,
    list_tables SHALL return all table names in that database.

    **Validates: Requirements 7.2**
    """

    @settings(max_examples=20, deadline=None)
    @given(
        table_names=st.lists(sql_identifiers, min_size=1, max_size=5, unique=True),
    )
    async def test_list_tables_returns_all_created_tables(
        self, table_names: list[str]
    ) -> None:
        # Create database with multiple tables
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "test.db")
        conn = sqlite3.connect(db_path)
        for table_name in table_names:
            conn.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        # List tables
        tools = SQLTools()
        result = await tools._list_tables(db_path)

        # Verify all table names appear in the result
        for table_name in table_names:
            assert table_name in result, (
                f"Expected table '{table_name}' in result: {result}"
            )


# ---------------------------------------------------------------------------
# Property 15: describe_table returns column schema
# ---------------------------------------------------------------------------


class TestDescribeTableReturnsColumnSchema:
    """For any table with known columns and types, describe_table SHALL return
    all column names and their types.

    **Validates: Requirements 7.3**
    """

    @settings(max_examples=20, deadline=None)
    @given(
        columns=st.lists(
            st.tuples(sql_identifiers, sqlite_types),
            min_size=1,
            max_size=6,
            unique_by=lambda x: x[0],
        ),
    )
    async def test_describe_table_returns_all_columns(
        self, columns: list[tuple[str, str]]
    ) -> None:
        # Ensure no column is named 'id' (we use that as PK)
        assume(all(col_name != "id" for col_name, _ in columns))

        # Create database with a table having the generated columns
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "test.db")
        conn = sqlite3.connect(db_path)
        col_defs = ", ".join(f"{name} {typ}" for name, typ in columns)
        conn.execute(f"CREATE TABLE test_table (id INTEGER PRIMARY KEY, {col_defs})")
        conn.commit()
        conn.close()

        # Describe the table
        tools = SQLTools()
        result = await tools._describe_table(db_path, "test_table")

        # Verify all column names and types appear
        for col_name, col_type in columns:
            assert col_name in result, (
                f"Expected column '{col_name}' in result: {result}"
            )
            assert col_type in result, (
                f"Expected type '{col_type}' in result: {result}"
            )

        # Also verify the 'id' primary key column is present
        assert "id" in result
        assert "INTEGER" in result


# ---------------------------------------------------------------------------
# Property 16: SQL read-only mode blocks write operations
# ---------------------------------------------------------------------------


class TestSQLReadOnlyModeBlocksWriteOperations:
    """For any SQL write statement (INSERT, UPDATE, DELETE, CREATE) executed with
    read_only=True, the tool SHALL return an error. With read_only=False, the
    same statement SHALL succeed.

    **Validates: Requirements 7.7, 7.8**
    """

    @settings(max_examples=20, deadline=None)
    @given(
        name=safe_text_values,
        age=int_values,
    )
    async def test_insert_blocked_in_read_only(
        self, name: str, age: int
    ) -> None:
        # Create a database with a table
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.execute("INSERT INTO people VALUES (1, 'seed', 0)")
        conn.commit()
        conn.close()

        tools_ro = SQLTools(read_only=True)
        statement = f"INSERT INTO people (name, age) VALUES ('{name}', {age})"
        result = await tools_ro._run_sql(db_path, statement)
        assert "Error" in result, f"Expected error for read-only INSERT, got: {result}"

    @settings(max_examples=20, deadline=None)
    @given(
        name=safe_text_values,
        age=int_values,
    )
    async def test_insert_succeeds_when_not_read_only(
        self, name: str, age: int
    ) -> None:
        # Create a database with a table
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.commit()
        conn.close()

        tools_rw = SQLTools(read_only=False)
        statement = f"INSERT INTO people (name, age) VALUES ('{name}', {age})"
        result = await tools_rw._run_sql(db_path, statement)
        assert "Error" not in result, f"Expected success for writable INSERT, got: {result}"
        assert "Rows affected" in result

    @settings(max_examples=20, deadline=None)
    @given(
        write_type=st.sampled_from(["UPDATE", "DELETE"]),
    )
    async def test_update_delete_blocked_in_read_only(
        self, write_type: str
    ) -> None:
        # Create a database with seed data
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.execute("INSERT INTO people VALUES (1, 'Alice', 30)")
        conn.commit()
        conn.close()

        tools_ro = SQLTools(read_only=True)
        if write_type == "UPDATE":
            statement = "UPDATE people SET age = 99 WHERE id = 1"
        else:
            statement = "DELETE FROM people WHERE id = 1"

        result = await tools_ro._run_sql(db_path, statement)
        assert "Error" in result, (
            f"Expected error for read-only {write_type}, got: {result}"
        )

    @settings(max_examples=20, deadline=None)
    @given(
        table_name=sql_identifiers,
    )
    async def test_create_table_blocked_in_read_only(
        self, table_name: str
    ) -> None:
        # Create an empty database
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.close()

        tools_ro = SQLTools(read_only=True)
        statement = f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY)"
        result = await tools_ro._run_sql(db_path, statement)
        assert "Error" in result, (
            f"Expected error for read-only CREATE TABLE, got: {result}"
        )


# ---------------------------------------------------------------------------
# Property 17: Missing database returns error
# ---------------------------------------------------------------------------


class TestMissingDatabaseReturnsError:
    """For any non-existent database path, SQL tool functions SHALL return an
    error message indicating the database was not found.

    **Validates: Requirements 7.4**
    """

    @settings(max_examples=20, deadline=None)
    @given(
        filename=st.text(
            alphabet=string.ascii_letters + string.digits + "_",
            min_size=1,
            max_size=20,
        ),
    )
    async def test_run_sql_missing_db(self, filename: str) -> None:
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, f"{filename}.db")
        tools = SQLTools()
        result = await tools._run_sql(db_path, "SELECT 1")
        assert "Error" in result
        assert "not found" in result

    @settings(max_examples=20, deadline=None)
    @given(
        filename=st.text(
            alphabet=string.ascii_letters + string.digits + "_",
            min_size=1,
            max_size=20,
        ),
    )
    async def test_list_tables_missing_db(self, filename: str) -> None:
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, f"{filename}.db")
        tools = SQLTools()
        result = await tools._list_tables(db_path)
        assert "Error" in result
        assert "not found" in result

    @settings(max_examples=20, deadline=None)
    @given(
        filename=st.text(
            alphabet=string.ascii_letters + string.digits + "_",
            min_size=1,
            max_size=20,
        ),
    )
    async def test_describe_table_missing_db(self, filename: str) -> None:
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, f"{filename}.db")
        tools = SQLTools()
        result = await tools._describe_table(db_path, "any_table")
        assert "Error" in result
        assert "not found" in result


# ---------------------------------------------------------------------------
# Property 18: Malformed SQL returns SQLite error
# ---------------------------------------------------------------------------


class TestMalformedSQLReturnsSQLiteError:
    """For any syntactically invalid SQL query, run_sql SHALL return a result
    containing the SQLite error message.

    **Validates: Requirements 7.5**
    """

    @settings(max_examples=20, deadline=None)
    @given(
        garbled=st.text(
            alphabet=string.ascii_letters + string.digits + " !@#$%^&*",
            min_size=5,
            max_size=50,
        ),
    )
    async def test_malformed_sql_returns_error(
        self, garbled: str
    ) -> None:
        # Filter out strings that might accidentally be valid SQL
        assume(not garbled.strip().upper().startswith("SELECT"))
        assume(not garbled.strip().upper().startswith("INSERT"))
        assume(not garbled.strip().upper().startswith("UPDATE"))
        assume(not garbled.strip().upper().startswith("DELETE"))
        assume(not garbled.strip().upper().startswith("CREATE"))
        assume(not garbled.strip().upper().startswith("DROP"))
        assume(not garbled.strip().upper().startswith("ALTER"))
        assume(not garbled.strip().upper().startswith("PRAGMA"))
        assume(not garbled.strip().upper().startswith("BEGIN"))
        assume(not garbled.strip().upper().startswith("COMMIT"))
        assume(not garbled.strip().upper().startswith("ROLLBACK"))
        assume(len(garbled.strip()) > 0)

        # Create a valid database
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        conn.close()

        tools = SQLTools()
        result = await tools._run_sql(db_path, garbled)
        assert "Error" in result, (
            f"Expected error for malformed SQL '{garbled}', got: {result}"
        )
