"""loomable.toolkits.sql_tools - SQLite database toolkit.

Provides tools for executing SQL queries, listing tables, and describing
table schemas against SQLite databases. Uses only the Python standard library
``sqlite3`` module — no external dependencies required.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit


class SQLTools(Toolkit):
    """SQLite database toolkit using the standard library sqlite3 module.

    By default, databases are opened in read-only mode to prevent accidental
    writes. Set ``read_only=False`` to allow write operations.

    Usage::

        from loomable.toolkits import SQLTools

        # Read-only (default)
        tools = SQLTools()

        # Allow writes
        tools = SQLTools(read_only=False)
    """

    def __init__(
        self,
        *,
        read_only: bool = True,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._read_only = read_only

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._run_sql, name="run_sql"),
            FunctionTool(self._list_tables, name="list_tables"),
            FunctionTool(self._describe_table, name="describe_table"),
        ]

    def _connect(self, db_path: str) -> sqlite3.Connection:
        """Open a connection to the SQLite database.

        In read-only mode, uses URI format with ``?mode=ro``.
        In write mode, opens normally with ``sqlite3.connect(db_path)``.

        Raises FileNotFoundError if the database does not exist in read-only mode.
        """
        if self._read_only:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database not found: {db_path}")
            uri = f"file:{db_path}?mode=ro"
            return sqlite3.connect(uri, uri=True)
        else:
            return sqlite3.connect(db_path)

    def _execute_run_sql(self, db_path: str, query: str) -> str:
        """Synchronous implementation for run_sql."""
        conn = self._connect(db_path)
        try:
            cursor = conn.execute(query)
            if cursor.description is not None:
                # SELECT query — return formatted rows
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                if not rows:
                    return f"Columns: {', '.join(columns)}\n(no rows)"
                lines = [" | ".join(columns)]
                lines.append("-" * len(lines[0]))
                for row in rows:
                    lines.append(" | ".join(str(v) for v in row))
                return "\n".join(lines)
            else:
                # Non-SELECT (INSERT, UPDATE, DELETE, etc.)
                conn.commit()
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"
        finally:
            conn.close()

    def _execute_list_tables(self, db_path: str) -> str:
        """Synchronous implementation for list_tables."""
        conn = self._connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            if not tables:
                return "(no tables)"
            return "\n".join(tables)
        finally:
            conn.close()

    def _execute_describe_table(self, db_path: str, table: str) -> str:
        """Synchronous implementation for describe_table."""
        conn = self._connect(db_path)
        try:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            if not columns:
                return f"Error: Table '{table}' not found or has no columns"
            # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
            lines = ["name | type | nullable | default | primary_key"]
            lines.append("-" * len(lines[0]))
            for col in columns:
                cid, name, col_type, notnull, default, pk = col
                nullable = "NO" if notnull else "YES"
                default_str = str(default) if default is not None else "NULL"
                pk_str = "YES" if pk else "NO"
                lines.append(f"{name} | {col_type} | {nullable} | {default_str} | {pk_str}")
            return "\n".join(lines)
        finally:
            conn.close()

    async def _run_sql(self, db_path: str, query: str) -> str:
        """Execute a SQL query against a SQLite database and return results.

        For SELECT queries, returns formatted rows with column headers.
        For other queries (INSERT, UPDATE, DELETE), returns the number of
        rows affected.
        """
        try:
            return await asyncio.to_thread(self._execute_run_sql, db_path, query)
        except FileNotFoundError:
            return f"Error: Database not found: {db_path}"
        except sqlite3.OperationalError as exc:
            return f"Error: {exc}"
        except sqlite3.DatabaseError as exc:
            return f"Error: {exc}"

    async def _list_tables(self, db_path: str) -> str:
        """List all tables in a SQLite database."""
        try:
            return await asyncio.to_thread(self._execute_list_tables, db_path)
        except FileNotFoundError:
            return f"Error: Database not found: {db_path}"
        except sqlite3.OperationalError as exc:
            return f"Error: {exc}"
        except sqlite3.DatabaseError as exc:
            return f"Error: {exc}"

    async def _describe_table(self, db_path: str, table: str) -> str:
        """Describe the columns and types of a table."""
        try:
            return await asyncio.to_thread(
                self._execute_describe_table, db_path, table
            )
        except FileNotFoundError:
            return f"Error: Database not found: {db_path}"
        except sqlite3.OperationalError as exc:
            return f"Error: {exc}"
        except sqlite3.DatabaseError as exc:
            return f"Error: {exc}"
