"""SQLite tool for the Northwind database.

This module provides a small, focused API around the local
`data/northwind.sqlite` file required by the assignment.

It is designed to be called from LangGraph nodes, DSPy modules, or
simple scripts. The main responsibilities are:

- Open a connection to the SQLite DB (read-only by default).
- Execute SELECT queries and return rows as list[dict].
- Introspect the schema using PRAGMA to build a text description that can be
  passed into NL→SQL modules.

Example usage:

    from agent.tools.sqlite_tool import (
        get_connection,
        execute_sql,
        get_schema_description,
    )

    rows = execute_sql("SELECT * FROM Customers LIMIT 3")
    schema_str = get_schema_description()

The LangGraph SQL executor node can simply call `execute_sql` and attach the
`sql` string + any errors to the agent state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Default path to the Northwind DB (relative to project root)
DEFAULT_DB_PATH = Path("data") / "northwind.sqlite"


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return a *new* SQLite connection to the Northwind DB.

    The connection is opened in read-only mode when possible to avoid
    accidental writes. For older SQLite versions that don't support the
    URI/read-only trick, it falls back to normal mode.
    """

    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {db_path}. Did you download northwind.sqlite?"
        )

    # Try read-only URI mode first (preferred for safety)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError:
        # Fallback: normal read-write mode (still fine if caller only does SELECTs)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def execute_sql(sql: str, params: Optional[Iterable[Any]] = None,
                db_path: Path | str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Execute a SQL query and return rows as a list of dicts.

    This is intended primarily for SELECT queries. Other statements will
    execute but their effects are not committed (connection is closed after
    execution).

    Raises sqlite3.Error if execution fails. The LangGraph SQL executor
    node should catch and record the error message in the agent state.
    """

    if not sql or not sql.strip():
        raise ValueError("SQL query is empty")

    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        if params is None:
            cur.execute(sql)
        else:
            cur.execute(sql, list(params))

        # Fetch all rows
        rows = cur.fetchall()

        # Convert sqlite3.Row objects → plain dicts
        result: List[Dict[str, Any]] = []
        for row in rows:
            result.append({k: row[k] for k in row.keys()})
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------

def list_tables(db_path: Path | str = DEFAULT_DB_PATH) -> List[str]:
    """Return a list of table names in the SQLite database.

    Skips SQLite internal tables (names starting with `sqlite_`).
    """

    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_table_schema(table: str, db_path: Path | str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Return PRAGMA table_info for a table as a list of dicts.

    Each item contains: cid, name, type, notnull, dflt_value, pk
    """

    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info('{table}')")
        rows = cur.fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            result.append({
                "cid": row[0],
                "name": row[1],
                "type": row[2],
                "notnull": row[3],
                "dflt_value": row[4],
                "pk": row[5],
            })
        return result
    finally:
        conn.close()


def get_schema_description(db_path: Path | str = DEFAULT_DB_PATH) -> str:
    """Return a compact text description of relevant tables & columns.

    This string is meant to be fed into NL→SQL DSPy modules so the model
    knows which tables/columns are available.

    Example output (simplified):

        Orders(OrderID, CustomerID, EmployeeID, OrderDate, ...)
        "Order Details"(OrderID, ProductID, UnitPrice, Quantity, Discount)
        Products(ProductID, ProductName, SupplierID, CategoryID, UnitPrice, ...)
        Customers(CustomerID, CompanyName, Country, ...)

    You can further restrict/format this according to the assignment's
    suggested canonical table names.
    """

    tables = list_tables(db_path)

    # Focus on canonical Northwind tables used in the assignment
    preferred_order = [
        "Orders",
        "Order Details",
        "Products",
        "Customers",
        "Categories",
        "Suppliers",
    ]

    # Keep any additional tables at the end
    tables_sorted = [t for t in preferred_order if t in tables] + [
        t for t in tables if t not in preferred_order
    ]

    lines: List[str] = []
    for table in tables_sorted:
        cols = get_table_schema(table, db_path)
        col_names = ", ".join(col["name"] for col in cols)
        # Quote table name if it contains spaces to match SQLite syntax
        if " " in table:
            table_repr = f'"{table}"'
        else:
            table_repr = table
        lines.append(f"{table_repr}({col_names})")

    return "\n".join(lines)


if __name__ == "__main__":  # simple manual test/debug
    schema = get_schema_description()
    print("Schema description:\n")
    print(schema)

    print("\nExample query (top 3 customers):")
    example_sql = "SELECT CustomerID, COUNT(*) as num_orders FROM Orders GROUP BY CustomerID ORDER BY num_orders DESC LIMIT 3;"
    for row in execute_sql(example_sql):
        print(row)
