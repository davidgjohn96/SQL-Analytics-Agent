"""Non-LLM tools used by the agent nodes: schema access, SQL validation,
and SQL execution. Kept free of LangGraph/LLM concerns so they're easy to test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp
from sqlalchemy import inspect, text

from database.schema import get_engine

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")
_SCHEMA_DESC_PATH = os.path.join(_DB_DIR, "schema_description.md")

# Concise per-table summaries used during schema selection (Node 1).
TABLE_SUMMARIES: dict[str, str] = {
    "customers": "Customer profiles: customer_id, first_name, last_name, state, signup_date.",
    "products": "Product catalog: product_id, product_name, category, price.",
    "orders": "Orders header: order_id, customer_id, order_date, total_amount.",
    "order_items": "Order line items: order_item_id, order_id, product_id, quantity, unit_price.",
}


def actual_schema() -> dict[str, list[str]]:
    """Return {table_name: [column, ...]} reflected live from retail.db."""
    insp = inspect(get_engine())
    return {t: [c["name"] for c in insp.get_columns(t)] for t in insp.get_table_names()}


def load_schema_description() -> str:
    """Full human-readable schema description (markdown)."""
    with open(_SCHEMA_DESC_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def schema_overview() -> list[dict]:
    """Structured, display-friendly overview of every queryable table.

    Reflected live from retail.db (so it never drifts from the real schema) and
    enriched with a short purpose from TABLE_SUMMARIES. Each entry is::

        {
          "name": "orders",
          "purpose": "Orders header",
          "row_count": 1200,
          "columns": [{"name", "type", "pk": bool, "fk": "customers.customer_id"|None}, ...],
        }

    Used by the Streamlit UI to show users what they can ask about.
    """
    engine = get_engine()
    insp = inspect(engine)
    overview: list[dict] = []
    with engine.connect() as conn:
        for table in insp.get_table_names():
            pk_cols = set(insp.get_pk_constraint(table).get("constrained_columns") or [])
            fk_map: dict[str, str] = {}
            for fk in insp.get_foreign_keys(table):
                for local, remote in zip(
                    fk.get("constrained_columns", []), fk.get("referred_columns", [])
                ):
                    fk_map[local] = f"{fk['referred_table']}.{remote}"

            columns = [
                {
                    "name": c["name"],
                    "type": str(c["type"]),
                    "pk": c["name"] in pk_cols,
                    "fk": fk_map.get(c["name"]),
                }
                for c in insp.get_columns(table)
            ]

            try:
                row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            except Exception:  # noqa: BLE001
                row_count = None

            # TABLE_SUMMARIES text is "Purpose: col, col, ..."; take the purpose.
            purpose = TABLE_SUMMARIES.get(table, "").split(":")[0].strip()

            overview.append(
                {
                    "name": table,
                    "purpose": purpose,
                    "row_count": row_count,
                    "columns": columns,
                }
            )
    return overview


def schema_context_for_tables(tables: list[str]) -> str:
    """Build a compact schema context string for the given tables.

    Uses the live column lists so the LLM never sees stale column names.
    """
    schema = actual_schema()
    lines: list[str] = []
    for table in tables:
        if table not in schema:
            continue
        cols = ", ".join(schema[table])
        lines.append(f"TABLE {table}({cols})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_FORBIDDEN = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
    exp.Alter, exp.TruncateTable, exp.Command,
)


@dataclass
class ValidationResult:
    ok: bool
    error: str | None = None


def validate_sql(sql: str) -> ValidationResult:
    """Validate SQL with sqlglot.

    Checks: parses as valid SQLite, is a single SELECT (read-only), and every
    referenced table/column exists in the live schema.
    """
    if not sql or not sql.strip():
        return ValidationResult(False, "Empty SQL.")

    # 1) Parse (SQLite dialect).
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except Exception as exc:  # sqlglot.errors.ParseError and friends
        return ValidationResult(False, f"Syntax error: {exc}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return ValidationResult(False, "Exactly one SQL statement is allowed.")

    tree = statements[0]

    # 2) Read-only: must be a SELECT and contain no mutating/DDL nodes.
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery, exp.With)):
        return ValidationResult(False, "Only SELECT queries are allowed.")
    for node_type in _FORBIDDEN:
        if tree.find(node_type) is not None:
            return ValidationResult(False, "Only read-only SELECT queries are allowed.")

    # 3) Referenced tables exist (CTE names defined in this query count too).
    schema = actual_schema()
    cte_names = {c.alias_or_name for c in tree.find_all(exp.CTE)}
    known_tables = set(schema) | cte_names
    # Map alias/name -> real table name for column checks.
    referenced: dict[str, str] = {}
    for tbl in tree.find_all(exp.Table):
        name = tbl.name
        if name in cte_names:
            continue
        if name not in known_tables:
            return ValidationResult(False, f"Unknown table: {name}")
        alias = tbl.alias_or_name
        referenced[alias] = name
        referenced[name] = name

    if not referenced:
        return ValidationResult(False, "No tables referenced.")

    # 4) Referenced columns exist (only check columns that are table-qualified
    #    or unambiguous; this keeps validation strict but avoids false errors
    #    on aliases/expressions).
    all_columns = {c for cols in schema.values() for c in cols}
    # SELECT-defined aliases (e.g. "... AS total_revenue") and CTE names are
    # valid identifiers even though they aren't physical columns.
    defined_aliases = {
        a.alias for a in tree.find_all(exp.Alias) if a.alias
    }
    allowed_unqualified = all_columns | defined_aliases | cte_names

    for col in tree.find_all(exp.Column):
        col_name = col.name
        if col_name == "*" or col_name == "":
            continue
        table_ref = col.table  # qualifier, e.g. "oi" in oi.quantity
        if table_ref:
            real = referenced.get(table_ref)
            if real is None:
                # Could be a CTE alias rather than a base table.
                if table_ref in cte_names:
                    continue
                return ValidationResult(False, f"Unknown table alias: {table_ref}")
            if col_name not in schema[real]:
                return ValidationResult(False, f"Unknown column: {table_ref}.{col_name}")
        else:
            # Unqualified column: must exist in a referenced table or be an alias.
            if col_name not in allowed_unqualified:
                return ValidationResult(False, f"Unknown column: {col_name}")

    return ValidationResult(True)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    columns: list[str]
    rows: list[list[Any]]
    error: str | None = None


def execute_sql(sql: str, row_limit: int = 1000) -> ExecutionResult:
    """Execute a validated SELECT and return columns + rows.

    A defensive row cap is applied so a missing LIMIT can't return the whole
    table to the UI.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [list(r) for r in result.fetchmany(row_limit)]
        return ExecutionResult(columns=columns, rows=rows)
    except Exception as exc:  # noqa: BLE001 — surface any DB error to state
        return ExecutionResult(columns=[], rows=[], error=str(exc))
