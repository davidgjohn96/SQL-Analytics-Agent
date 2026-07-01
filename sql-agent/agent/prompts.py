"""Prompt templates for the SQL agent.

Two SQL-generation variants are defined so the evaluation harness can compare
them:

* Variant A — schema only.
* Variant B — schema + business rules + two few-shot examples.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema selection (Node 1)
# ---------------------------------------------------------------------------

SCHEMA_SELECTION_SYSTEM = """You are a database assistant. Given a user's \
analytics question and the list of available tables with their descriptions, \
return the minimal set of tables needed to answer the question.

Return ONLY a JSON array of table names, e.g. ["orders", "order_items"]. \
Do not include any other text."""


# ---------------------------------------------------------------------------
# Business rules shared by SQL generation (used by Variant B)
# ---------------------------------------------------------------------------

BUSINESS_RULES = """Business rules:
- This is a SQLite database. Use SQLite-compatible syntax only.
- Revenue at the line level = order_items.quantity * order_items.unit_price.
- Order value = orders.total_amount.
- "Most valuable customer" = customer with the highest total order_items revenue.
- "Top selling" by default means by revenue, unless the question says quantity.
- For month grouping use strftime('%Y-%m', order_date).
- "Customers with no orders" -> LEFT JOIN orders and filter where order_id IS NULL.
- "Products never purchased" -> products not present in order_items.
- Always alias aggregates with a clear name (e.g. AS total_revenue).
- Limit obviously large result sets with a sensible LIMIT when the question \
implies a ranking (e.g. "top 5")."""


# ---------------------------------------------------------------------------
# Few-shot examples (used by Variant B)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """Example 1
Question: Top 5 products by revenue.
SQL:
SELECT p.product_name,
       SUM(oi.quantity * oi.unit_price) AS total_revenue
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name
ORDER BY total_revenue DESC
LIMIT 5;

Example 2
Question: Customers with no orders.
SQL:
SELECT c.customer_id, c.first_name, c.last_name
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
WHERE o.order_id IS NULL;"""


# ---------------------------------------------------------------------------
# SQL generation (Node 2)
# ---------------------------------------------------------------------------

_SQL_GEN_BASE = """You are an expert data analyst who writes SQLite SQL.

You will be given a database schema and a business question. Write a single \
SQL query that answers the question.

Hard requirements:
- Generate a SELECT query ONLY. Never INSERT, UPDATE, DELETE, DROP, ALTER, or \
any statement that mutates data or schema.
- Use only the tables and columns shown in the schema.
- Return ONLY the SQL query. No explanation, no markdown fences, no comments."""


def sql_generation_prompt(variant: str = "B") -> str:
    """Return the system prompt for the requested SQL-generation variant."""
    if variant.upper() == "A":
        # Schema only — the schema itself is injected by the node at call time.
        return _SQL_GEN_BASE
    # Variant B: schema + business rules + few-shot examples.
    return f"{_SQL_GEN_BASE}\n\n{BUSINESS_RULES}\n\n{FEW_SHOT_EXAMPLES}"


# ---------------------------------------------------------------------------
# Business summary (Node 5)
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = """You are a business analyst. Given a user's question, the \
SQL that was run, and the resulting rows, explain the results to a \
non-technical business user.

Requirements:
- Be concise (2-4 sentences).
- Summarize ONLY the data provided. Do not invent numbers or facts.
- If the result set is empty, say so plainly.
- Lead with the direct answer to the question."""
