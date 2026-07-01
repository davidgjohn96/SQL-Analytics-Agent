"""Shared state object threaded through the LangGraph nodes."""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between nodes.

    Every node reads from and writes to this dict. `total=False` lets each node
    populate only the keys it produces.
    """

    # Input
    user_question: str

    # Node 1 — Retrieve Schema
    relevant_tables: list[str]
    schema_context: str

    # Node 2 — Generate SQL
    generated_sql: str

    # Node 3 — Validate SQL
    validated_sql: Optional[str]
    validation_error: Optional[str]

    # Node 4 — Execute SQL
    columns: list[str]
    rows: list[list[Any]]
    execution_error: Optional[str]

    # Node 5 — Generate Summary
    final_summary: str

    # Misc / control
    prompt_variant: str  # "A" or "B" — selects the SQL-generation prompt
