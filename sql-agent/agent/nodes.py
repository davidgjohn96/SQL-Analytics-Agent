"""The five LangGraph nodes.

Each node reads/writes the shared AgentState. Spans are produced entirely by
auto-instrumentation: LangChainInstrumentor emits one span per LangGraph node
plus the underlying ChatOpenAI LLM call. Nodes no longer create explicit spans
of their own — see agent/observability.py for the instrumentor setup.
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent import prompts, tools
from agent.state import AgentState

ALL_TABLES = ["customers", "products", "orders", "order_items"]


def _model_name() -> str:
    # Default: OpenAI's open-weight gpt-oss-120b, served by Groq.
    return os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")


def _temperature() -> float:
    try:
        return float(os.getenv("OPENAI_TEMPERATURE", "0"))
    except ValueError:
        return 0.0


def _llm() -> ChatOpenAI:
    """Build the chat model.

    Uses an OpenAI-compatible endpoint. `OPENAI_BASE_URL` points at the provider
    (default: Groq, which serves open-weight models); `OPENAI_API_KEY` holds that
    provider's key. Leaving `OPENAI_BASE_URL` unset falls back to OpenAI itself.
    """
    kwargs: dict = {"model": _model_name(), "temperature": _temperature()}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _strip_sql_fences(text: str) -> str:
    """Remove ```sql ... ``` fences the model sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.replace("```sql", "").replace("```", "")
    return text.strip().rstrip(";").strip() + ";"


# ---------------------------------------------------------------------------
# Node 1 — Retrieve Schema
# ---------------------------------------------------------------------------

def retrieve_schema(state: AgentState) -> AgentState:
    question = state["user_question"]
    table_menu = "\n".join(
        f"- {name}: {summary}" for name, summary in tools.TABLE_SUMMARIES.items()
    )
    llm = _llm()
    messages = [
        SystemMessage(content=prompts.SCHEMA_SELECTION_SYSTEM),
        HumanMessage(content=f"Available tables:\n{table_menu}\n\nQuestion: {question}"),
    ]
    raw = llm.invoke(messages).content.strip()

    # Parse the JSON array of table names; fall back to all tables.
    tables = _parse_tables(raw)
    schema_context = tools.schema_context_for_tables(tables)

    return {"relevant_tables": tables, "schema_context": schema_context}


def _parse_tables(raw: str) -> list[str]:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        tables = [t for t in parsed if t in ALL_TABLES]
        return tables or ALL_TABLES
    except Exception:  # noqa: BLE001
        # Heuristic fallback: keep any table name mentioned in the response.
        tables = [t for t in ALL_TABLES if t in cleaned]
        return tables or ALL_TABLES


# ---------------------------------------------------------------------------
# Node 2 — Generate SQL
# ---------------------------------------------------------------------------

def generate_sql(state: AgentState) -> AgentState:
    question = state["user_question"]
    schema_context = state.get("schema_context", "")
    variant = state.get("prompt_variant", "B")

    system_prompt = prompts.sql_generation_prompt(variant)
    user_content = f"Schema:\n{schema_context}\n\nQuestion: {question}\n\nSQL:"

    llm = _llm()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    raw = llm.invoke(messages).content
    sql = _strip_sql_fences(raw)
    return {"generated_sql": sql}


# ---------------------------------------------------------------------------
# Node 3 — Validate SQL
# ---------------------------------------------------------------------------

def validate_sql(state: AgentState) -> AgentState:
    sql = state.get("generated_sql", "")
    result = tools.validate_sql(sql)
    if result.ok:
        return {"validated_sql": sql, "validation_error": None}
    return {"validated_sql": None, "validation_error": result.error}


# ---------------------------------------------------------------------------
# Node 4 — Execute SQL
# ---------------------------------------------------------------------------

def execute_sql(state: AgentState) -> AgentState:
    sql = state.get("validated_sql")
    result = tools.execute_sql(sql)
    if result.error:
        return {
            "columns": [],
            "rows": [],
            "execution_error": result.error,
        }
    return {
        "columns": result.columns,
        "rows": result.rows,
        "execution_error": None,
    }


# ---------------------------------------------------------------------------
# Node 5 — Generate Business Summary
# ---------------------------------------------------------------------------

def generate_summary(state: AgentState) -> AgentState:
    question = state["user_question"]
    columns = state.get("columns", [])
    rows = state.get("rows", [])
    sql = state.get("validated_sql") or state.get("generated_sql", "")

    # Cap rows fed to the LLM to keep the prompt small.
    preview_rows = rows[:50]
    results_blob = json.dumps({"columns": columns, "rows": preview_rows}, default=str)
    user_content = (
        f"Question: {question}\n\nSQL run:\n{sql}\n\n"
        f"Results (up to 50 rows):\n{results_blob}"
    )

    llm = _llm()
    messages = [
        SystemMessage(content=prompts.SUMMARY_SYSTEM),
        HumanMessage(content=user_content),
    ]
    summary = llm.invoke(messages).content.strip()
    return {"final_summary": summary}


# ---------------------------------------------------------------------------
# Error summary (used when validation/execution fails — no LLM call needed)
# ---------------------------------------------------------------------------

def error_summary(state: AgentState) -> AgentState:
    err = state.get("validation_error") or state.get("execution_error") or "Unknown error."
    summary = (
        "I couldn't produce a result for that question. "
        f"The query could not be {'validated' if state.get('validation_error') else 'executed'}: {err}"
    )
    return {"final_summary": summary}
