"""The five LangGraph nodes.

Each node:
  * reads/writes the shared AgentState,
  * is wrapped in an explicit OpenInference span (so it shows up in Arize with
    its kind, input, output, latency, and any error), and
  * for LLM nodes, records the prompt and model on the span.

LLM API calls are additionally auto-instrumented by OpenAIInstrumentor, so each
generation appears as a child LLM span under the node's CHAIN span.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent import prompts, tools
from agent.observability import get_tracer
from agent.state import AgentState

ALL_TABLES = ["customers", "products", "orders", "order_items"]


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o")


def _temperature() -> float:
    try:
        return float(os.getenv("OPENAI_TEMPERATURE", "0"))
    except ValueError:
        return 0.0


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=_model_name(), temperature=_temperature())


@contextmanager
def _span(name: str, kind: str, input_value: str, extra: dict | None = None):
    """Open an OpenInference span. No-op if tracing is disabled."""
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", kind)
        span.set_attribute("input.value", input_value)
        for key, val in (extra or {}).items():
            span.set_attribute(key, val)
        try:
            yield span
        except Exception as exc:  # record + re-raise
            span.record_exception(exc)
            span.set_attribute("error", str(exc))
            raise


def _set_output(span, value: str) -> None:
    if span is not None:
        span.set_attribute("output.value", value)


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
    with _span("Schema Retrieval", "CHAIN", question) as span:
        table_menu = "\n".join(
            f"- {name}: {summary}" for name, summary in tools.TABLE_SUMMARIES.items()
        )
        llm = _llm()
        if span is not None:
            span.set_attribute("llm.model_name", _model_name())
        messages = [
            SystemMessage(content=prompts.SCHEMA_SELECTION_SYSTEM),
            HumanMessage(content=f"Available tables:\n{table_menu}\n\nQuestion: {question}"),
        ]
        raw = llm.invoke(messages).content.strip()

        # Parse the JSON array of table names; fall back to all tables.
        tables = _parse_tables(raw)
        schema_context = tools.schema_context_for_tables(tables)

        _set_output(span, json.dumps({"relevant_tables": tables}))
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

    with _span(
        "SQL Generation",
        "LLM",
        user_content,
        extra={
            "llm.model_name": _model_name(),
            "llm.prompt_variant": variant,
            "llm.system_prompt": system_prompt,
        },
    ) as span:
        llm = _llm()
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        raw = llm.invoke(messages).content
        sql = _strip_sql_fences(raw)
        _set_output(span, sql)
        return {"generated_sql": sql}


# ---------------------------------------------------------------------------
# Node 3 — Validate SQL
# ---------------------------------------------------------------------------

def validate_sql(state: AgentState) -> AgentState:
    sql = state.get("generated_sql", "")
    with _span("SQL Validation", "TOOL", sql) as span:
        result = tools.validate_sql(sql)
        if result.ok:
            _set_output(span, "valid")
            return {"validated_sql": sql, "validation_error": None}
        _set_output(span, f"invalid: {result.error}")
        if span is not None:
            span.set_attribute("error", result.error or "validation failed")
        return {"validated_sql": None, "validation_error": result.error}


# ---------------------------------------------------------------------------
# Node 4 — Execute SQL
# ---------------------------------------------------------------------------

def execute_sql(state: AgentState) -> AgentState:
    sql = state.get("validated_sql")
    with _span("SQL Execution", "TOOL", sql or "") as span:
        result = tools.execute_sql(sql)
        if result.error:
            _set_output(span, f"error: {result.error}")
            if span is not None:
                span.set_attribute("error", result.error)
            return {
                "columns": [],
                "rows": [],
                "execution_error": result.error,
            }
        _set_output(
            span,
            json.dumps({"columns": result.columns, "row_count": len(result.rows)}),
        )
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

    with _span(
        "Summary Generation",
        "LLM",
        user_content,
        extra={"llm.model_name": _model_name(), "llm.system_prompt": prompts.SUMMARY_SYSTEM},
    ) as span:
        llm = _llm()
        messages = [
            SystemMessage(content=prompts.SUMMARY_SYSTEM),
            HumanMessage(content=user_content),
        ]
        summary = llm.invoke(messages).content.strip()
        _set_output(span, summary)
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
