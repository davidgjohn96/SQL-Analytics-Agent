"""SQL Analytics Agent — Streamlit interface.

Run locally with:  streamlit run app.py
Deployed on Streamlit Community Cloud (see README.md § Deploy).
"""

from __future__ import annotations

import os
import sys

# Ensure this app's own directory (sql-agent/) is on the import path. On
# Streamlit Community Cloud the app runs from the repo root, so without this the
# sibling `agent` / `database` packages may not be importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# override=True so edits to .env are picked up on rerun (not masked by a stale
# value cached in the process env). No-op on Streamlit Cloud, which has no .env.
load_dotenv(override=True)

# Import after load_dotenv so tracing/LLM pick up env vars. Importing graph
# also initializes Arize tracing exactly once (all spans go to the single
# project named by ARIZE_PROJECT_NAME).
from agent import tools  # noqa: E402
from agent.graph import run_agent  # noqa: E402
from agent.observability import flush, is_tracing_active, tracing_status  # noqa: E402

st.set_page_config(page_title="SQL Analytics Agent", page_icon="📊", layout="wide")

st.title("📊 SQL Analytics Agent")
st.caption(
    "Ask a business question in plain English. The agent finds the right tables, "
    "writes read-only SQL, validates it, runs it against the retail database, "
    "and explains the results — no SQL knowledge required."
)

EXAMPLE_QUESTIONS = [
    "What are the top 5 products by revenue?",
    "What is the average order value?",
    "How many orders were placed in each state?",
    "Which customers have never placed an order?",
    "What is total revenue by month?",
    "Who are the 10 most valuable customers?",
]


def _trace_status() -> str:
    # Reflects the real tracer state (whether register() actually succeeded),
    # not just whether env vars are present.
    return f"Tracing: {tracing_status()}"


@st.cache_data(show_spinner=False)
def _schema_overview() -> list[dict]:
    """Cached so we reflect the DB once per session, not on every rerun."""
    return tools.schema_overview()


def _render_schema_browser() -> None:
    """Show the tables, columns, and relationships users can query."""
    overview = _schema_overview()

    st.markdown(
        "This assistant answers questions about a **sample retail dataset**. "
        "Here's everything it can see — use it to phrase your question."
    )

    # Quick at-a-glance row counts.
    metric_cols = st.columns(len(overview))
    for col, table in zip(metric_cols, overview):
        rows = f"{table['row_count']:,}" if table["row_count"] is not None else "—"
        col.metric(label=table["name"], value=rows, help=table["purpose"])
    st.caption("Row counts per table.")

    # One tab per table with its columns.
    tabs = st.tabs([t["name"] for t in overview])
    for tab, table in zip(tabs, overview):
        with tab:
            st.markdown(f"**{table['purpose']}** · {table['row_count']:,} rows")
            df = pd.DataFrame(
                [
                    {
                        "Column": c["name"],
                        "Type": c["type"],
                        "Key": "🔑 PK"
                        if c["pk"]
                        else (f"→ {c['fk']}" if c["fk"] else ""),
                    }
                    for c in table["columns"]
                ]
            )
            st.dataframe(df, width="stretch", hide_index=True)

    st.markdown(
        "**How the tables connect**\n"
        "- `customers` → `orders` (one customer has many orders)\n"
        "- `orders` → `order_items` (one order has many line items)\n"
        "- `products` → `order_items` (one product appears in many line items)\n\n"
        "**Handy definitions** — *revenue* = `quantity × unit_price`, "
        "*order value* = `orders.total_amount`. Data spans **2023-01-01 → 2024-12-31**."
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Model:** `{os.getenv('OPENAI_MODEL', 'openai/gpt-oss-120b')}`")
    if is_tracing_active():
        st.success(_trace_status(), icon="🔭")
    else:
        st.warning(_trace_status(), icon="⚠️")

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Picks the relevant tables\n"
        "2. Generates read-only SQL\n"
        "3. Validates it (SELECT-only)\n"
        "4. Runs it on the database\n"
        "5. Explains the results"
    )


# --------------------------------------------------------------------------- #
# Onboarding: what you can ask
# --------------------------------------------------------------------------- #
with st.expander("📚 What you can ask about (database schema)", expanded=True):
    _render_schema_browser()


# --------------------------------------------------------------------------- #
# Question input
# --------------------------------------------------------------------------- #
if not os.getenv("OPENAI_API_KEY"):
    st.warning(
        "No `OPENAI_API_KEY` found. Set it in your environment (local) or in "
        "**App settings → Secrets** (Streamlit Cloud) to run queries.",
        icon="🔑",
    )

# Let example buttons prefill the question box via session state.
if "question" not in st.session_state:
    st.session_state["question"] = ""

st.markdown("**Try an example:**")
example_cols = st.columns(3)
for i, example in enumerate(EXAMPLE_QUESTIONS):
    if example_cols[i % 3].button(example, width="stretch", key=f"ex_{i}"):
        st.session_state["question"] = example

question = st.text_input(
    "Business question",
    key="question",
    placeholder="e.g. What are the top 5 products by revenue?",
)
run = st.button("Run", type="primary")


if run and question.strip():
    with st.spinner("Thinking..."):
        try:
            state = run_agent(question.strip())
        except Exception as exc:  # noqa: BLE001
            st.error(f"The agent failed: {exc}")
            st.stop()
        finally:
            # Long-running server: push this run's spans to Arize now instead of
            # waiting on the batch timer (which can drop spans if the container
            # idles). No-op when tracing is disabled.
            flush()

    # --- Summary at the top ---
    st.subheader("Summary")
    st.write(state.get("final_summary", "_No summary produced._"))

    validation_error = state.get("validation_error")
    execution_error = state.get("execution_error")
    if validation_error:
        st.warning(f"Validation failed: {validation_error}")
    elif execution_error:
        st.warning(f"Execution failed: {execution_error}")

    # --- Query results ---
    with st.expander("Query Results", expanded=not (validation_error or execution_error)):
        columns = state.get("columns", [])
        rows = state.get("rows", [])
        if columns and rows:
            df = pd.DataFrame(rows, columns=columns)
            st.dataframe(df, width="stretch")
            st.caption(f"{len(rows)} row(s) returned.")
        elif columns:
            st.info("Query ran successfully but returned no rows.")
        else:
            st.info("No results to display.")

    # --- Generated SQL ---
    with st.expander("Generated SQL"):
        st.code(state.get("validated_sql") or state.get("generated_sql", ""), language="sql")

    # --- Agent trace (per-node state) ---
    with st.expander("Agent Trace"):
        st.markdown("**Relevant tables**")
        st.write(state.get("relevant_tables", []))
        st.markdown("**Schema context**")
        st.code(state.get("schema_context", ""), language="text")
        if os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY"):
            st.caption(
                "Full step-by-step spans (inputs, outputs, latency, model, "
                "prompts) are available in Arize AX."
            )

elif run:
    st.info("Please enter a question.")
