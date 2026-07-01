"""LangGraph wiring for the SQL analytics agent.

Flow:

    retrieve_schema -> generate_sql -> validate_sql --(valid)--> execute_sql
                                                  \\--(invalid)--> error_summary
    execute_sql --(ok)--> generate_summary --> END
    execute_sql --(error)--> error_summary --> END

The validation/execution branches ensure invalid SQL is never executed and that
failures still return a coherent response.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent import nodes
from agent.observability import setup_tracing
from agent.state import AgentState

# Initialize tracing as soon as the graph module is imported (before any LLM
# client is constructed inside the nodes).
setup_tracing()


def _after_validation(state: AgentState) -> str:
    return "execute" if state.get("validated_sql") else "error"


def _after_execution(state: AgentState) -> str:
    return "summary" if not state.get("execution_error") else "error"


def build_graph():
    """Construct and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_schema", nodes.retrieve_schema)
    graph.add_node("generate_sql", nodes.generate_sql)
    graph.add_node("validate_sql", nodes.validate_sql)
    graph.add_node("execute_sql", nodes.execute_sql)
    graph.add_node("generate_summary", nodes.generate_summary)
    graph.add_node("error_summary", nodes.error_summary)

    graph.set_entry_point("retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")

    graph.add_conditional_edges(
        "validate_sql",
        _after_validation,
        {"execute": "execute_sql", "error": "error_summary"},
    )
    graph.add_conditional_edges(
        "execute_sql",
        _after_execution,
        {"summary": "generate_summary", "error": "error_summary"},
    )

    graph.add_edge("generate_summary", END)
    graph.add_edge("error_summary", END)

    return graph.compile()


# Compile once and reuse.
_COMPILED = None


def get_agent():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def run_agent(question: str, prompt_variant: str = "B") -> AgentState:
    """Run the full agent for one question and return the final state."""
    agent = get_agent()
    initial: AgentState = {"user_question": question, "prompt_variant": prompt_variant}
    return agent.invoke(initial)
