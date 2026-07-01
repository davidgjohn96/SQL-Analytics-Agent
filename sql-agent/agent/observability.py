"""Arize AX tracing setup.

Registers an OpenTelemetry tracer that exports to Arize AX and attaches the
OpenInference instrumentors for LangChain/LangGraph and OpenAI. Auto-
instrumentation gives one span per LangGraph node plus the underlying LLM call
spans; nodes additionally enrich their spans with explicit input/output/error
attributes (see nodes.py).

Call `setup_tracing()` exactly once, before any LLM client is created.
Credentials are read from the environment — never hard-coded.
"""

from __future__ import annotations

import os

_TRACER = None
_TRACER_PROVIDER = None
_INITIALIZED = False


def setup_tracing():
    """Initialize Arize tracing. Idempotent and fail-safe.

    Returns the tracer (or None if tracing is disabled / not configured).
    """
    global _TRACER, _TRACER_PROVIDER, _INITIALIZED

    if _INITIALIZED:
        return _TRACER

    _INITIALIZED = True

    if os.getenv("ARIZE_TRACING_ENABLED", "true").lower() == "false":
        print("[observability] ARIZE_TRACING_ENABLED=false — tracing disabled.")
        return None

    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")
    project_name = os.getenv("ARIZE_PROJECT_NAME", "sql-analytics-agent")

    if not space_id or not api_key:
        print(
            "[observability] ARIZE_SPACE_ID / ARIZE_API_KEY not set — "
            "running without tracing."
        )
        return None

    try:
        from arize.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor

        _TRACER_PROVIDER = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
        )

        # LangChain instrumentor traces LangGraph node execution.
        LangChainInstrumentor().instrument(tracer_provider=_TRACER_PROVIDER)
        # OpenAI instrumentor captures model, prompt, tokens, latency.
        OpenAIInstrumentor().instrument(tracer_provider=_TRACER_PROVIDER)

        _TRACER = _TRACER_PROVIDER.get_tracer(__name__)
        print(f"[observability] Arize tracing enabled (project='{project_name}').")
    except Exception as exc:  # noqa: BLE001 — never let tracing break the app
        print(f"[observability] Failed to initialize tracing: {exc}")
        _TRACER = None

    return _TRACER


def get_tracer():
    """Return the active tracer, initializing if needed."""
    if not _INITIALIZED:
        return setup_tracing()
    return _TRACER


def flush() -> None:
    """Flush buffered spans. Call before a short-lived process exits."""
    if _TRACER_PROVIDER is not None:
        try:
            _TRACER_PROVIDER.force_flush()
        except Exception:  # noqa: BLE001
            pass
