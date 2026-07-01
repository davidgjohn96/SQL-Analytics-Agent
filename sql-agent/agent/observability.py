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
_STATUS = "not initialized"


def setup_tracing():
    """Initialize Arize tracing. Idempotent and fail-safe.

    Returns the tracer (or None if tracing is disabled / not configured).
    """
    global _TRACER, _TRACER_PROVIDER, _INITIALIZED, _STATUS

    if _INITIALIZED:
        return _TRACER

    _INITIALIZED = True

    if os.getenv("ARIZE_TRACING_ENABLED", "true").lower() == "false":
        _STATUS = "disabled (ARIZE_TRACING_ENABLED=false)"
        print(f"[observability] {_STATUS}.")
        return None

    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")
    project_name = os.getenv("ARIZE_PROJECT_NAME", "sql-analytics-agent")

    if not space_id or not api_key:
        missing = ", ".join(
            n for n, v in (("ARIZE_SPACE_ID", space_id), ("ARIZE_API_KEY", api_key)) if not v
        )
        _STATUS = f"off — missing secret(s): {missing}"
        print(f"[observability] {_STATUS} — running without tracing.")
        return None

    # ARIZE_DEBUG=true surfaces span/export activity in stdout (Cloud logs):
    # every span is printed to the console, and OpenTelemetry's own export
    # errors (e.g. rejected credentials) are logged at DEBUG.
    debug = os.getenv("ARIZE_DEBUG", "false").lower() == "true"
    if debug:
        import logging

        logging.basicConfig()
        logging.getLogger("opentelemetry").setLevel(logging.DEBUG)

    try:
        from arize.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor

        _TRACER_PROVIDER = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
            log_to_console=debug,
        )

        # LangChain instrumentor traces LangGraph node execution.
        LangChainInstrumentor().instrument(tracer_provider=_TRACER_PROVIDER)
        # OpenAI instrumentor captures model, prompt, tokens, latency.
        OpenAIInstrumentor().instrument(tracer_provider=_TRACER_PROVIDER)

        _TRACER = _TRACER_PROVIDER.get_tracer(__name__)
        _STATUS = f"enabled → project '{project_name}'"
        print(f"[observability] Arize tracing {_STATUS}"
              f"{' [DEBUG]' if debug else ''}.")
    except Exception as exc:  # noqa: BLE001 — never let tracing break the app
        _STATUS = f"error: {exc}"
        print(f"[observability] Failed to initialize tracing: {exc}")
        _TRACER = None

    return _TRACER


def tracing_status() -> str:
    """Human-readable state of tracing (reflects the actual tracer, not just env)."""
    if not _INITIALIZED:
        setup_tracing()
    return _STATUS


def is_tracing_active() -> bool:
    """True only if the tracer/provider actually initialized and will export."""
    return _TRACER is not None


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
