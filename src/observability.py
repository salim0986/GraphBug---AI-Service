"""
M10 — Production observability: LangFuse tracing + OpenTelemetry.

LangFuse (all optional – tracing is disabled when env vars are absent):
  LANGFUSE_SECRET_KEY  — LangFuse server secret key
  LANGFUSE_PUBLIC_KEY  — LangFuse public key
  LANGFUSE_HOST        — override host (default: https://cloud.langfuse.com)

OpenTelemetry (optional – skipped silently when packages are absent):
  OTLP_ENDPOINT  — OTLP gRPC endpoint (e.g. http://localhost:4317)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from .logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# LangFuse singleton
# ---------------------------------------------------------------------------

_lf_instance: Optional[Any] = None
_lf_init_attempted: bool = False


def get_langfuse() -> Optional[Any]:
    """
    Return the LangFuse client singleton, or None if not configured/available.
    Lazy-initialised on first call; cached thereafter.
    """
    global _lf_instance, _lf_init_attempted
    if _lf_init_attempted:
        return _lf_instance
    _lf_init_attempted = True

    if not os.getenv("LANGFUSE_SECRET_KEY"):
        logger.debug("[M10] LANGFUSE_SECRET_KEY not set – LangFuse tracing disabled")
        return None

    try:
        from langfuse import Langfuse  # type: ignore[import]
        _lf_instance = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("[M10] LangFuse initialised")
    except ImportError:
        logger.info("[M10] langfuse package not installed – tracing disabled")
    except Exception as exc:
        logger.warning(f"[M10] LangFuse init failed: {exc}")

    return _lf_instance


def reset_langfuse() -> None:
    """Reset the LangFuse singleton. Used in tests."""
    global _lf_instance, _lf_init_attempted
    _lf_instance = None
    _lf_init_attempted = False


# ---------------------------------------------------------------------------
# Fire-and-forget LangFuse generation recording
# ---------------------------------------------------------------------------

def observe_llm_call(
    name: str,
    model: str,
    input_text: str,
    output: str,
    tokens_in: int,
    tokens_out: int,
    **meta: Any,
) -> None:
    """
    Record a completed LLM call as a LangFuse generation.
    Safe to call even when LangFuse is not configured – silently no-ops.
    """
    lf = get_langfuse()
    if not lf:
        return
    try:
        trace = lf.trace(name=name, metadata=meta)
        gen = trace.generation(
            name=name,
            model=model,
            input=input_text,
            metadata=meta,
        )
        gen.end(
            output=output[:1000] if output else "",
            usage={"input": tokens_in, "output": tokens_out},
        )
    except Exception as exc:
        logger.debug(f"[M10] LangFuse record failed: {exc}")


# ---------------------------------------------------------------------------
# LangFuse generation span context manager (for external callers)
# ---------------------------------------------------------------------------

@contextmanager
def llm_span(
    name: str,
    model: str,
    input_text: str,
    **meta: Any,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that records a LangFuse generation span.
    The yielded dict has ``output`` and ``usage`` keys that callers must
    populate before the context exits.

    Example::

        with llm_span("review", model="gemini-2.5-flash", input_text=prompt) as span:
            text = call_llm(prompt)
            span["output"] = text
            span["usage"] = {"input": 120, "output": 40}
    """
    result: Dict[str, Any] = {"output": None, "usage": None}
    lf = get_langfuse()
    trace = None
    generation = None

    if lf:
        try:
            trace = lf.trace(name=name, metadata=meta)
            generation = trace.generation(
                name=name,
                model=model,
                input=input_text,
                metadata=meta,
            )
        except Exception as exc:
            logger.debug(f"[M10] llm_span init failed: {exc}")
            generation = None

    try:
        yield result
    finally:
        if generation:
            try:
                out = result.get("output")
                generation.end(
                    output=out[:1000] if isinstance(out, str) else out,
                    usage=result.get("usage"),
                )
            except Exception as exc:
                logger.debug(f"[M10] llm_span end failed: {exc}")


# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------

def setup_otel(app: Any, service_name: str = "graph-bug-ai") -> bool:
    """
    Instrument a FastAPI *app* with OpenTelemetry.

    Returns True when instrumentation was applied, False when packages are
    absent (graceful no-op – never raises).

    The OTLP exporter is added only when ``OTLP_ENDPOINT`` is set.
    """
    try:
        from opentelemetry import trace  # type: ignore[import]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import]
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore[import]

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        otlp_endpoint = os.getenv("OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import]
                    OTLPSpanExporter,
                )
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
                )
                logger.info(f"[M10] OTel OTLP exporter → {otlp_endpoint}")
            except Exception as exc:
                logger.warning(f"[M10] OTel OTLP exporter failed: {exc}")

        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[M10] OpenTelemetry instrumentation applied")
        return True

    except ImportError:
        logger.debug("[M10] opentelemetry packages not installed – OTel skipped")
        return False


def get_tracer(name: str = "graph-bug") -> Any:
    """Return an OTel tracer, or a no-op stand-in when OTel is absent."""
    try:
        from opentelemetry import trace  # type: ignore[import]
        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


class _NoopTracer:
    """Minimal no-op tracer used when opentelemetry is not installed."""

    @contextmanager
    def start_as_current_span(self, name: str, **_kwargs: Any):  # type: ignore[override]
        yield None
