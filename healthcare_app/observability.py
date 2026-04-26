from contextlib import contextmanager
from typing import Dict, Optional

from healthcare_app.config import MLFLOW_ENABLED, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, resolve_llm_backend

try:
    import mlflow
except Exception:
    mlflow = None


def configure_mlflow() -> bool:
    """
    Configure MLflow experiment and enable GenAI tracing when MLflow 3+ is available.
    Falls back cleanly if mlflow is missing or tracing APIs are unavailable.
    """
    if not MLFLOW_ENABLED or mlflow is None:
        return False

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    tracing = getattr(mlflow, "tracing", None)
    if tracing is not None:
        enable = getattr(tracing, "enable", None)
        if callable(enable):
            try:
                enable()
            except Exception:
                pass

    if resolve_llm_backend() == "openai":
        openai_mod = getattr(mlflow, "openai", None)
        autolog = getattr(openai_mod, "autolog", None) if openai_mod else None
        if callable(autolog):
            try:
                autolog()
            except Exception:
                pass

    return True


@contextmanager
def traced_step(name: str, attrs: Optional[Dict[str, str]] = None):
    """
    Step-level observability: prefers MLflow 3 tracing spans (start_span),
    then root span without context, then nested runs, then no-op.
    """
    attrs = attrs or {}
    if not MLFLOW_ENABLED or mlflow is None:
        yield
        return

    start_span = getattr(mlflow, "start_span", None)
    if callable(start_span):
        span_attempts = []
        if attrs:
            span_attempts.append({"name": name, "attributes": attrs})
        span_attempts.append({"name": name})
        for kwargs in span_attempts:
            try:
                with start_span(**kwargs):
                    yield
                return
            except Exception:
                continue

    nctx = getattr(mlflow, "start_span_no_context", None)
    if callable(nctx):
        try:
            span = nctx(name=name, attributes=attrs or None)
            try:
                yield
            finally:
                end = getattr(span, "end", None)
                if callable(end):
                    try:
                        end()
                    except Exception:
                        pass
            return
        except Exception:
            pass

    try:
        with mlflow.start_run(run_name=name, nested=True):
            for k, v in attrs.items():
                mlflow.log_param(k, str(v))
            yield
    except Exception:
        yield
