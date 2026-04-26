"""
Databricks integration for TrustCare India.

Supports:
- Unity Catalog Delta table reads via Databricks SQL connector
- Mosaic AI Vector Search (production alternative to FAISS)
- Foundation Model API (Meta-Llama 3, DBRX via Databricks serving)
- MLflow 3 with Databricks as tracking server

Set these env vars to enable:
    DATABRICKS_HOST      = https://your-workspace.azuredatabricks.net
    DATABRICKS_TOKEN     = your-personal-access-token
    DATABRICKS_WAREHOUSE_ID  (SQL warehouse for Unity Catalog reads)
    DATABRICKS_VECTOR_SEARCH_ENDPOINT
    DATABRICKS_VECTOR_SEARCH_INDEX
    DATABRICKS_CATALOG   = hive_metastore  (or your UC catalog)
    DATABRICKS_SCHEMA    = healthcare
    DATABRICKS_TABLE     = facilities
    DATABRICKS_FM_ENDPOINT = databricks-meta-llama-3-3-70b-instruct
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── env config ────────────────────────────────────────────────────────────────
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").strip().rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()
DATABRICKS_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "").strip()
DATABRICKS_VS_ENDPOINT = os.getenv("DATABRICKS_VECTOR_SEARCH_ENDPOINT", "").strip()
DATABRICKS_VS_INDEX = os.getenv("DATABRICKS_VECTOR_SEARCH_INDEX", "").strip()
DATABRICKS_CATALOG = os.getenv("DATABRICKS_CATALOG", "hive_metastore").strip()
DATABRICKS_SCHEMA = os.getenv("DATABRICKS_SCHEMA", "healthcare").strip()
DATABRICKS_TABLE = os.getenv("DATABRICKS_TABLE", "facilities").strip()
# Foundation Model API endpoint (pay-per-token on Free Edition)
DATABRICKS_FM_ENDPOINT = os.getenv(
    "DATABRICKS_FM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
).strip()
DATABRICKS_EMBED_ENDPOINT = os.getenv(
    "DATABRICKS_EMBED_ENDPOINT", "databricks-bge-large-en"
).strip()


# ── connectivity helpers ───────────────────────────────────────────────────────

def is_databricks_configured() -> bool:
    """True when host + token are both set."""
    return bool(DATABRICKS_HOST and DATABRICKS_TOKEN)


def get_databricks_status() -> Dict[str, Any]:
    """Return connection config for /health endpoint."""
    return {
        "configured": is_databricks_configured(),
        "host": DATABRICKS_HOST or None,
        "sql_warehouse": bool(DATABRICKS_WAREHOUSE_ID),
        "vector_search": bool(DATABRICKS_VS_ENDPOINT and DATABRICKS_VS_INDEX),
        "fm_endpoint": DATABRICKS_FM_ENDPOINT if is_databricks_configured() else None,
        "unity_catalog": f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.{DATABRICKS_TABLE}",
    }


# ── LLM / embeddings ──────────────────────────────────────────────────────────

def get_databricks_llm():
    """
    Return a LangChain ChatDatabricks instance pointing at the Databricks
    Foundation Model API. Falls back to None if unconfigured or SDK missing.
    """
    if not is_databricks_configured():
        return None
    try:
        from langchain_community.chat_models.databricks import ChatDatabricks

        return ChatDatabricks(
            endpoint=DATABRICKS_FM_ENDPOINT,
            target_uri=DATABRICKS_HOST,
            temperature=0,
            max_tokens=2048,
        )
    except ImportError:
        logger.warning("langchain_community.chat_models.databricks not available")
        return None
    except Exception as exc:
        logger.warning("ChatDatabricks init failed: %s", exc)
        return None


def get_databricks_embeddings():
    """
    Return Databricks-hosted embedding model (BGE-Large).
    Falls back to None if unconfigured or SDK missing.
    """
    if not is_databricks_configured():
        return None
    try:
        from langchain_community.embeddings.databricks import DatabricksEmbeddings

        return DatabricksEmbeddings(
            endpoint=DATABRICKS_EMBED_ENDPOINT,
            target_uri=DATABRICKS_HOST,
        )
    except ImportError:
        logger.warning("DatabricksEmbeddings not available")
        return None
    except Exception as exc:
        logger.warning("DatabricksEmbeddings init failed: %s", exc)
        return None


# ── Unity Catalog / SQL ───────────────────────────────────────────────────────

def load_from_unity_catalog():
    """
    Read facility data directly from a Unity Catalog Delta table using the
    Databricks SQL connector.  Returns a pandas DataFrame or None.

    Requires: pip install databricks-sql-connector pandas
    """
    if not is_databricks_configured() or not DATABRICKS_WAREHOUSE_ID:
        return None
    try:
        from databricks import sql as dbsql  # type: ignore
        import pandas as pd

        hostname = DATABRICKS_HOST.replace("https://", "").replace("http://", "")
        conn = dbsql.connect(
            server_hostname=hostname,
            http_path=f"/sql/1.0/warehouses/{DATABRICKS_WAREHOUSE_ID}",
            access_token=DATABRICKS_TOKEN,
        )
        fqt = f"`{DATABRICKS_CATALOG}`.`{DATABRICKS_SCHEMA}`.`{DATABRICKS_TABLE}`"
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {fqt} LIMIT 15000")
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
        conn.close()
        df = pd.DataFrame(rows, columns=cols)
        logger.info("Loaded %d rows from Unity Catalog %s", len(df), fqt)
        return df
    except ImportError:
        logger.warning("databricks-sql-connector not installed; skipping Unity Catalog")
        return None
    except Exception as exc:
        logger.warning("Unity Catalog load failed: %s", exc)
        return None


def ingest_to_unity_catalog(df) -> bool:
    """
    Write/overwrite the facility DataFrame to a Unity Catalog Delta table.
    Useful for initial data load from the Excel dataset.
    """
    if not is_databricks_configured() or not DATABRICKS_WAREHOUSE_ID:
        return False
    try:
        import pandas as pd
        from databricks import sql as dbsql  # type: ignore

        hostname = DATABRICKS_HOST.replace("https://", "").replace("http://", "")
        conn = dbsql.connect(
            server_hostname=hostname,
            http_path=f"/sql/1.0/warehouses/{DATABRICKS_WAREHOUSE_ID}",
            access_token=DATABRICKS_TOKEN,
        )
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE SCHEMA IF NOT EXISTS `{DATABRICKS_CATALOG}`.`{DATABRICKS_SCHEMA}`"
            )
            fqt = f"`{DATABRICKS_CATALOG}`.`{DATABRICKS_SCHEMA}`.`{DATABRICKS_TABLE}`"
            cursor.execute(f"DROP TABLE IF EXISTS {fqt}")
            # Build CREATE TABLE from df dtypes
            col_defs = ", ".join(
                f"`{c}` STRING" for c in df.columns
            )
            cursor.execute(f"CREATE TABLE {fqt} ({col_defs}) USING DELTA")
            # Batch insert
            for i in range(0, len(df), 500):
                batch = df.iloc[i : i + 500]
                placeholders = ", ".join(
                    "(" + ", ".join("?" * len(batch.columns)) + ")"
                    for _ in range(len(batch))
                )
                values = [
                    str(v) if v is not None else None
                    for row in batch.itertuples(index=False)
                    for v in row
                ]
                cursor.execute(f"INSERT INTO {fqt} VALUES {placeholders}", values)
        conn.close()
        logger.info("Ingested %d rows to %s", len(df), fqt)
        return True
    except Exception as exc:
        logger.warning("Unity Catalog ingest failed: %s", exc)
        return False


# ── Mosaic AI Vector Search ────────────────────────────────────────────────────

def get_mosaic_vector_search_retriever(embedding, k: int = 10):
    """
    Return a Mosaic AI Vector Search retriever (production replacement for FAISS).
    Requires DATABRICKS_VECTOR_SEARCH_ENDPOINT and DATABRICKS_VECTOR_SEARCH_INDEX.
    Falls back to None if unconfigured.
    """
    if not is_databricks_configured():
        return None
    if not (DATABRICKS_VS_ENDPOINT and DATABRICKS_VS_INDEX):
        return None
    try:
        from langchain_community.vectorstores import DatabricksVectorSearch  # type: ignore

        vstore = DatabricksVectorSearch(
            endpoint_name=DATABRICKS_VS_ENDPOINT,
            index_name=DATABRICKS_VS_INDEX,
            embedding=embedding,
            host=DATABRICKS_HOST,
            api_token=DATABRICKS_TOKEN,
        )
        logger.info("Using Mosaic AI Vector Search: %s", DATABRICKS_VS_INDEX)
        return vstore.as_retriever(search_kwargs={"k": k})
    except ImportError:
        logger.warning("DatabricksVectorSearch not available in langchain_community")
        return None
    except Exception as exc:
        logger.warning("Mosaic AI Vector Search init failed: %s", exc)
        return None


def create_vector_search_index(
    docs: List[Any],
    embedding,
    index_name: Optional[str] = None,
) -> bool:
    """
    Sync documents into a Mosaic AI Vector Search Delta Sync Index.
    This is the Databricks-native replacement for FAISS.save_local().
    """
    target = index_name or DATABRICKS_VS_INDEX
    if not is_databricks_configured() or not (DATABRICKS_VS_ENDPOINT and target):
        return False
    try:
        from langchain_community.vectorstores import DatabricksVectorSearch  # type: ignore

        DatabricksVectorSearch.from_documents(
            documents=docs,
            embedding=embedding,
            endpoint_name=DATABRICKS_VS_ENDPOINT,
            index_name=target,
            host=DATABRICKS_HOST,
            api_token=DATABRICKS_TOKEN,
        )
        logger.info("Synced %d documents to Mosaic AI index %s", len(docs), target)
        return True
    except Exception as exc:
        logger.warning("Vector Search index sync failed: %s", exc)
        return False


# ── MLflow 3 + Databricks tracking ────────────────────────────────────────────

def configure_databricks_mlflow(experiment: str = "serving-a-nation") -> bool:
    """
    Configure MLflow to use Databricks as the tracking server.
    This enables experiment tracking, model registry, and LLM tracing
    in the Databricks workspace UI.

    Databricks Free Edition supports this via the built-in MLflow tracking.
    """
    if not is_databricks_configured():
        return False
    try:
        import mlflow

        # Set Databricks as tracking server
        os.environ.setdefault("DATABRICKS_HOST", DATABRICKS_HOST)
        os.environ.setdefault("DATABRICKS_TOKEN", DATABRICKS_TOKEN)
        mlflow.set_tracking_uri("databricks")

        # Create / set experiment under the user workspace
        mlflow.set_experiment(f"/Users/trustcare/{experiment}")

        # Enable GenAI tracing (MLflow 3+)
        tracing = getattr(mlflow, "tracing", None)
        if tracing and callable(getattr(tracing, "enable", None)):
            tracing.enable()

        logger.info("MLflow tracking set to Databricks workspace")
        return True
    except Exception as exc:
        logger.warning("Databricks MLflow config failed: %s", exc)
        return False


def log_agent_trace(
    run_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Log a single agent inference as an MLflow run with inputs/outputs."""
    try:
        import mlflow

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v)[:250] for k, v in inputs.items()})
            if outputs.get("chain_of_thought"):
                mlflow.log_text(
                    "\n".join(outputs["chain_of_thought"]), "chain_of_thought.txt"
                )
            if metrics:
                mlflow.log_metrics(metrics)
            if outputs.get("source_citations"):
                import json
                mlflow.log_text(
                    json.dumps(outputs["source_citations"], indent=2),
                    "source_citations.json",
                )
    except Exception:
        pass  # tracing is best-effort
