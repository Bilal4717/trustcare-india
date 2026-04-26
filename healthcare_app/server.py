import csv
import difflib
import io
import os
import re

from flask import Flask, Response, jsonify, render_template, request
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from healthcare_app.agents import AgentRuntime
from healthcare_app.config import (
    BASE_DIR,
    CHAT_MODEL,
    EMBED_MODEL,
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBED_MODEL,
    LOCAL_EMBED_MODEL,
    TAVILY_API_KEY,
    VECTORSTORE_PATH,
    VECTORSTORE_PATH_GEMINI,
    VECTORSTORE_PATH_HF,
    resolve_llm_backend,
)
from healthcare_app.data import _build_metadata, build_facility_text, get_vectorstore, load_dataset
from healthcare_app.databricks_client import (
    configure_databricks_mlflow,
    get_databricks_embeddings,
    get_databricks_llm,
    get_databricks_status,
    get_mosaic_vector_search_retriever,
    is_databricks_configured,
    load_from_unity_catalog,
)
from healthcare_app.graph import build_graph
from healthcare_app.observability import configure_mlflow, traced_step
from healthcare_app.statistics import (
    dataset_quality_report,
    desert_severity_interval,
    score_confidence_interval,
)
from healthcare_app.confidence_metrics import confidence_metrics_report


class _KeywordFallbackRetriever:
    """
    Lightweight zero-embedding retriever for serverless quota fallback.
    """

    def __init__(self, docs, k: int = 10):
        self.docs = docs or []
        self.k = max(1, int(k))

    @staticmethod
    def _tokens(text: str):
        return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) >= 3]

    def invoke(self, query: str):
        q_tokens = self._tokens(query)
        if not q_tokens:
            return self.docs[: self.k]
        scored = []
        for i, doc in enumerate(self.docs):
            blob = f"{doc.page_content} {doc.metadata}".lower()
            hits = sum(1 for t in q_tokens if t in blob)
            if hits == 0:
                continue
            # Stable tie-breaker keeps deterministic order.
            scored.append((hits, -i, doc))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [d for _, _, d in scored[: self.k]]


def _build_keyword_retriever(df, k: int = 10):
    docs = []
    for idx, row in df.iterrows():
        text = build_facility_text(row)
        if len(text.strip()) < 20:
            continue
        docs.append(Document(page_content=text, metadata=_build_metadata(idx, row)))
    return _KeywordFallbackRetriever(docs, k=k)


try:
    import mlflow
except Exception:
    mlflow = None


def _active_trace_run_id():
    if mlflow is None:
        return None
    try:
        run = mlflow.active_run()
        if run and getattr(run, "info", None):
            return getattr(run.info, "run_id", None)
    except Exception:
        return None
    return None


def _ngo_action_for_missing(missing_specialties):
    miss = [str(m).lower() for m in (missing_specialties or [])]
    actions = []
    if any("oncology" in m for m in miss):
        actions.append("Partner oncology referral buses + tele-oncology screening camps")
    if any("dialysis" in m for m in miss):
        actions.append("Deploy dialysis shuttle/referral network + nephrology outreach days")
    if any("emergency" in m or "trauma" in m for m in miss):
        actions.append("Stand up 24x7 emergency transfer protocol with ambulance NGO partners")
    if any("neonatal" in m or "nicu" in m for m in miss):
        actions.append("Maternal-neonatal stabilization training and NICU referral hotline")
    if not actions:
        actions.append("General specialist referral camps and mobile diagnostics")
    return actions[:2]


def _norm_state_name(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("-", " ").replace("_", " ").split())


def _state_suggestions(all_states, query: str, limit: int = 5):
    if not query:
        return []
    norm_query = _norm_state_name(query)
    norm_map = {}
    for st in all_states or []:
        key = _norm_state_name(str(st))
        if key and key not in norm_map:
            norm_map[key] = str(st)
    matches = difflib.get_close_matches(norm_query, list(norm_map.keys()), n=limit, cutoff=0.45)
    return [norm_map[m] for m in matches]


def _initial_state(user_message: str):
    return {
        "messages": [HumanMessage(content=user_message)],
        "intent": None,
        "retrieved_docs": None,
        "answer": None,
        "audit_result": None,
        "trust_score": None,
        "trust_flags": None,
        "desert_regions": None,
        "validated": None,
        "correction_notes": None,
        "chain_of_thought": [],
        "source_citations": None,
        "confidence_score": None,
        "confidence_band": None,
        "reason_codes": None,
        "trust_breakdown": None,
        "trust_evidence_map": None,
        "planner_summary": None,
        "validation_attempts": 0,
        "correction_applied": False,
    }


def create_app() -> Flask:
    template_dir = os.path.join(BASE_DIR, "templates")
    app = Flask(__name__, template_folder=template_dir)

    @app.after_request
    def add_dev_cors_headers(resp):
        # Allow static preview on :5500 to call Flask API on :5001.
        origin = request.headers.get("Origin", "")
        if origin in {"http://127.0.0.1:5500", "http://localhost:5500"}:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    @app.before_request
    def handle_cors_preflight():
        if request.method == "OPTIONS":
            return ("", 204)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    # Prefer Databricks tracking server if configured, else local MLflow
    if is_databricks_configured():
        configure_databricks_mlflow()
    else:
        configure_mlflow()

    # ── Dataset load: Unity Catalog → local Excel fallback ───────────────────
    df = None
    _data_source = "local_excel"
    if is_databricks_configured():
        df = load_from_unity_catalog()
        if df is not None:
            _data_source = "unity_catalog"
    if df is None:
        df = load_dataset()

    # ── Tavily ────────────────────────────────────────────────────────────────
    tavily_client = None
    if TAVILY_API_KEY:
        try:
            from tavily import TavilyClient
            tavily_client = TavilyClient(TAVILY_API_KEY)
        except Exception:
            pass

    # ── LLM + embeddings: Databricks → Gemini → OpenAI → local ──────────────
    llm = None
    embedding = None
    retriever = None
    startup_warning = None
    vs_path = VECTORSTORE_PATH_HF
    embed_label = f"local:{LOCAL_EMBED_MODEL}"
    _backend = "local"

    # 1. Try Databricks Foundation Model API
    if is_databricks_configured():
        db_llm = get_databricks_llm()
        db_emb = get_databricks_embeddings()
        if db_llm and db_emb:
            llm = db_llm
            embedding = db_emb
            _backend = "databricks"
            embed_label = "databricks:bge-large-en"

    # 2. Gemini fallback
    if embedding is None:
        backend = resolve_llm_backend()
        if backend == "gemini":
            os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                GoogleGenerativeAIEmbeddings,
            )
            embedding = GoogleGenerativeAIEmbeddings(
                model=GEMINI_EMBED_MODEL, google_api_key=GEMINI_API_KEY
            )
            vs_path = VECTORSTORE_PATH_GEMINI
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_CHAT_MODEL, temperature=0, google_api_key=GEMINI_API_KEY
            )
            embed_label = f"gemini:{GEMINI_EMBED_MODEL}"
            _backend = "gemini"
        elif backend == "openai":
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            embedding = OpenAIEmbeddings(model=EMBED_MODEL)
            vs_path = VECTORSTORE_PATH
            llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
            embed_label = f"openai:{EMBED_MODEL}"
            _backend = "openai"
        else:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings

                embedding = HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)
                _backend = "local"
            except Exception as exc:
                raise RuntimeError(
                    "Local embedding backend unavailable. Set GEMINI_API_KEY or OPENAI_API_KEY "
                    "for cloud mode, or install local extras with: pip install -r requirements.local.txt"
                ) from exc

    # ── Vector store: Mosaic AI VS → FAISS ───────────────────────────────────
    if is_databricks_configured():
        retriever = get_mosaic_vector_search_retriever(embedding, k=10)

    if retriever is None:
        try:
            vectorstore = get_vectorstore(df, embedding, vectorstore_path=vs_path)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        except Exception as exc:
            err = str(exc)
            err_l = err.lower()
            recoverable_startup = (
                "resource_exhausted" in err_l
                or "quota" in err_l
                or "429" in err_l
                or "rate limit" in err_l
                or "faiss" in err_l
                or "could not import faiss" in err_l
                or "faiss backend unavailable" in err_l
            )
            if not recoverable_startup:
                raise
            retriever = _build_keyword_retriever(df, k=10)
            if "faiss" in err_l:
                startup_warning = (
                    "FAISS unavailable in serverless runtime. "
                    "Using keyword fallback retriever (reduced semantic quality)."
                )
            else:
                startup_warning = (
                    "Embedding quota/rate limit hit during startup. "
                    "Using keyword fallback retriever (reduced semantic quality)."
                )
            vs_path = "keyword_fallback_retriever"

    runtime = AgentRuntime(llm, retriever, df, tavily_client=tavily_client)
    healthcare_bot = build_graph(runtime)

    # ── Precompute dataset quality once at startup ────────────────────────────
    _quality_report = dataset_quality_report(df)

    # ─────────────────────────────────────────────────────────────────────────
    # Routes
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("home.html")

    @app.route("/onboarding")
    @app.route("/onboarding.html")
    @app.route("/guide")
    def onboarding():
        return render_template("onboarding.html")

    @app.route("/app")
    @app.route("/index.html")
    def app_workspace():
        return render_template("index.html")

    # ── /health ───────────────────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health():
        chat_model = None
        if _backend == "databricks":
            from healthcare_app.databricks_client import DATABRICKS_FM_ENDPOINT
            chat_model = DATABRICKS_FM_ENDPOINT
        elif _backend == "gemini":
            chat_model = GEMINI_CHAT_MODEL
        elif _backend == "openai":
            chat_model = CHAT_MODEL

        return jsonify(
            {
                "status": "ok",
                "llm_backend": _backend,
                "tavily_configured": bool(TAVILY_API_KEY),
                "embeddings": embed_label,
                "chat_model": chat_model,
                "data_source": _data_source,
                "vector_index": vs_path if retriever else "mosaic_ai_vector_search",
                "databricks": get_databricks_status(),
                "dataset": {
                    "total_facilities": _quality_report.get("total_facilities", 0),
                    "states_covered": _quality_report.get("states_covered", 0),
                    "overall_completeness": _quality_report.get("overall_completeness", 0),
                },
                "startup_warning": startup_warning,
            }
        )

    # ── /stats ────────────────────────────────────────────────────────────────
    @app.route("/stats", methods=["GET"])
    def stats():
        """Dataset quality report with field fill rates and coverage metrics."""
        return jsonify(_quality_report)

    # ── /metrics/confidence ───────────────────────────────────────────────────
    @app.route("/metrics/confidence", methods=["GET"])
    def confidence_metrics():
        """
        Confidence diagnostics:
        - proxy empirical coverage vs target
        - interval width stats
        - uncertainty decomposition
        """
        try:
            report = confidence_metrics_report(df, _quality_report)
            return jsonify(report)
        except Exception as e:
            return jsonify({"status": "error", "error": f"Confidence metrics failed: {e}"}), 500

    # ── /facilities/geojson ───────────────────────────────────────────────────
    @app.route("/facilities/geojson", methods=["GET"])
    def facilities_geojson():
        """
        GeoJSON FeatureCollection of all facilities that have lat/lon.
        Used by the Leaflet map to show facility density overlays.
        Query params:
          state  - filter by state name (partial, case-insensitive)
          limit  - max features (default 2000)
        """
        import pandas as pd

        state_filter = (request.args.get("state") or "").strip().lower()
        try:
            limit = min(int(request.args.get("limit", 2000)), 5000)
        except ValueError:
            limit = 2000

        frame = df.copy() if df is not None else None
        if frame is None or frame.empty:
            return jsonify({"type": "FeatureCollection", "features": []})

        for col in ["latitude", "longitude"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        frame = frame.dropna(subset=["latitude", "longitude"])
        if state_filter:
            col = "address_stateOrRegion"
            if col in frame.columns:
                frame = frame[frame[col].str.lower().str.contains(state_filter, na=False)]

        frame = frame.head(limit)
        features = []
        for _, row in frame.iterrows():
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["longitude"]), float(row["latitude"])],
                    },
                    "properties": {
                        "name": str(row.get("name") or ""),
                        "type": str(row.get("facilityTypeId") or ""),
                        "city": str(row.get("address_city") or ""),
                        "state": str(row.get("address_stateOrRegion") or ""),
                        "pin": str(row.get("pin") or ""),
                        "completeness": float(row.get("_completeness") or 0),
                    },
                }
            )
        return jsonify({"type": "FeatureCollection", "features": features})

    # ── /chat ─────────────────────────────────────────────────────────────────
    @app.route("/chat", methods=["POST"])
    def chat_endpoint():
        try:
            data = request.get_json() or {}
            user_message = data.get("message", "").strip()
            if not user_message:
                return jsonify({"error": "No message provided"}), 400

            with traced_step("endpoint_chat", {"message_chars": str(len(user_message))}):
                result = healthcare_bot.invoke(_initial_state(user_message))
                trace_run_id = _active_trace_run_id()

            # Attach confidence intervals to the response
            cs = result.get("confidence_score") or 0.0
            n_docs = len(result.get("retrieved_docs") or [])
            comp = (
                result["retrieved_docs"][0].get("completeness", 0.5)
                if result.get("retrieved_docs")
                else 0.5
            )
            ci = score_confidence_interval(cs, comp, n_docs)

            return jsonify(
                {
                    "response": result["messages"][-1].content,
                    "trust_score": result.get("trust_score"),
                    "trust_flags": result.get("trust_flags"),
                    "trust_breakdown": result.get("trust_breakdown"),
                    "trust_evidence_map": result.get("trust_evidence_map"),
                    "desert_regions": result.get("desert_regions"),
                    "planner_summary": result.get("planner_summary"),
                    "confidence_score": result.get("confidence_score"),
                    "confidence_band": result.get("confidence_band"),
                    "confidence_interval": ci,
                    "reason_codes": result.get("reason_codes"),
                    "validation_attempts": result.get("validation_attempts"),
                    "correction_applied": result.get("correction_applied"),
                    "chain_of_thought": result.get("chain_of_thought", []),
                    "source_citations": result.get("source_citations"),
                    "trace_run_id": trace_run_id,
                }
            )
        except Exception as e:
            return jsonify({"response": f"Internal error — {e}"}), 500

    # ── /audit ────────────────────────────────────────────────────────────────
    @app.route("/audit", methods=["POST"])
    def audit_endpoint():
        try:
            data = request.get_json() or {}
            facility_name = str(data.get("facility_name", "")).strip()
            if not facility_name:
                return jsonify({"error": "facility_name is required"}), 400
            out = runtime.audit_agent(_initial_state(f'Audit facility "{facility_name}"'))
            trace_run_id = _active_trace_run_id()

            cs = out.get("confidence_score") or 0.0
            n_docs = len(out.get("retrieved_docs") or [])
            comp = (
                out["retrieved_docs"][0].get("completeness", 0.5)
                if out.get("retrieved_docs")
                else 0.5
            )
            ci = score_confidence_interval(cs, comp, n_docs)

            return jsonify(
                {
                    "audit_result": out.get("audit_result"),
                    "confidence_score": out.get("confidence_score"),
                    "confidence_band": out.get("confidence_band"),
                    "confidence_interval": ci,
                    "reason_codes": out.get("reason_codes"),
                    "validation_attempts": out.get("validation_attempts"),
                    "correction_applied": out.get("correction_applied"),
                    "chain_of_thought": out.get("chain_of_thought", []),
                    "response": out["messages"][-1].content,
                    "source_citations": out.get("source_citations"),
                    "trace_run_id": trace_run_id,
                }
            )
        except Exception as e:
            return jsonify({"error": f"Audit failed: {e}"}), 500

    # ── /desert ───────────────────────────────────────────────────────────────
    @app.route("/desert", methods=["GET"])
    def desert_endpoint():
        try:
            from healthcare_app.config import HIGH_ACUITY_SPECIALTIES

            out = runtime.desert_finder(_initial_state("run full desert scan"))
            regions = out.get("desert_regions") or []

            # Attach per-region severity prediction intervals
            enriched = []
            for r in regions:
                si = desert_severity_interval(
                    missing_count=len(r.get("missing_specialties") or []),
                    total_specialties=len(HIGH_ACUITY_SPECIALTIES),
                    facility_count=r.get("facility_count", 1),
                )
                enriched.append({**r, "severity_interval": si})

            return jsonify(
                {
                    "count": len(enriched),
                    "desert_regions": enriched,
                    "planner_summary": out.get("planner_summary"),
                    "confidence_score": out.get("confidence_score"),
                    "confidence_band": out.get("confidence_band"),
                    "reason_codes": out.get("reason_codes"),
                    "chain_of_thought": out.get("chain_of_thought", []),
                }
            )
        except Exception as e:
            return jsonify({"error": f"Desert analysis failed: {e}"}), 500

    # ── /desert/export.csv ────────────────────────────────────────────────────
    @app.route("/desert/export.csv", methods=["GET"])
    def desert_export_csv():
        try:
            out = runtime.desert_finder(_initial_state("desert planner export"))
            rows = out.get("planner_summary") or []
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "pin", "city", "state", "priority",
                    "severity_index", "missing_specialties", "recommended_action",
                ],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {
                        "pin": r.get("pin"),
                        "city": r.get("city"),
                        "state": r.get("state"),
                        "priority": r.get("priority"),
                        "severity_index": r.get("severity_index"),
                        "missing_specialties": ", ".join(r.get("missing_specialties") or []),
                        "recommended_action": r.get("recommended_action"),
                    }
                )
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=desert_planner_summary.csv"},
            )
        except Exception as e:
            return jsonify({"error": f"Desert CSV export failed: {e}"}), 500

    # ── /map ──────────────────────────────────────────────────────────────────
    @app.route("/map", methods=["GET"])
    def map_view():
        """
        GeoJSON FeatureCollection of healthcare desert regions.
        Each feature includes severity_interval for confidence visualization.
        """
        try:
            from healthcare_app.config import HIGH_ACUITY_SPECIALTIES

            out = runtime.desert_finder(_initial_state("map desert analysis"))
            regions = out.get("desert_regions") or []
            state_filter = (request.args.get("state") or "").strip().lower()
            priority_filter = (request.args.get("priority") or "").strip().lower()
            try:
                min_severity = float(request.args.get("min_severity", "0") or 0.0)
            except Exception:
                min_severity = 0.0
            try:
                top_n = max(0, int(request.args.get("top_n", "0") or 0))
            except Exception:
                top_n = 0

            all_states = sorted({str(r.get("state") or "").strip() for r in regions if str(r.get("state") or "").strip()})
            if state_filter:
                sf = _norm_state_name(state_filter)
                regions = [r for r in regions if sf in _norm_state_name(str(r.get("state", "")))]
            if priority_filter:
                regions = [r for r in regions if str(r.get("intervention_priority", "")).lower() == priority_filter]
            if min_severity > 0:
                regions = [r for r in regions if float(r.get("severity_index") or 0.0) >= min_severity]

            regions = sorted(regions, key=lambda r: float(r.get("severity_index") or 0.0), reverse=True)
            if top_n > 0:
                regions = regions[:top_n]

            features = []
            for r in regions:
                if r.get("lat") is None or r.get("lon") is None:
                    continue
                si = desert_severity_interval(
                    missing_count=len(r.get("missing_specialties") or []),
                    total_specialties=len(HIGH_ACUITY_SPECIALTIES),
                    facility_count=r.get("facility_count", 1),
                )
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                        "properties": {
                            "pin": r.get("pin"),
                            "city": r.get("city"),
                            "state": r.get("state"),
                            "missing_specialties": r.get("missing_specialties", []),
                            "facility_count": r.get("facility_count", 0),
                            "severity_index": r.get("severity_index"),
                            "intervention_priority": r.get("intervention_priority"),
                            "severity_interval": si,
                        },
                    }
                )
            top_risk_pins = [
                {
                    "pin": r.get("pin"),
                    "city": r.get("city"),
                    "state": r.get("state"),
                    "priority": r.get("intervention_priority"),
                    "severity_index": r.get("severity_index"),
                    "missing_specialties_count": len(r.get("missing_specialties") or []),
                    "ngo_actions": _ngo_action_for_missing(r.get("missing_specialties") or []),
                }
                for r in regions[:20]
            ]
            return jsonify(
                {
                    "type": "FeatureCollection",
                    "features": features,
                    "top_risk_pins": top_risk_pins,
                    "state_suggestions": _state_suggestions(all_states, state_filter),
                    "filters_applied": {
                        "state": state_filter or None,
                        "priority": priority_filter or None,
                        "min_severity": min_severity,
                        "top_n": top_n or None,
                    },
                }
            )
        except Exception as e:
            return jsonify({"error": f"Map generation failed: {e}"}), 500

    return app
