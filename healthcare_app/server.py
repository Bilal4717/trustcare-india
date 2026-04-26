import os
import io
import csv

from flask import Flask, Response, jsonify, render_template, request
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
from healthcare_app.data import get_vectorstore, load_dataset
from healthcare_app.graph import build_graph
from healthcare_app.observability import configure_mlflow, traced_step


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
        "planner_summary": None,
    }


def create_app() -> Flask:
    template_dir = os.path.join(BASE_DIR, "templates")
    app = Flask(__name__, template_folder=template_dir)

    configure_mlflow()
    df = load_dataset()

    tavily_client = None
    if TAVILY_API_KEY:
        try:
            from tavily import TavilyClient

            tavily_client = TavilyClient(TAVILY_API_KEY)
        except Exception:
            tavily_client = None

    backend = resolve_llm_backend()
    llm = None
    vs_path = VECTORSTORE_PATH_HF
    embed_label = f"local:{LOCAL_EMBED_MODEL}"

    if backend == "gemini":
        os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        embedding = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBED_MODEL,
            google_api_key=GEMINI_API_KEY,
        )
        vs_path = VECTORSTORE_PATH_GEMINI
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            temperature=0,
            google_api_key=GEMINI_API_KEY,
        )
        embed_label = f"gemini:{GEMINI_EMBED_MODEL}"
    elif backend == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        embedding = OpenAIEmbeddings(model=EMBED_MODEL)
        vs_path = VECTORSTORE_PATH
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        embed_label = f"openai:{EMBED_MODEL}"
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embedding = HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)

    vectorstore = get_vectorstore(df, embedding, vectorstore_path=vs_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    runtime = AgentRuntime(llm, retriever, df, tavily_client=tavily_client)
    healthcare_bot = build_graph(runtime)

    @app.route("/")
    def index():
        return render_template("home.html")

    @app.route("/onboarding")
    def onboarding():
        return render_template("onboarding.html")

    @app.route("/app")
    def app_workspace():
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "llm_backend": backend,
                "tavily_configured": bool(TAVILY_API_KEY),
                "embeddings": embed_label,
                "chat_model": GEMINI_CHAT_MODEL if backend == "gemini" else (CHAT_MODEL if backend == "openai" else None),
                "vector_index": vs_path,
            }
        )

    @app.route("/chat", methods=["POST"])
    def chat_endpoint():
        try:
            data = request.get_json() or {}
            user_message = data.get("message", "").strip()
            if not user_message:
                return jsonify({"error": "No message provided"}), 400

            with traced_step("endpoint_chat", {"message_chars": str(len(user_message))}):
                result = healthcare_bot.invoke(_initial_state(user_message))

            return jsonify(
                {
                    "response": result["messages"][-1].content,
                    "trust_score": result.get("trust_score"),
                    "trust_flags": result.get("trust_flags"),
                    "trust_breakdown": result.get("trust_breakdown"),
                    "desert_regions": result.get("desert_regions"),
                    "planner_summary": result.get("planner_summary"),
                    "confidence_score": result.get("confidence_score"),
                    "confidence_band": result.get("confidence_band"),
                    "reason_codes": result.get("reason_codes"),
                    "chain_of_thought": result.get("chain_of_thought", []),
                    "source_citations": result.get("source_citations"),
                }
            )
        except Exception as e:
            return jsonify({"response": f"Internal error — {e}"}), 500

    @app.route("/audit", methods=["POST"])
    def audit_endpoint():
        try:
            data = request.get_json() or {}
            facility_name = str(data.get("facility_name", "")).strip()
            if not facility_name:
                return jsonify({"error": "facility_name is required"}), 400
            out = runtime.audit_agent(_initial_state(f'Audit facility "{facility_name}"'))
            return jsonify(
                {
                    "audit_result": out.get("audit_result"),
                    "confidence_score": out.get("confidence_score"),
                    "confidence_band": out.get("confidence_band"),
                    "reason_codes": out.get("reason_codes"),
                    "chain_of_thought": out.get("chain_of_thought", []),
                    "response": out["messages"][-1].content,
                    "source_citations": out.get("source_citations"),
                }
            )
        except Exception as e:
            return jsonify({"error": f"Audit failed: {e}"}), 500

    @app.route("/desert", methods=["GET"])
    def desert_endpoint():
        try:
            out = runtime.desert_finder(_initial_state("run full desert scan"))
            regions = out.get("desert_regions") or []
            return jsonify(
                {
                    "count": len(regions),
                    "desert_regions": regions,
                    "planner_summary": out.get("planner_summary"),
                    "confidence_score": out.get("confidence_score"),
                    "confidence_band": out.get("confidence_band"),
                    "reason_codes": out.get("reason_codes"),
                    "chain_of_thought": out.get("chain_of_thought", []),
                }
            )
        except Exception as e:
            return jsonify({"error": f"Desert analysis failed: {e}"}), 500

    @app.route("/desert/export.csv", methods=["GET"])
    def desert_export_csv():
        try:
            out = runtime.desert_finder(_initial_state("desert planner export"))
            rows = out.get("planner_summary") or []
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "pin",
                    "city",
                    "state",
                    "priority",
                    "severity_index",
                    "missing_specialties",
                    "recommended_action",
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

    @app.route("/map", methods=["GET"])
    def map_view():
        try:
            out = runtime.desert_finder(_initial_state("map desert analysis"))
            regions = out.get("desert_regions") or []
            features = []
            for r in regions:
                if r.get("lat") is None or r.get("lon") is None:
                    continue
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
                        },
                    }
                )
            return jsonify({"type": "FeatureCollection", "features": features})
        except Exception as e:
            return jsonify({"error": f"Map generation failed: {e}"}), 500

    return app
