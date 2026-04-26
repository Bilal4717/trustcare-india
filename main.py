"""
Serving a Nation — entrypoint.

``load_dotenv()`` runs **before** importing the app so ``GEMINI_API_KEY`` / ``OPENAI_API_KEY``
are visible to ``healthcare_app.config`` when routes start.

**Providers** (``resolve_llm_backend()`` in ``healthcare_app.config``):

1. **Gemini** — set ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``. Chat: ``GEMINI_CHAT_MODEL`` (default
   ``gemini-2.0-flash``). Embeddings: ``GEMINI_EMBED_MODEL`` (default ``gemini-embedding-001``).
   FAISS path: ``facility_faiss_index_gemini`` (or ``VECTORSTORE_PATH_GEMINI``).
2. **OpenAI** — if no Gemini key but ``OPENAI_API_KEY`` is set (or ``LLM_PROVIDER=openai``).
3. **Local** — HuggingFace embeddings + heuristics; optional ``TAVILY_API_KEY`` for web snippets.

Override order: ``LLM_PROVIDER=gemini|openai|local``.

Run: ``python main.py``. Optional: ``MLFLOW_TRACKING_URI``, ``MLFLOW_EXPERIMENT``.
"""

from dotenv import load_dotenv
from flask import Flask, jsonify

load_dotenv()

from healthcare_app.server import create_app

try:
    app = create_app()
except Exception as exc:
    startup_error = str(exc)
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def startup_failed_home():
        return (
            "TrustCare startup failed on this runtime. "
            "Check /health for initialization error details.",
            503,
        )

    @app.route("/health", methods=["GET"])
    def startup_failed_health():
        return jsonify({"status": "error", "startup_error": startup_error}), 503

if __name__ == "__main__":
    app.run(debug=True, port=5001)
