import os
import re


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_gemini_chat_model_id(raw: str) -> str:
    """
    Accept plain ids (gemini-2.0-flash) or mistaken .env values like
    genai.GenerativeModel('gemini-1.5-flash').
    """
    s = (raw or "").strip().strip('"').strip("'")
    if not s:
        return ""
    m = re.search(r"(gemini[-\w.]+)", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    if "/" in s:
        return s.split("/")[-1].strip().lower()
    return s.lower()


def _resolved_gemini_chat_model() -> str:
    raw = (os.getenv("GEMINI_CHAT_MODEL") or os.getenv("GEMINI_MODEL") or "").strip()
    parsed = _parse_gemini_chat_model_id(raw)
    # Some legacy model ids are no longer available for new users.
    # Auto-upgrade to a currently supported default.
    deprecated = {"gemini-2.0-flash", "models/gemini-2.0-flash"}
    if parsed:
        if parsed in deprecated:
            return "gemini-2.5-flash"
        return parsed
    return "gemini-2.5-flash"


DATASET_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(BASE_DIR, "VF_Hackathon_Dataset_India_Large.xlsx"),
)
DATA_SOURCE = os.getenv("DATA_SOURCE", "local_excel").strip().lower()
# Databricks-ready adapter input (exported Delta/parquet/csv/json from Databricks jobs)
DATABRICKS_EXPORT_PATH = os.getenv(
    "DATABRICKS_EXPORT_PATH",
    os.path.join(BASE_DIR, "databricks_facility_export.parquet"),
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# Google AI Studio / Gemini keys (either name works)
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY", "").strip()
    or os.getenv("GOOGLE_API_KEY", "").strip()
)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# Legacy flag — prefer resolve_llm_backend() for routing
USE_OPENAI_LLM = bool(OPENAI_API_KEY)

# Separate FAISS dirs — embedding dimensions differ per provider
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", os.path.join(BASE_DIR, "facility_faiss_index"))
VECTORSTORE_PATH_HF = os.getenv(
    "VECTORSTORE_PATH_HF",
    os.path.join(BASE_DIR, "facility_faiss_index_hf"),
)
VECTORSTORE_PATH_GEMINI = os.getenv(
    "VECTORSTORE_PATH_GEMINI",
    os.path.join(BASE_DIR, "facility_faiss_index_gemini"),
)

# OpenAI
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")

# Gemini chat model (default 2.0 Flash). Use GEMINI_CHAT_MODEL=... or GEMINI_MODEL=... (plain id only).
GEMINI_CHAT_MODEL = _resolved_gemini_chat_model()
# Gemini API embedding id (new SDK); see https://ai.google.dev/gemini-api/docs/embeddings
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# Local fallback (no cloud LLM)
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_BATCH_SIZE = int(os.getenv("CHUNK_BATCH_SIZE", "500"))


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# FAISS / Mosaic: how many chunks to pass into the LLM (lower = faster retrieval + shorter prompts).
try:
    RETRIEVAL_TOP_K = max(1, min(50, int(os.getenv("RETRIEVAL_TOP_K", "10"))))
except ValueError:
    RETRIEVAL_TOP_K = 10

# If true, skip correction_agent → second validator LLM round-trip when the first validation fails.
# Tradeoff: faster responses; less chance to "repair" wording the validator rejected.
CHAT_SINGLE_PASS_VALIDATION = _env_bool("CHAT_SINGLE_PASS_VALIDATION", "false")

# Max characters of `answer` sent to the LLM validator (0 = no limit). Shorter = faster/cheaper.
try:
    VALIDATOR_ANSWER_MAX_CHARS = max(0, int(os.getenv("VALIDATOR_ANSWER_MAX_CHARS", "6000")))
except ValueError:
    VALIDATOR_ANSWER_MAX_CHARS = 6000

MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() in {"1", "true", "yes"}
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "serving-a-nation")

HIGH_ACUITY_SPECIALTIES = [
    "oncology",
    "dialysis",
    "emergencyMedicine",
    "cardiology",
    "neonatalMedicine",
    "traumaSurgery",
    "intensiveCare",
    "neurology",
]

TRUST_RULES = {
    "Advanced Surgery": ["anesthesiologist", "operation theatre", "OT"],
    "ICU": ["ventilator", "intensivist", "critical care"],
    "Emergency Trauma": ["24/7", "emergency", "trauma surgeon"],
    "Dialysis": ["nephrologist", "dialysis machine"],
    "Oncology": ["oncologist", "chemotherapy", "radiation"],
    "Neonatal Care": ["neonatologist", "NICU", "incubator"],
}


def resolve_llm_backend() -> str:
    """
    Which remote/local stack to use. Call after load_dotenv().

    Priority (unless LLM_PROVIDER forces):
      1. gemini — if GEMINI_API_KEY or GOOGLE_API_KEY is set
      2. openai — if OPENAI_API_KEY is set
      3. local — HuggingFace embeddings + heuristics (optional Tavily)
    """
    override = os.getenv("LLM_PROVIDER", "").strip().lower()
    if override == "openai" and OPENAI_API_KEY:
        return "openai"
    if override == "gemini" and GEMINI_API_KEY:
        return "gemini"
    if override == "local":
        return "local"
    if GEMINI_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return "local"
