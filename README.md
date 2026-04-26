# Serving a Nation

Agentic Healthcare Intelligence System for Indian facility discovery, capability audit, trust scoring, and healthcare desert detection.

## Why this exists

In India, discovery-to-care is often the bottleneck: families travel long distances only to find that a nearby facility does not actually have the required specialist, equipment, or emergency capability.  
This project turns 10k+ messy facility records into an explainable reasoning layer with:

- capability verification (audit)
- contradiction-aware trust scoring
- specialty desert detection (PIN/city/state-level)
- transparent evidence snippets and reasoning trace

## Hackathon alignment

- **Discovery & Verification (35%)**
  - `audit_agent` extracts and verifies clinical claims against evidence
  - `trust_scorer` flags claim-evidence contradictions and profile gaps
  - `validator_agent` performs a self-check pass before final response

- **IDP Innovation (30%)**
  - unstructured notes are normalized + embedded
  - multi-attribute retrieval + reasoning across specialties, procedures, equipment, staffing, and geography

- **Social Impact & Utility (25%)**
  - `desert_finder` identifies high-acuity specialty gaps
  - `/desert` and `/map` expose actionable region-level output

- **UX & Transparency (10%)**
  - UI shows answer, trust score, flags, chain-of-thought, and row-level evidence snippets

## Architecture

```text
main.py
  -> healthcare_app.server.create_app()
      -> load_dataset() + get_vectorstore()
      -> AgentRuntime + LangGraph pipeline
      -> Flask routes: /chat /audit /desert /map /health
```

Core modules:

- `healthcare_app/config.py` - providers, model ids, paths, trust rules
- `healthcare_app/data.py` - ingestion, normalization, embedding corpus prep, FAISS
- `healthcare_app/agents.py` - intent/query/audit/trust/desert/validator/response agents
- `healthcare_app/graph.py` - LangGraph wiring and routing
- `healthcare_app/server.py` - Flask API and runtime bootstrap
- `healthcare_app/observability.py` - MLflow tracing hooks

## Providers

Backend auto-selection (unless `LLM_PROVIDER` is explicitly set):

1. **Gemini** (recommended) - if `GEMINI_API_KEY` or `GOOGLE_API_KEY` exists
2. **OpenAI** - if `OPENAI_API_KEY` exists
3. **Local fallback** - local embeddings + heuristic reasoning

> Note: chat and embedding indexes are provider-specific.  
> Separate FAISS directories are used to avoid vector dimension mismatch.

## Quick start

### 1) Install

```bash
pip install -r requirements.txt
```

For local-only heavyweight features (FAISS local index, HuggingFace local fallback, MLflow):

```bash
pip install -r requirements.local.txt
```

### 2) Configure environment

Copy `.env.example` to `.env` in project root.

Minimum Gemini setup:

```env
GEMINI_API_KEY=your_key
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001
```

Optional:

- `TAVILY_API_KEY` (supplemental web context in query agent)
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`
- `LLM_PROVIDER=gemini|openai|local`

### 3) Run

```bash
python main.py
```

Open:

- `http://127.0.0.1:5001/` - UI
- `http://127.0.0.1:5001/health` - runtime/provider status

## Deploy on Render (Web Service)

1. **Push** this repo to GitHub (include `VF_Hackathon_Dataset_India_Large.xlsx` if you rely on the default `DATASET_PATH`).

2. In [Render](https://dashboard.render.com): **New** → **Blueprint** (or **Web Service** if you prefer manual setup).

3. **Blueprint path:** `render.yaml` (already in the repo). Approve the service.

4. **Environment variables** (service → *Environment*): add at least **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`**. Mark them **Secret**. Optional: `TAVILY_API_KEY`, `LLM_PROVIDER`, `GEMINI_CHAT_MODEL`, `GEMINI_EMBED_MODEL`, `CHAT_SINGLE_PASS_VALIDATION`, etc. (see `.env.example`).

5. **Build / start** are defined in `render.yaml`: installs `requirements.render.txt` (app deps + `faiss-cpu` + `gunicorn`), starts Gunicorn on **`$PORT`**.

6. **Health check:** `GET /health` — if deploys fail, increase instance type (free tier can run out of RAM while building the FAISS index) or pre-build `facility_faiss_index_*` and ship it (may require `git` tracking or an external object store + code change).

7. **Smoke test:** open the Render URL with paths `/`, `/app`, and `/health`.

*Manual Web Service (no Blueprint):* Runtime **Python 3.12**, build `pip install -r requirements.render.txt`, start `gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 300 main:app`, add the same env vars.

## Deploy on Vultr (VPS) — recommended for this app

Serverless hosts (Vercel) are a poor fit for loading a large Excel file, building FAISS, and long LLM runs. A small **Linux VM** avoids function timeouts and bundle limits.

1. **Create a server:** [Vultr](https://www.vultr.com) → Deploy → **Cloud Compute** → **Ubuntu 22.04 LTS** → pick a region near users → **$6/mo** (1 GB RAM) or **$12/mo** (2 GB) for safer FAISS builds. Add your SSH key.

2. **Firewall (Vultr panel or UFW):** allow **22** (SSH), **80** (HTTP), **443** (HTTPS).

3. **SSH in** and install dependencies:

```bash
sudo apt update && sudo apt install -y python3.12-venv python3-pip git nginx
```

4. **App user & code**

```bash
sudo adduser --disabled-password --gecos "" trustcare
sudo mkdir -p /opt/trustcare && sudo chown trustcare:trustcare /opt/trustcare
sudo -u trustcare -H bash -c 'cd /opt/trustcare && git clone https://github.com/Bilal4717/trustcare-india.git app && cd app && python3.12 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.local.txt'
```

(Change the URL if you fork. Use `requirements.txt` if you skip local ML extras and rely on Gemini only.)

5. **Environment file** (as `trustcare` user): create `/opt/trustcare/app/.env` with `GEMINI_API_KEY=...`, `DATA_SOURCE=local_excel`, etc. — same keys as local. **Chmod 600** the file.

6. **Gunicorn systemd service** — create `/etc/systemd/system/trustcare.service`:

```ini
[Unit]
Description=TrustCare India Flask
After=network.target

[Service]
User=trustcare
Group=trustcare
WorkingDirectory=/opt/trustcare/app
EnvironmentFile=/opt/trustcare/app/.env
ExecStart=/opt/trustcare/app/.venv/bin/gunicorn -w 1 -b 127.0.0.1:5001 --timeout 300 main:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Then: `sudo systemctl daemon-reload && sudo systemctl enable --now trustcare`

7. **Nginx reverse proxy** — site config proxying to `http://127.0.0.1:5001`, then `sudo certbot --nginx` for HTTPS on your domain.

8. **Dataset:** ensure `VF_Hackathon_Dataset_India_Large.xlsx` is in `/opt/trustcare/app` (pulled with git) or set `DATASET_PATH` in `.env` to an absolute path.

**Vercel note:** if `vercel.json` `functions` patterns fail to match, use a glob (`api/**/*.py`) or set **Function max duration** in the Vercel project settings instead. For this workload, a VPS is simpler.

## API endpoints

- `POST /chat`
  - input: `{ "message": "..." }`
  - output includes:
    - `response`
    - `trust_score`
    - `trust_flags`
    - `chain_of_thought`
    - `source_citations`
    - `trust_evidence_map` (flag -> exact supporting sentence / metadata)
    - `validation_attempts`, `correction_applied` (self-correction loop telemetry)
    - `trace_run_id` (MLflow run id when available)
    - `desert_regions` (when relevant)

- `POST /audit`
  - input: `{ "facility_name": "..." }`
  - output: direct audit result for target facility

- `GET /desert`
  - returns ranked desert regions with missing specialties

- `GET /map`
  - returns GeoJSON-like point features for visualization
  - query params: `state`, `priority`, `min_severity`, `top_n`
  - includes `top_risk_pins` for ranked crisis triage

- `GET /health`
  - backend, model, embedding, vector index status

- `GET /metrics/confidence`
  - confidence calibration diagnostics:
    - `empirical_coverage_proxy` vs `target_coverage`
    - interval width statistics
    - uncertainty component breakdown
    - regional proxy summary

## Data

- Dataset file expected at project root:
  - `VF_Hackathon_Dataset_India_Large.xlsx`

The app normalizes null-like values, cleans PIN data, computes completeness hints, and builds vector embeddings for retrieval.

## Observability

`healthcare_app/observability.py` supports MLflow step-level tracing with graceful fallback if MLflow is unavailable.

## Stretch goals implemented

- **Agentic traceability**
  - Row-level citations + reasoning trace in UI.
  - Trust flag evidence mapping (`trust_evidence_map`) with penalty and source metadata.
  - MLflow-compatible tracing hooks and API run id emission (`trace_run_id`).

- **Self-correction loop**
  - Graph-level validator retry path:
    - core agent -> validator -> correction -> validator -> response
  - Exposes `validation_attempts` and `correction_applied` in API/UI.

- **Dynamic crisis mapping**
  - Interactive Leaflet map with facility and desert overlays.
  - Filterable desert view by state/priority/severity, top-risk PIN ranking, and CSV export.

## 2-minute demo script

1. Open `/app` and run:
   - `Find the nearest facility in rural Bihar that can perform emergency appendectomy and uses part-time doctors`
2. Show:
   - answer + confidence band
   - trust flags and **Trust Evidence Map**
   - reasoning trace + row-level citations
   - validator loop metadata (`validation_attempts`, `correction_applied`)
3. Switch to **Crisis Map** tab:
   - set `state=Bihar`, `priority=critical`, `min_severity=0.5`
   - click **Load Desert Regions**
   - show top-risk PIN panel + map popups
4. Export planner CSV via top-nav **Export CSV**.

## Current limitations / next steps

- Add robust model fallback chain in-chat (for account-specific model availability)
- Add interactive India map (Leaflet/Mapbox) in UI
- Add confidence intervals / uncertainty quantification for desert/trust outputs
- Add export/download for NGO planning workflows

## Security notes

- Do **not** commit `.env` (contains API keys)
- Rotate keys if accidentally exposed

