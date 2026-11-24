# 🤖 Autonomous Email Intelligence Agent

**🔗 Live demo:** LIVE_URL_PLACEHOLDER — opens in 🧪 Demo Mode, no API key needed. *(Free Render instance: the first load after a while can take ~30–60 s to wake up.)*

An AI agent that reads emails, detects intent, drafts replies, and routes approvals safely — with an OpenAI-compatible backend (works with OpenAI, Groq, OpenRouter, Together, Ollama, etc.).

## ✨ Features

- **Intent Detection** — Classifies emails into 8 categories (meeting, complaint, sales, spam, etc.)
- **Sentiment Analysis** — Positive / neutral / negative / urgent
- **Smart Reply Drafting** — Context-aware, professional replies, editable and regeneratable in-place
- **Auto-Routing** — Routes to the right team based on intent
- **Approval Flow** — Risky emails (complaints, urgent) require human review, with one-click approve/reject (single or bulk) and an "undo" back to review
- **🧪 Demo Mode** — Try the entire app with a local rule-based simulator, no API key required
- **Bulk Processing** — Run the built-in sample batch or upload your own CSV of emails (with a downloadable template)
- **Analytics Dashboard** — KPI cards, intent/sentiment/priority charts, search + filters, sort, and JSON/CSV export of the full audit trail
- **Connection Tools** — Provider presets (OpenAI/Groq/OpenRouter/Together/Ollama), a "Test connection" check, and friendly, actionable error messages
- **Light/Dark Theme** — Toggle in the sidebar
- **Beautiful Dashboard** — Streamlit UI with a live, step-by-step processing log for every run

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI    | Any OpenAI-compatible model (OpenAI, Groq, OpenRouter, Together, Ollama, ...) |
| Frontend | Streamlit |
| Backend API | FastAPI |
| Runtime | Python 3.10+ |

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard (recommended)
```bash
streamlit run app.py
```
Flip on **🧪 Demo Mode** in the sidebar to try every feature instantly with no API key — it simulates the pipeline locally. Turn it off and enter a key + base URL + model in the sidebar to use a real provider (with a one-click "Test connection" check).

### 3. (Optional) Run the FastAPI backend
```bash
cp .env.example .env   # add API_KEY / BASE_URL / MODEL for a real model
python api.py          # honours $PORT; set RELOAD=1 for auto-reload
# API docs at: http://localhost:8000/docs
```
No key? Send `"mock": true` in the `POST /emails/process` body to use the Demo Mode simulator.

### 4. (Optional) Run the CLI agent directly
```bash
API_KEY=sk-... BASE_URL=https://api.openai.com/v1 MODEL=gpt-4o-mini python agent.py
python agent.py --demo   # no key: runs the sample batch through the simulator
```

### 5. Run the tests
```bash
pip install -r requirements-dev.txt
pytest -q   # pipeline, API and headless Streamlit UI tests
```

## 📁 Project Structure

```
AutoMail-Agent/
├── app.py                  # Streamlit dashboard (main interface)
├── agent.py                # Core pipeline (analyze → draft → route) + Demo Mode simulator, shared by all entry points
├── api.py                  # FastAPI REST backend
├── tests/                  # pytest: agent, API, and Streamlit AppTest UI flows
├── .streamlit/config.toml  # native theme matched to the app's dark design
├── render.yaml             # Render blueprint (one-click deploy)
├── .env.example
├── requirements.txt
└── requirements-dev.txt
```

## 🔄 Pipeline Flow

```
Email Input
    ↓
[Step 1] Intent Detection (real model, or simulated locally in Demo Mode)
    ↓ intent + sentiment + confidence
[Step 2] Reply Drafting — editable & regeneratable
    ↓ professional reply draft
[Step 3] Routing Engine
    ↓ team + priority assignment
[Step 4] Approval Gate
    → Auto-approved (low-risk intents)
    → Human review required (complaints, urgent, etc.) — approve, reject, or send back to review
```

## 🎯 Intent → Routing Map

| Intent | Team | Priority | Auto-approve |
|--------|------|----------|--------------|
| meeting_request | Calendar/Admin | medium | ✅ Yes |
| support_request | Support Team | high | ❌ No |
| sales_inquiry | Sales Team | medium | ✅ Yes |
| complaint | Customer Success | urgent | ❌ No |
| follow_up | Original Handler | low | ✅ Yes |
| information_request | General | low | ✅ Yes |
| urgent | Management | urgent | ❌ No |
| spam | None | none | ✅ Yes |

## 🔐 Responsible AI Design

- **No auto-send**: Drafts are always surfaced for human review before sending
- **Confidence scores**: Agent reports its certainty on every classification
- **Escalation paths**: Complaints and urgent emails always require human approval
- **Audit trail**: Full processing log stored per email

## ☁️ Deployment (Render)

The dashboard is deployed as a Render web service (see `render.yaml`):

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |

Visitors use Demo Mode by default, or paste their own provider key in the sidebar — keys live only in that browser session's memory and are never stored server-side.

## 🐛 Bug fixes (Nov 2026)

| Area | Problem | Fix |
|---|---|---|
| Setup | `streamlit>=1.32` allowed versions where `width="stretch"` doesn't exist → app crashed on load | Require `streamlit>=1.50` (tested on 1.50 and 1.67) |
| API | `python api.py` exited immediately (`reload=True` with an app object) | Pass `"api:app"` import string; honour `$PORT`; reload opt-in via `RELOAD=1` |
| Compose | Uploading a `.txt` body crashed (`StreamlitWidgetAlreadyInstantiatedError`) | Load the file in an `on_change` callback |
| Theme | On a light-mode OS the dark app rendered dark text on a near-black background; the in-app light toggle left widgets dark | `.streamlit/config.toml` pins the native theme; light mode re-skins widgets and charts from the design tokens |
| Result card | Cards with no entities (or multi-line summaries) showed raw `<hr>`/`<p>` HTML as a code block | Card HTML built on one line, whitespace collapsed |
| Security | Email text, model output and the model name were injected unescaped into HTML | Everything is HTML-escaped |
| Model output | Prose around JSON, unknown intents (`"Meeting Request"`), string/percent confidence, missing keys or empty replies crashed the pipeline | Robust JSON extraction + `normalize_analysis()` validation |
| Providers | Reasoning models (o-series, gpt-5) reject `max_tokens` | Automatic retry with `max_completion_tokens` (also in "Test connection") |
| Config | `python-dotenv` was installed but `.env` was never loaded | `load_dotenv()` in `agent.py` |
| Demo Mode | Substring keyword matching (`plan` in "explanation", `call` in "recall") misclassified emails; confidence changed between restarts; entities were filler words ("This", "STILL") | Whole-word matching, deterministic confidence, stop-word filtering |
| Batch CSV | Excel "CSV UTF-8" files (with BOM) failed with "Missing column: from"; parse errors also showed a second bogus warning | Decode with `utf-8-sig`; single clear message |
| Dashboard | "Auto-Approved" KPI kept counting emails sent back to review, so totals didn't add up | KPIs count by current status |
| Dashboard | Regenerate / Re-run errors vanished instantly (rerun after `st.error`) | Rerun only on success |
| API | Generated ids could collide with client ids; approvals hit the first duplicate; empty emails were sent to the model; missing key gave an opaque 500 | UUID ids + upsert store, request validation (422), 503 with a hint when no key, 502 for provider errors |
