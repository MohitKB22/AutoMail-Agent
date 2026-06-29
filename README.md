# 🤖 Autonomous Email Intelligence Agent

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
python api.py
# API docs at: http://localhost:8000/docs
```

### 4. (Optional) Run the CLI agent directly
```bash
API_KEY=sk-... BASE_URL=https://api.openai.com/v1 MODEL=gpt-4o-mini python agent.py
```

## 📁 Project Structure

```
email-agent/
├── app.py           # Streamlit dashboard (main interface)
├── agent.py         # Core AI pipeline (analyze → draft → route)
├── api.py           # FastAPI REST backend
└── requirements.txt
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
