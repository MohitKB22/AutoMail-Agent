# 🤖 Autonomous Email Intelligence Agent

An AI agent that reads emails, detects intent, drafts replies, and routes approvals safely using Claude.

## ✨ Features

- **Intent Detection** — Classifies emails into 8 categories (meeting, complaint, sales, spam, etc.)
- **Sentiment Analysis** — Positive / neutral / negative / urgent
- **Smart Reply Drafting** — Context-aware, professional replies
- **Auto-Routing** — Routes to the right team based on intent
- **Approval Flow** — Risky emails (complaints, urgent) require human review
- **Beautiful Dashboard** — Streamlit UI with real-time processing log

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI    | Claude (claude-sonnet-4-20250514) |
| Frontend | Streamlit |
| Backend API | FastAPI |
| Runtime | Python 3.10+ |


## 📁 Project Structure

```
email-agent/
├── app.py           # Streamlit UI (main interface)
├── agent.py         # Core AI pipeline (analyze → draft → route)
├── api.py           # FastAPI REST backend
├── requirements.txt
└── .env.example
```

## 🔄 Pipeline Flow

```
Email Input
    ↓
[Step 1] Intent Detection via Claude
    ↓ intent + sentiment + confidence
[Step 2] Reply Drafting via Claude
    ↓ professional reply draft
[Step 3] Routing Engine
    ↓ team + priority assignment
[Step 4] Approval Gate
    → Auto-approved (low-risk intents)
    → Human review required (complaints, urgent, etc.)
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
