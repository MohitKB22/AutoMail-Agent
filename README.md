# 🤖 MailPilot AI — Autonomous Email Intelligence Agent

> An AI-powered email operations agent that understands incoming emails, detects intent and sentiment, drafts contextual replies, routes conversations to the right teams, and safely manages approval workflows.

Built using **Python**, **FastAPI**, and **Streamlit**, this project demonstrates how modern LLM-powered systems can automate inbox operations while keeping humans in control for sensitive communication.

---

# ✨ Features

## 🧠 Intelligent Email Understanding
The agent analyzes incoming emails and extracts:

- Intent classification
- Sentiment detection
- Urgency level
- Confidence score
- Suggested routing team

Supported intents include:

- Meeting requests
- Support requests
- Complaints
- Sales inquiries
- Follow-ups
- Information requests
- Urgent escalations
- Spam detection

---

## 💬 Smart Reply Generation
The AI drafts professional, context-aware responses automatically.

Examples:
- Customer support acknowledgments
- Meeting confirmations
- Sales inquiry replies
- Complaint handling drafts
- Escalation notices

---

## 🔀 Automated Routing Engine
Based on email intent, the system routes conversations to the correct internal team.

| Intent | Team | Priority |
|--------|------|----------|
| Meeting Request | Calendar/Admin | Medium |
| Support Request | Support Team | High |
| Sales Inquiry | Sales Team | Medium |
| Complaint | Customer Success | Urgent |
| Follow Up | Original Handler | Low |
| Information Request | General Queue | Low |
| Urgent | Management | Urgent |
| Spam | Ignored | None |

---

## 🔐 Human-in-the-Loop Approval Flow
Sensitive emails are never auto-approved.

The system automatically flags:
- Complaints
- Urgent escalations
- Negative sentiment emails
- Low-confidence AI predictions

This ensures:
- Responsible AI behavior
- Safer automation
- Reduced hallucination risks
- Human oversight where necessary

---

## 📊 Interactive Dashboard
Built with Streamlit for real-time visibility into:

- Email processing pipeline
- Intent predictions
- Confidence scores
- Draft replies
- Routing decisions
- Approval status

---

# 🏗️ Architecture

```text
                ┌─────────────────┐
                │ Incoming Email  │
                └────────┬────────┘
                         │
                         ▼
              ┌───────────────────┐
              │ Intent Detection  │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ Sentiment Analysis│
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ Reply Generation  │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ Routing Engine    │
              └────────┬──────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Auto Approved       Human Approval

      
