"""
Autonomous Email Intelligence Agent
Uses the openai Python library — works with ANY OpenAI-compatible API:
  OpenAI, Groq, Together AI, OpenRouter, Mistral, local Ollama, etc.
"""

import json
import os
from openai import OpenAI

# ── Client config ──────────────────────────────────────────────────────────────
# Reads from environment variables. Override in .env or pass at runtime.
# Examples:
#   OpenAI       → base_url="https://api.openai.com/v1",        model="gpt-4o"
#   Groq         → base_url="https://api.groq.com/openai/v1",   model="llama3-70b-8192"
#   OpenRouter   → base_url="https://openrouter.ai/api/v1",     model="mistralai/mistral-7b-instruct"
#   Ollama local → base_url="http://localhost:11434/v1",         model="llama3", api_key="ollama"

API_KEY  = os.getenv("API_KEY",  "your-api-key-here")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
MODEL    = os.getenv("MODEL",    "gpt-4o-mini")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

INTENT_CATEGORIES = {
    "meeting_request":     "Schedule or reschedule a meeting",
    "support_request":     "Technical or customer support needed",
    "sales_inquiry":       "Potential sales lead or product question",
    "complaint":           "Customer complaint requiring escalation",
    "follow_up":           "Follow-up on previous conversation",
    "information_request": "Requesting information or documentation",
    "urgent":              "Time-sensitive matter requiring immediate attention",
    "spam":                "Unsolicited or irrelevant email",
}

ROUTING_RULES = {
    "meeting_request":     {"team": "Calendar/Admin",   "auto_approve": True,  "priority": "medium"},
    "support_request":     {"team": "Support Team",     "auto_approve": False, "priority": "high"},
    "sales_inquiry":       {"team": "Sales Team",       "auto_approve": True,  "priority": "medium"},
    "complaint":           {"team": "Customer Success", "auto_approve": False, "priority": "urgent"},
    "follow_up":           {"team": "Original Handler", "auto_approve": True,  "priority": "low"},
    "information_request": {"team": "General",          "auto_approve": True,  "priority": "low"},
    "urgent":              {"team": "Management",       "auto_approve": False, "priority": "urgent"},
    "spam":                {"team": "None",             "auto_approve": True,  "priority": "none"},
}


def _chat(system: str, user: str, max_tokens: int = 600) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def analyze_email(email: dict) -> dict:
    system = "You are an email intelligence system. Always respond with valid JSON only — no markdown, no backticks."
    user = f"""Analyze this email and return a JSON object with exactly these fields:
{{
  "intent": "<meeting_request|support_request|sales_inquiry|complaint|follow_up|information_request|urgent|spam>",
  "sentiment": "<positive|neutral|negative|urgent>",
  "summary": "<1-2 sentence summary of what the sender wants>",
  "key_entities": ["<names, companies, dates, order numbers extracted>"],
  "requires_human_approval": <true|false>,
  "confidence": <0.0-1.0>
}}

EMAIL:
From: {email['from']}
Subject: {email['subject']}
Body: {email['body']}"""

    raw = _chat(system, user, max_tokens=500)
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.split("```")[0]
    return json.loads(raw.strip())


def draft_reply(email: dict, analysis: dict) -> str:
    intent_desc = INTENT_CATEGORIES.get(analysis["intent"], "general inquiry")
    routing     = ROUTING_RULES.get(analysis["intent"], {})

    system = "You are a professional email assistant. Write concise, warm, human-sounding replies."
    user = f"""Draft a reply to this email. Rules:
- Under 150 words
- Professional and warm tone
- Address the sender's core need directly
- If routing to another team, mention the handoff naturally
- Sign as "The Team" — never use placeholder brackets
- If spam: one polite "unsubscribe acknowledged" line only

ORIGINAL EMAIL:
From: {email['from']}
Subject: {email['subject']}
Body: {email['body']}

CONTEXT:
- Detected intent: {analysis['intent']} ({intent_desc})
- Sentiment: {analysis['sentiment']}
- Handling team: {routing.get('team', 'General Team')}"""

    return _chat(system, user, max_tokens=400)


def route_email(analysis: dict) -> dict:
    intent  = analysis.get("intent", "information_request")
    routing = dict(ROUTING_RULES.get(intent, ROUTING_RULES["information_request"]))
    if analysis.get("requires_human_approval"):
        routing["auto_approve"] = False
    routing["approval_status"] = (
        "auto-approved" if routing["auto_approve"] else "pending-human-review"
    )
    return routing


def process_email(email: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Processing: {email['subject']}")
    print(f"{'='*60}")

    print("🔍 Analyzing intent…")
    analysis = analyze_email(email)
    print(f"   Intent:    {analysis['intent']}  ({analysis['confidence']:.0%} confidence)")
    print(f"   Sentiment: {analysis['sentiment']}")
    print(f"   Summary:   {analysis['summary']}")

    print("\n✍️  Drafting reply…")
    draft = draft_reply(email, analysis)
    print(f"   {len(draft.split())} words drafted.")

    print("\n🔀 Routing…")
    routing = route_email(analysis)
    print(f"   Team:     {routing['team']}")
    print(f"   Priority: {routing['priority']}")
    print(f"   Status:   {routing['approval_status']}")

    return {"email": email, "analysis": analysis, "draft_reply": draft, "routing": routing}


SAMPLE_EMAILS = [
    {
        "id": "001",
        "from": "sarah.chen@techcorp.com",
        "subject": "Quick sync this week?",
        "body": "Hi, I'd love to schedule a 30-minute call to discuss the Q2 roadmap alignment. Are you available Thursday or Friday afternoon?",
    },
    {
        "id": "002",
        "from": "angry.customer@gmail.com",
        "subject": "STILL no resolution after 2 weeks!!!",
        "body": "This is absolutely unacceptable. I've been waiting 2 weeks for a refund on order #78432 and nobody has gotten back to me. I will be disputing this charge with my bank if this isn't resolved TODAY.",
    },
    {
        "id": "003",
        "from": "john.smith@startup.io",
        "subject": "Interested in your enterprise plan",
        "body": "Hello, we're a 50-person startup evaluating tools for our engineering team. Could you send over pricing for your enterprise tier? We're also curious about SSO support and SLA guarantees.",
    },
]


if __name__ == "__main__":
    print("🤖 Autonomous Email Intelligence Agent")
    print(f"   Provider : {BASE_URL}")
    print(f"   Model    : {MODEL}\n")

    for email in SAMPLE_EMAILS:
        r = process_email(email)
        print(f"\n   Draft preview: {r['draft_reply'][:120]}…")
