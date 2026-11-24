"""
Autonomous Email Intelligence Agent — core pipeline (analyze → draft → route).

Uses the openai Python library, so it works with ANY OpenAI-compatible API:
  OpenAI, Groq, Together AI, OpenRouter, Mistral, local Ollama, etc.

This module is shared by the CLI (`python agent.py`), the FastAPI backend
(`api.py`) and the Streamlit dashboard (`app.py`). It also contains the local,
rule-based "Demo Mode" simulator so every entry point can run without a key.
"""

from __future__ import annotations

import json
import os
import re
import zlib
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

# ── Client config ──────────────────────────────────────────────────────────────
# Reads from environment variables (or a local .env file). Examples:
#   OpenAI       → BASE_URL=https://api.openai.com/v1         MODEL=gpt-4o-mini
#   Groq         → BASE_URL=https://api.groq.com/openai/v1    MODEL=llama-3.3-70b-versatile
#   OpenRouter   → BASE_URL=https://openrouter.ai/api/v1      MODEL=mistralai/mistral-7b-instruct
#   Ollama local → BASE_URL=http://localhost:11434/v1         MODEL=llama3   API_KEY=ollama
load_dotenv()

API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip() or "https://api.openai.com/v1"
MODEL = os.getenv("MODEL", "").strip() or "gpt-4o-mini"

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
DEFAULT_INTENT = "information_request"
SENTIMENTS = ("positive", "neutral", "negative", "urgent")

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


class MissingAPIKeyError(RuntimeError):
    """Raised when a real model call is attempted without an API key configured."""


# ── LLM plumbing ───────────────────────────────────────────────────────────────
_default_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazily build the env-configured client (so importing this module never fails)."""
    global _default_client
    if not API_KEY:
        raise MissingAPIKeyError(
            "API_KEY is not set. Export API_KEY (plus BASE_URL / MODEL for non-OpenAI "
            "providers) or put them in a .env file — or use Demo Mode."
        )
    if _default_client is None:
        _default_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _default_client


def chat(client: OpenAI, model: str, system: str, user: str, max_tokens: int = 600) -> str:
    """Single chat completion that tolerates provider quirks.

    Newer OpenAI reasoning models (o-series, gpt-5…) reject ``max_tokens`` and
    require ``max_completion_tokens``; retry with that instead of failing.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        resp = client.chat.completions.create(model=model, max_tokens=max_tokens, messages=messages)
    except BadRequestError as e:
        if "max_tokens" not in str(e):
            raise
        resp = client.chat.completions.create(
            model=model, max_completion_tokens=max_tokens * 4, messages=messages
        )
    content = resp.choices[0].message.content if resp.choices else None
    if not content or not content.strip():
        raise ValueError("The model returned an empty response — try a different model.")
    return content.strip()


# ── Parsing & normalisation of model output ────────────────────────────────────
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def parse_json_object(raw: str) -> dict:
    """Extract a JSON object from a model reply.

    Handles ```json fences, prose before/after the object, and trailing text —
    all of which small/open models produce regularly.
    """
    text = (raw or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("The model didn't return a JSON object.")
    return obj


_INTENT_ALIASES = {
    "meeting": "meeting_request", "schedule": "meeting_request", "scheduling": "meeting_request",
    "support": "support_request", "technical_support": "support_request",
    "sales": "sales_inquiry", "sales_lead": "sales_inquiry", "lead": "sales_inquiry",
    "followup": "follow_up", "follow_up_request": "follow_up",
    "info_request": "information_request", "information": "information_request",
    "question": "information_request", "inquiry": "information_request",
    "escalation": "complaint", "junk": "spam", "promotion": "spam", "marketing": "spam",
}


def normalize_intent(value: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    if key in INTENT_CATEGORIES:
        return key
    if key in _INTENT_ALIASES:
        return _INTENT_ALIASES[key]
    for intent in INTENT_CATEGORIES:  # e.g. "meeting_request_reschedule" → meeting_request
        if key.startswith(intent):
            return intent
    return DEFAULT_INTENT


def normalize_confidence(value: Any) -> float:
    """Coerce 0.87 / "0.87" / "87%" / 87 → 0.87, clamped to [0, 1]."""
    if isinstance(value, bool):
        return 0.5
    if isinstance(value, str):
        value = value.strip().rstrip("%").strip()
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5
    if f != f:  # NaN
        return 0.5
    if f > 1:
        f /= 100.0
    return round(min(max(f, 0.0), 1.0), 2)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "y", "1")
    return bool(value)


def normalize_analysis(data: dict) -> dict:
    """Validate/clean an analysis dict so downstream code never crashes on bad model output."""
    data = data if isinstance(data, dict) else {}
    sentiment = str(data.get("sentiment") or "").strip().lower()
    entities = data.get("key_entities") or []
    if isinstance(entities, str):
        entities = [entities]
    if not isinstance(entities, list):
        entities = []
    clean_entities = list(dict.fromkeys(str(e).strip() for e in entities if str(e).strip()))[:10]
    return {
        "intent": normalize_intent(data.get("intent")),
        "sentiment": sentiment if sentiment in SENTIMENTS else "neutral",
        "summary": str(data.get("summary") or "").strip() or "No summary provided.",
        "key_entities": clean_entities,
        "requires_human_approval": _as_bool(data.get("requires_human_approval", False)),
        "confidence": normalize_confidence(data.get("confidence", 0.5)),
    }


# ── Pipeline steps (real model) ────────────────────────────────────────────────
def _email_block(email: dict) -> str:
    return (f"From: {email.get('from', '')}\n"
            f"Subject: {email.get('subject', '')}\n"
            f"Body: {email.get('body', '')}")


def analyze_email(email: dict, client: OpenAI | None = None, model: str | None = None) -> dict:
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
{_email_block(email)}"""

    raw = chat(client or get_client(), model or MODEL, system, user, max_tokens=500)
    return normalize_analysis(parse_json_object(raw))


def draft_reply(email: dict, analysis: dict, client: OpenAI | None = None,
                model: str | None = None, team: str | None = None) -> str:
    intent = normalize_intent(analysis.get("intent"))
    intent_desc = INTENT_CATEGORIES[intent]
    team = team or ROUTING_RULES[intent]["team"]

    system = "You are a professional email assistant. Write concise, warm, human-sounding replies."
    user = f"""Draft a reply to this email. Rules:
- Under 150 words
- Professional and warm tone
- Address the sender's core need directly
- If routing to another team, mention the handoff naturally
- Sign as "The Team" — never use placeholder brackets
- If spam: one polite "unsubscribe acknowledged" line only

ORIGINAL EMAIL:
{_email_block(email)}

CONTEXT:
- Detected intent: {intent} ({intent_desc})
- Sentiment: {analysis.get('sentiment', 'neutral')}
- Handling team: {team}"""

    return chat(client or get_client(), model or MODEL, system, user, max_tokens=400)


def route_email(analysis: dict, rules: dict | None = None) -> dict:
    rules = rules or ROUTING_RULES
    intent = normalize_intent(analysis.get("intent"))
    routing = dict(rules.get(intent, rules[DEFAULT_INTENT]))
    if analysis.get("requires_human_approval"):
        routing["auto_approve"] = False
    routing["approval_status"] = "auto-approved" if routing["auto_approve"] else "pending-human-review"
    return routing


# ── Demo Mode: local rule-based simulator (no API key required) ────────────────
def _patterns(words: Iterable[str]) -> list[re.Pattern]:
    # Whole-word matching so "plan" doesn't fire on "explanation", "call" on "recall", etc.
    return [re.compile(rf"(?<![\w]){w}(?![\w])") for w in words]


_NEG = _patterns([r"unacceptable", r"angry", r"frustrat\w*", r"disappoint\w*", r"refunds?", r"complain\w*",
                  r"terrible", r"worst", r"awful", r"never again", r"disput\w*", r"cancel\w*"])
_URGENT = _patterns([r"urgent\w*", r"asap", r"immediately", r"right away", r"emergency", r"critical", r"today"])
_MEETING = _patterns([r"meetings?", r"calls?", r"sync", r"schedul\w*", r"calendar", r"available",
                      r"availability", r"catch up"])
_SALES = _patterns([r"pricing", r"prices?", r"quotes?", r"enterprise", r"plans?", r"demo", r"trial",
                    r"interested in", r"purchas\w*", r"sla", r"sso"])
_SUPPORT = _patterns([r"bugs?", r"errors?", r"issues?", r"not working", r"broken", r"help", r"support",
                      r"crash\w*", r"trouble"])
_SPAM = _patterns([r"unsubscribe", r"free gift", r"click here", r"exclusive offer", r"winner",
                   r"congratulations", r"limited time", r"\d+\s?% off"])
_FOLLOWUP = _patterns([r"following up", r"follow up", r"just checking", r"circling back", r"any updates?",
                       r"touching base"])
_POSITIVE = _patterns([r"thanks", r"thank you", r"great", r"appreciate\w*", r"looking forward"])

_ENTITY_RE = re.compile(r"#\d+|\b[QH][1-4]\b|\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,})?\b")
_ENTITY_STOPWORDS = {
    "hi", "hello", "hey", "dear", "thanks", "thank", "this", "that", "these", "those", "the", "there",
    "are", "could", "would", "can", "will", "please", "best", "regards", "cheers", "our", "we", "you",
    "your", "what", "when", "where", "how", "why", "who", "just", "also", "let", "any", "and", "but",
    "for", "not", "nobody", "still", "quick", "click", "limited", "interested", "issue", "question",
    "sorry", "yes", "sure", "great", "looking", "following", "unsubscribe", "free", "hot", "deals",
    "week", "only", "exclusive", "offer", "selected", "you've", "i'd", "i'm", "it", "is", "if", "my",
}


def _extract_entities(text: str) -> list[str]:
    out = []
    for match in _ENTITY_RE.findall(text):
        words = [w for w in match.split() if w.lower() not in _ENTITY_STOPWORDS
                 and not (w.isupper() and len(w) > 3)]  # skip SHOUTING words like "STILL"/"TODAY"
        if words:
            out.append(" ".join(words))
    return list(dict.fromkeys(out))[:6]


def mock_analyze(email: dict) -> dict:
    subject = email.get("subject", "") or ""
    body = email.get("body", "") or ""
    text = f"{subject} {body}".lower()

    def hits(patterns):
        return sum(1 for p in patterns if p.search(text))

    scores = {
        "complaint": hits(_NEG) * 2,
        "urgent": hits(_URGENT) * 2,
        "meeting_request": hits(_MEETING),
        "sales_inquiry": hits(_SALES),
        "support_request": hits(_SUPPORT),
        "spam": hits(_SPAM) * 2,
        "follow_up": hits(_FOLLOWUP) * 2,
    }
    bang = text.count("!")
    letters = [c for c in subject if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
    if bang >= 2:
        scores["urgent"] += 1
        scores["complaint"] += 1
    if caps_ratio > 0.3:
        scores["spam"] += 1

    intent = max(scores, key=scores.get) if max(scores.values(), default=0) > 0 else DEFAULT_INTENT

    if scores["complaint"] >= 2:
        sentiment = "negative"
    elif intent == "urgent":
        sentiment = "urgent"
    elif bang >= 2 or hits(_NEG):
        sentiment = "negative"
    elif hits(_POSITIVE):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    flat_body = " ".join(body.split())
    summary = (flat_body[:140] + "…") if len(flat_body) > 140 else flat_body

    return {
        "intent": intent,
        "sentiment": sentiment,
        "summary": summary or "No content provided.",
        "key_entities": _extract_entities(f"{subject} {body}"),
        "requires_human_approval": intent in ("complaint", "urgent", "support_request"),
        # Deterministic pseudo-confidence (built-in hash() is salted per process).
        "confidence": round(0.74 + (zlib.crc32(text.encode("utf-8")) % 21) / 100, 2),
    }


MOCK_TEMPLATES = {
    "meeting_request": "Hi,\n\nThanks for reaching out — happy to find time this week. {team} will follow up shortly with a couple of slots that work.\n\nBest,\nThe Team",
    "support_request": "Hi,\n\nThanks for flagging this. I've routed your message to {team} so we can dig into it properly — you'll hear back shortly with next steps.\n\nBest,\nThe Team",
    "sales_inquiry": "Hi,\n\nThanks for your interest! I've looped in {team}, who'll follow up with pricing details and answers to your questions.\n\nBest,\nThe Team",
    "complaint": "Hi,\n\nI'm sorry for the trouble this has caused — that's not the experience we want for you. I've escalated this to {team} for immediate attention.\n\nBest,\nThe Team",
    "follow_up": "Hi,\n\nThanks for checking in! {team} has this on file and will get back to you shortly with an update.\n\nBest,\nThe Team",
    "information_request": "Hi,\n\nThanks for reaching out. {team} has received your request and will send over the information shortly.\n\nBest,\nThe Team",
    "urgent": "Hi,\n\nThank you for flagging this as urgent. I've escalated it directly to {team} for immediate review.\n\nBest,\nThe Team",
    "spam": "Unsubscribe acknowledged — you won't hear from this list again.",
}


def mock_draft_reply(email: dict, analysis: dict, team: str | None = None) -> str:
    intent = normalize_intent(analysis.get("intent"))
    team = team or f"our {ROUTING_RULES[intent]['team']}"
    return MOCK_TEMPLATES[intent].format(team=team)


# ── Orchestration ──────────────────────────────────────────────────────────────
def process_email(email: dict, client: OpenAI | None = None, model: str | None = None,
                  mock: bool = False, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    log(f"\n{'=' * 60}")
    log(f"Processing: {email.get('subject', '(no subject)')}")
    log(f"{'=' * 60}")

    log("🔍 Analyzing intent…")
    analysis = mock_analyze(email) if mock else analyze_email(email, client, model)
    log(f"   Intent:    {analysis['intent']}  ({analysis['confidence']:.0%} confidence)")
    log(f"   Sentiment: {analysis['sentiment']}")
    log(f"   Summary:   {analysis['summary']}")

    log("\n✍️  Drafting reply…")
    draft = mock_draft_reply(email, analysis) if mock else draft_reply(email, analysis, client, model)
    log(f"   {len(draft.split())} words drafted.")

    log("\n🔀 Routing…")
    routing = route_email(analysis)
    log(f"   Team:     {routing['team']}")
    log(f"   Priority: {routing['priority']}")
    log(f"   Status:   {routing['approval_status']}")

    return {"email": email, "analysis": analysis, "draft_reply": draft, "routing": routing, "mock": mock}


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
    {
        "id": "004",
        "from": "noreply@newsletters.biz",
        "subject": "🔥 HOT DEALS This Week Only!!!",
        "body": "You've been selected for our EXCLUSIVE offer! Click here to claim your FREE gift. Limited time only.",
    },
]


if __name__ == "__main__":
    import sys

    demo = "--demo" in sys.argv or not API_KEY
    print("🤖 Autonomous Email Intelligence Agent")
    if demo:
        print("   Mode     : 🧪 Demo (local simulator — set API_KEY to use a real model)\n")
    else:
        print(f"   Provider : {BASE_URL}")
        print(f"   Model    : {MODEL}\n")

    for sample in SAMPLE_EMAILS:
        r = process_email(sample, mock=demo)
        print(f"\n   Draft preview: {r['draft_reply'][:120]}…")
