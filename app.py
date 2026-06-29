"""
Autonomous Email Intelligence Agent — Streamlit Dashboard
Works with ANY OpenAI-compatible provider (OpenAI, Groq, OpenRouter, Together, Ollama, ...).
"""

import csv
import io
import json
import re
import uuid
from datetime import datetime

import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="Email Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ────────────────────────────────────────────────────────────
THEMES = {
    "dark": dict(
        bg="#0a0a0f", surface="#111118", surface2="#0d1117",
        border="#1e1e2e", border2="#21262d",
        text="#e5e7eb", text_muted="#9ca3af", text_dim="#6b7280",
        accent1="#818cf8", accent2="#c084fc", code_text="#c9d1d9",
        shadow="rgba(0,0,0,0.45)",
    ),
    "light": dict(
        bg="#f5f5fb", surface="#ffffff", surface2="#f3f4f9",
        border="#e3e4ee", border2="#e9e9f2",
        text="#1c1c28", text_muted="#52576b", text_dim="#787e92",
        accent1="#4f46e5", accent2="#9333ea", code_text="#1c1c28",
        shadow="rgba(30,30,60,0.10)",
    ),
}


def inject_css(theme_name: str) -> None:
    t = THEMES.get(theme_name, THEMES["dark"])
    css_vars = "\n".join(f"  --{k.replace('_', '-')}: {v};" for k, v in t.items())
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');
:root {{
{css_vars}
}}
html, body, [class*="css"] {{ font-family: 'Syne', sans-serif; }}
.stApp {{ background: var(--bg); }}
h1, h2, h3 {{ font-family: 'Syne', sans-serif !important; }}

.email-card {{
    background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--border);
    border-radius: 12px; padding: 20px; margin: 10px 0; transition: box-shadow .15s ease;
}}
.email-card:hover {{ box-shadow: 0 4px 16px var(--shadow); }}
.email-card.priority-border-urgent {{ border-left-color: #ef4444; }}
.email-card.priority-border-high {{ border-left-color: #f97316; }}
.email-card.priority-border-medium {{ border-left-color: #eab308; }}
.email-card.priority-border-low {{ border-left-color: #22c55e; }}
.email-card.priority-border-none {{ border-left-color: #6b7280; }}

.card-top {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.card-row {{ color:var(--text-muted); margin:8px 0 0; font-size:13px; display:flex; justify-content:space-between; gap:8px; }}
.card-row b, .card-row span {{ color:var(--text); }}
.card-hr {{ border-color:var(--border); margin:12px 0; }}
.card-label {{ color:var(--text-dim); font-size:11px; letter-spacing:.08em; margin-bottom:4px; text-transform:uppercase; }}
.card-summary {{ color:var(--code-text); font-size:14px; margin:0; line-height:1.5; }}
.entity-row {{ margin-top:10px; display:flex; flex-wrap:wrap; gap:6px; }}
.entity-chip {{ font-size:11px; padding:3px 9px; border-radius:8px; background:var(--surface2); border:1px solid var(--border2); color:var(--text-muted); font-family:'JetBrains Mono',monospace; }}
.sim-tag {{ font-size:10px; padding:2px 8px; border-radius:10px; background:rgba(192,132,252,0.15); color:var(--accent2); font-family:'JetBrains Mono',monospace; letter-spacing:.05em; }}

.intent-badge {{
    display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600;
    font-family:'JetBrains Mono',monospace; text-transform:uppercase; letter-spacing:0.05em;
}}
.badge-meeting_request {{ background:#1e3a5f; color:#60a5fa; }}
.badge-support_request {{ background:#3b1f1f; color:#f87171; }}
.badge-sales_inquiry {{ background:#1a3a1a; color:#4ade80; }}
.badge-complaint {{ background:#3b1f1f; color:#fb923c; }}
.badge-follow_up {{ background:#2a1f3b; color:#c084fc; }}
.badge-information_request {{ background:#1f2e3b; color:#38bdf8; }}
.badge-urgent {{ background:#3b1f1f; color:#ef4444; }}
.badge-spam {{ background:#1f1f1f; color:#9ca3af; }}

.priority-urgent {{ color:#ef4444; font-weight:600; }}
.priority-high {{ color:#f97316; font-weight:600; }}
.priority-medium {{ color:#eab308; font-weight:600; }}
.priority-low {{ color:#22c55e; font-weight:600; }}
.priority-none {{ color:#6b7280; font-weight:600; }}

.status-auto-approved {{ color:#4ade80; font-weight:600; }}
.status-pending-human-review {{ color:#fbbf24; font-weight:600; }}
.status-approved {{ color:#34d399; font-weight:600; }}
.status-rejected {{ color:#f87171; font-weight:600; }}

.metric-card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:18px; text-align:center; }}
.metric-value {{ font-size:30px; font-weight:800; }}
.metric-label {{ font-size:11px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.1em; margin-top:2px; }}

.status-chip {{
    display:inline-block; padding:4px 12px; border-radius:8px; font-size:12px;
    font-family:'JetBrains Mono',monospace; background:var(--surface2); border:1px solid var(--border2); color:var(--text-muted);
}}

div[data-testid="stCodeBlock"] pre {{ font-size: 13px !important; }}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
    border: none !important;
}}
section[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
</style>
""",
        unsafe_allow_html=True,
    )


# ── Domain constants ─────────────────────────────────────────────────────────
INTENT_CATEGORIES = {
    "meeting_request": "Schedule or reschedule a meeting",
    "support_request": "Technical or customer support needed",
    "sales_inquiry": "Potential sales lead or product question",
    "complaint": "Customer complaint requiring escalation",
    "follow_up": "Follow-up on previous conversation",
    "information_request": "Requesting information or documentation",
    "urgent": "Time-sensitive matter",
    "spam": "Unsolicited or irrelevant email",
}
ROUTING_RULES = {
    "meeting_request": {"team": "📅 Calendar/Admin", "auto_approve": True, "priority": "medium"},
    "support_request": {"team": "🛠️ Support Team", "auto_approve": False, "priority": "high"},
    "sales_inquiry": {"team": "💼 Sales Team", "auto_approve": True, "priority": "medium"},
    "complaint": {"team": "🚨 Customer Success", "auto_approve": False, "priority": "urgent"},
    "follow_up": {"team": "🔁 Original Handler", "auto_approve": True, "priority": "low"},
    "information_request": {"team": "📋 General", "auto_approve": True, "priority": "low"},
    "urgent": {"team": "⚡ Management", "auto_approve": False, "priority": "urgent"},
    "spam": {"team": "🗑️ None", "auto_approve": True, "priority": "none"},
}
SAMPLE_EMAILS = [
    {"from": "sarah.chen@techcorp.com", "subject": "Quick sync this week?",
     "body": "Hi, I'd love to schedule a 30-minute call to discuss the Q2 roadmap alignment. Are you available Thursday or Friday afternoon?"},
    {"from": "angry.customer@gmail.com", "subject": "STILL no resolution after 2 weeks!!!",
     "body": "This is absolutely unacceptable. I've been waiting 2 weeks for a refund on order #78432 and nobody has gotten back to me. I will be disputing this charge with my bank if this isn't resolved TODAY."},
    {"from": "john.smith@startup.io", "subject": "Interested in your enterprise plan",
     "body": "Hello, we're a 50-person startup evaluating tools for our engineering team. Could you send over pricing for your enterprise tier? We're also curious about SSO and SLA guarantees."},
    {"from": "noreply@newsletters.biz", "subject": "🔥 HOT DEALS This Week Only!!!",
     "body": "You've been selected for our EXCLUSIVE offer! Click here to claim your FREE gift. Limited time only."},
]
PROVIDER_PRESETS = {
    "OpenAI": {"base_url": "https://api.openai.com/v1", "models": ["gpt-4o-mini", "gpt-4o"]},
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]},
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "models": ["mistralai/mistral-7b-instruct"]},
    "Together AI": {"base_url": "https://api.together.xyz/v1", "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"]},
    "Ollama (local)": {"base_url": "http://localhost:11434/v1", "models": ["llama3"]},
    "Custom": {"base_url": "", "models": []},
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


def _build_csv_template() -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["from", "subject", "body"])
    w.writerow(["jane@example.com", "Quick question about pricing", "Hi, could you send over your enterprise pricing?"])
    w.writerow(["mike@example.com", "Issue with my order", "My order #4521 hasn't arrived and support hasn't replied."])
    return buf.getvalue().encode("utf-8")


CSV_TEMPLATE = _build_csv_template()

# ── Session-state defaults ───────────────────────────────────────────────────
_DEFAULTS = {
    "processed_emails": [],
    "last_compose_result": None,
    "theme": "dark",
    "mock_mode": True,
    "provider_choice": "OpenAI",
    "base_url": PROVIDER_PRESETS["OpenAI"]["base_url"],
    "model": PROVIDER_PRESETS["OpenAI"]["models"][0],
    "api_key": "",
    "conn_status": None,
    "compose_from": "",
    "compose_subject": "",
    "compose_body": "",
    "compose_file_loaded": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── AI helpers (real provider) ───────────────────────────────────────────────
def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key or "none", base_url=base_url)


def _chat(client, model, system, user, max_tokens=600):
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


def analyze_email(client, model, email):
    system = "You are an email intelligence system. Respond with valid JSON only — no markdown, no backticks."
    user = f"""Return a JSON object with exactly these fields:
{{
  "intent": "<meeting_request|support_request|sales_inquiry|complaint|follow_up|information_request|urgent|spam>",
  "sentiment": "<positive|neutral|negative|urgent>",
  "summary": "<1-2 sentence summary>",
  "key_entities": ["<extracted names/companies/dates>"],
  "requires_human_approval": <true|false>,
  "confidence": <0.0-1.0>
}}
EMAIL:
From: {email['from']}
Subject: {email['subject']}
Body: {email['body']}"""
    raw = _chat(client, model, system, user, 500)
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.split("```")[0]
    return json.loads(raw.strip())


def draft_reply(client, model, email, analysis):
    routing = ROUTING_RULES.get(analysis["intent"], {})
    system = "You are a professional email assistant. Write concise, warm, human-sounding replies."
    user = f"""Draft a reply. Rules: under 150 words, warm & professional, sign as "The Team", no placeholders.
If spam: one polite unsubscribe-acknowledged line only.
EMAIL — From: {email['from']} | Subject: {email['subject']}
Body: {email['body']}
CONTEXT — Intent: {analysis['intent']} | Sentiment: {analysis['sentiment']} | Team: {routing.get('team', 'General')}"""
    return _chat(client, model, system, user, 400)


def route_email(analysis):
    intent = analysis.get("intent", "information_request")
    routing = dict(ROUTING_RULES.get(intent, ROUTING_RULES["information_request"]))
    if analysis.get("requires_human_approval"):
        routing["auto_approve"] = False
    routing["approval_status"] = "auto-approved" if routing["auto_approve"] else "pending-human-review"
    return routing


# ── Mock / offline pipeline (no API key required) ────────────────────────────
_NEG = ("unacceptable", "angry", "frustrat", "disappoint", "refund", "complain", "terrible",
        "worst", "awful", "never again", "disputing", "cancel")
_URGENT = ("urgent", "asap", "immediately", "right away", "emergency", "critical", "today")
_MEETING = ("meeting", "call", "sync", "schedule", "calendar", "available", "availability", "catch up")
_SALES = ("pricing", "price", "quote", "enterprise", "plan", "demo", "trial", "interested in", "purchase", "sla", "sso")
_SUPPORT = ("bug", "error", "issue", "not working", "broken", "help", "support", "crash", "trouble")
_SPAM = ("unsubscribe", "free gift", "click here", "exclusive offer", "winner", "congratulations", "limited time", "% off")
_FOLLOWUP = ("following up", "follow up", "just checking", "circling back", "any update", "touching base")


def mock_analyze(email):
    text = f"{email.get('subject', '')} {email.get('body', '')}".lower()

    def hits(words):
        return sum(1 for w in words if w in text)

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
    subj = email.get("subject", "")
    caps_ratio = sum(1 for c in subj if c.isupper()) / max(len(subj), 1)
    if bang >= 2:
        scores["urgent"] += 1
        scores["complaint"] += 1
    if caps_ratio > 0.3:
        scores["spam"] += 1

    intent = max(scores, key=scores.get) if max(scores.values(), default=0) > 0 else "information_request"

    if scores.get("complaint", 0) >= 2:
        sentiment = "negative"
    elif intent == "urgent":
        sentiment = "urgent"
    elif bang >= 2 or hits(_NEG):
        sentiment = "negative"
    elif any(w in text for w in ("thanks", "great", "appreciate", "looking forward")):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    raw_entities = re.findall(r"#\d+|\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,})?\b",
                               f"{email.get('subject', '')} {email.get('body', '')}")
    entities = list(dict.fromkeys(raw_entities))[:6]

    body = (email.get("body", "") or "").strip().replace("\n", " ")
    summary = (body[:140] + "…") if len(body) > 140 else body

    return {
        "intent": intent,
        "sentiment": sentiment,
        "summary": summary or "No content provided.",
        "key_entities": entities,
        "requires_human_approval": intent in ("complaint", "urgent", "support_request"),
        "confidence": round(0.74 + (hash(text) % 21) / 100, 2),
    }


def mock_draft_reply(email, analysis):
    intent = analysis.get("intent", "information_request")
    team = ROUTING_RULES.get(intent, {}).get("team", "our team")
    template = MOCK_TEMPLATES.get(intent, MOCK_TEMPLATES["information_request"])
    return template.format(team=team)


# ── Shared pipeline orchestration ────────────────────────────────────────────
def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "401" in msg or "authentic" in low or "api key" in low:
        return "🔑 Authentication failed — double-check your API key in the sidebar."
    if "404" in msg or "model_not_found" in low or "does not exist" in low:
        return "🤖 Model not found — check the model name for this provider."
    if "connect" in low or "timed out" in low or "timeout" in low or "name resolution" in low:
        return "🌐 Couldn't reach the provider — check the Base URL and your network."
    if "json" in low or isinstance(e, json.JSONDecodeError):
        return "🧩 The model didn't return valid JSON — try a different or larger model."
    if "429" in msg or "rate limit" in low:
        return "⏳ Rate limited by the provider — wait a moment and try again."
    return f"⚠️ {msg}"


def process_full(client, model, email, mock=False, expanded=True):
    result = {"id": uuid.uuid4().hex[:8], "email": email, "timestamp": datetime.now().strftime("%H:%M:%S")}
    with st.status("Processing email…", expanded=expanded) as status:
        try:
            st.write("🔍 Analyzing intent & sentiment…")
            analysis = mock_analyze(email) if mock else analyze_email(client, model, email)
            result["analysis"] = analysis
            st.write(f"→ **{analysis['intent'].replace('_', ' ')}** · {analysis['sentiment']} · {analysis['confidence']:.0%} confidence")

            st.write("✍️ Drafting reply…")
            draft = mock_draft_reply(email, analysis) if mock else draft_reply(client, model, email, analysis)
            result["draft"] = draft
            st.write(f"→ {len(draft.split())} words drafted")

            st.write("🔀 Routing & queueing…")
            routing = route_email(analysis)
            result["routing"] = routing
            st.write(f"→ {routing['team']} · priority **{routing['priority']}** · {routing['approval_status']}")

            status.update(label="✅ Done", state="complete")
            result["mock"] = mock
            return result, None
        except Exception as e:  # noqa: BLE001
            status.update(label="❌ Failed", state="error")
            return None, e


def parse_email_csv(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("Couldn't read any columns from this CSV.")
    cols = {c.lower().strip(): c for c in reader.fieldnames}

    def find(*aliases):
        for a in aliases:
            if a in cols:
                return cols[a]
        return None

    f_from, f_subj, f_body = find("from", "sender", "email"), find("subject", "title"), find("body", "message", "content", "text")
    missing = [n for n, v in [("from", f_from), ("subject", f_subj), ("body", f_body)] if v is None]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Found: {', '.join(reader.fieldnames)}")
    rows = [{"from": (row.get(f_from) or "").strip(), "subject": (row.get(f_subj) or "").strip(),
             "body": (row.get(f_body) or "").strip()} for row in reader]
    return [r for r in rows if r["subject"] or r["body"]]


def counts_for(emails, key_fn, categories):
    counts = {c: 0 for c in categories}
    for r in emails:
        k = key_fn(r)
        counts[k] = counts.get(k, 0) + 1
    return counts


def to_export_rows(emails):
    rows = []
    for r in emails:
        e, a, rt = r["email"], r["analysis"], r["routing"]
        rows.append({
            "id": r["id"], "timestamp": r.get("timestamp", ""), "from": e.get("from", ""), "subject": e.get("subject", ""),
            "intent": a.get("intent", ""), "sentiment": a.get("sentiment", ""), "confidence": a.get("confidence", ""),
            "team": rt.get("team", ""), "priority": rt.get("priority", ""), "status": rt.get("approval_status", ""),
            "summary": a.get("summary", ""), "draft_reply": r.get("draft", ""),
        })
    return rows


def export_json_bytes(emails):
    return json.dumps(to_export_rows(emails), indent=2, ensure_ascii=False).encode("utf-8")


def export_csv_bytes(emails):
    rows = to_export_rows(emails)
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ── Render helpers (shared between Compose tab and Dashboard tab) ────────────
def render_result_card(result):
    a, r = result["analysis"], result["routing"]
    intent = a.get("intent", "information_request")
    sim_tag = ' <span class="sim-tag">SIMULATED</span>' if result.get("mock") else ""

    entities_html = ""
    ents = a.get("key_entities") or []
    if ents:
        chips = "".join(f'<span class="entity-chip">{e}</span>' for e in ents[:6])
        entities_html = f'<div class="entity-row">{chips}</div>'

    st.markdown(f"""
    <div class="email-card priority-border-{r.get('priority', 'none')}">
        <div class="card-top">
            <span class="intent-badge badge-{intent}">{intent.replace('_', ' ')}</span>{sim_tag}
        </div>
        <p class="card-row">Sentiment <b>{a.get('sentiment', '—')}</b></p>
        <p class="card-row">Confidence <b>{a.get('confidence', 0):.0%}</b></p>
        <p class="card-row">Routes to <b>{r.get('team', '—')}</b></p>
        <p class="card-row">Priority <span class="priority-{r.get('priority', 'none')}">{r.get('priority', 'none').upper()}</span></p>
        <p class="card-row">Status <span class="status-{r.get('approval_status', '').replace(' ', '-')}">{r.get('approval_status', '—')}</span></p>
        {entities_html}
        <hr class="card-hr">
        <p class="card-label">Summary</p>
        <p class="card-summary">{a.get('summary', '—')}</p>
    </div>""", unsafe_allow_html=True)


def render_draft_block(result, context):
    rid = result["id"]
    st.markdown('<p class="card-label" style="margin-top:14px;">Draft reply</p>', unsafe_allow_html=True)
    st.code(result.get("draft", ""), language=None)

    with st.expander("✏️ Edit draft"):
        new_text = st.text_area("Edit", value=result.get("draft", ""), height=160,
                                 key=f"{context}_ta_{rid}", label_visibility="collapsed")
        ec1, ec2 = st.columns(2)
        with ec1:
            if st.button("💾 Save edit", key=f"{context}_save_{rid}", width="stretch"):
                result["draft"] = new_text
                st.toast("Draft updated")
                st.rerun()
        with ec2:
            can_regen = st.session_state.mock_mode or bool(st.session_state.api_key)
            if st.button("🔁 Regenerate", key=f"{context}_regen_{rid}", width="stretch", disabled=not can_regen):
                try:
                    if st.session_state.mock_mode:
                        result["draft"] = mock_draft_reply(result["email"], result["analysis"])
                    else:
                        oai = make_client(st.session_state.api_key, st.session_state.base_url)
                        result["draft"] = draft_reply(oai, st.session_state.model, result["email"], result["analysis"])
                    st.toast("Draft regenerated")
                except Exception as e:  # noqa: BLE001
                    st.error(friendly_error(e))
                st.rerun()

    status = result["routing"]["approval_status"]
    if status == "pending-human-review":
        st.markdown('<p class="card-label" style="margin-top:10px;">Human review required</p>', unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
        with ac1:
            if st.button("✅ Approve", key=f"{context}_approve_{rid}", width="stretch"):
                result["routing"]["approval_status"] = "approved"
                st.toast("Approved")
                st.rerun()
        with ac2:
            if st.button("❌ Reject", key=f"{context}_reject_{rid}", width="stretch"):
                result["routing"]["approval_status"] = "rejected"
                st.toast("Rejected")
                st.rerun()
    else:
        st.markdown(
            f'<p style="color:var(--text-dim);font-size:12px;margin-top:8px;">'
            f'Current status: <span class="status-chip">{status}</span></p>',
            unsafe_allow_html=True,
        )
        if st.button("↩️ Send back to review", key=f"{context}_repend_{rid}", width="stretch"):
            result["routing"]["approval_status"] = "pending-human-review"
            st.rerun()


def run_batch(emails_list, mock, container):
    client_obj = None if mock else make_client(st.session_state.api_key, st.session_state.base_url)
    prog = container.progress(0.0)
    status_line = container.empty()
    ok, failed = 0, []
    for i, email in enumerate(emails_list):
        status_line.caption(f"Processing {i + 1}/{len(emails_list)} — {email.get('subject', '(no subject)')}")
        result, err = process_full(client_obj, st.session_state.model, email, mock=mock, expanded=False)
        if err:
            failed.append((email.get("subject", "(no subject)"), friendly_error(err)))
        else:
            st.session_state.processed_emails.append(result)
            ok += 1
        prog.progress((i + 1) / len(emails_list))
    status_line.empty()
    if ok:
        container.success(f"✅ Processed {ok} of {len(emails_list)} email(s).")
        container.balloons()
    if failed:
        with container.expander(f"⚠️ {len(failed)} failed"):
            for subj, msg in failed:
                st.write(f"**{subj}** — {msg}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def apply_preset():
    preset = PROVIDER_PRESETS[st.session_state.provider_choice]
    st.session_state.base_url = preset["base_url"]
    if preset["models"]:
        st.session_state.model = preset["models"][0]


def fill_sample(sample):
    st.session_state.compose_from = sample["from"]
    st.session_state.compose_subject = sample["subject"]
    st.session_state.compose_body = sample["body"]


def clear_form():
    st.session_state.compose_from = ""
    st.session_state.compose_subject = ""
    st.session_state.compose_body = ""


with st.sidebar:
    st.markdown("## 🤖 Email Agent")
    st.markdown("---")

    mock_mode = st.toggle(
        "🧪 Demo Mode", key="mock_mode",
        help="Simulate analysis locally with no API calls — great for trying the UI without a key.",
    )
    if mock_mode:
        st.caption("Running on a local rule-based simulator. Turn this off to use a real model.")

    st.selectbox("Provider", list(PROVIDER_PRESETS), key="provider_choice", on_change=apply_preset, disabled=mock_mode)
    st.text_input("Base URL", key="base_url", disabled=mock_mode,
                  help="Change to use Groq, OpenRouter, Together, Ollama, etc.")
    st.text_input("Model", key="model", disabled=mock_mode,
                  help="e.g. gpt-4o-mini, llama-3.3-70b-versatile, mistral-7b-instruct")
    st.text_input("API Key", type="password", key="api_key", placeholder="sk-...", disabled=mock_mode)

    if st.button("🔌 Test connection", width="stretch", disabled=mock_mode):
        try:
            test_client = make_client(st.session_state.api_key, st.session_state.base_url)
            test_client.chat.completions.create(
                model=st.session_state.model, max_tokens=5, messages=[{"role": "user", "content": "ping"}],
            )
            st.session_state.conn_status = ("ok", "✅ Connected — credentials and model look good.")
        except Exception as e:  # noqa: BLE001
            st.session_state.conn_status = ("error", friendly_error(e))
    if st.session_state.conn_status and not mock_mode:
        kind, msg = st.session_state.conn_status
        (st.success if kind == "ok" else st.error)(msg)

    st.markdown("---")
    st.toggle("☀️ Light mode", key="_light_toggle", value=(st.session_state.theme == "light"))
    st.session_state.theme = "light" if st.session_state._light_toggle else "dark"

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("1. 🔍 Intent Detection\n2. ✍️ Reply Drafting\n3. 🔀 Smart Routing\n4. ✅ Approval Flow")

    if st.session_state.processed_emails:
        st.markdown("---")
        _emails = st.session_state.processed_emails
        _auto = sum(1 for e in _emails if e["routing"]["auto_approve"])
        _pending = sum(1 for e in _emails if e["routing"]["approval_status"] == "pending-human-review")
        st.markdown(f"**Processed:** {len(_emails)}  |  **Auto:** {_auto}  |  **Review:** {_pending}")
        with st.popover("🗑️ Clear All", width="stretch"):
            st.write(f"Permanently delete all {len(_emails)} processed email(s)? This can't be undone.")
            if st.button("Yes, clear everything", type="primary", width="stretch", key="confirm_clear"):
                st.session_state.processed_emails = []
                st.session_state.last_compose_result = None
                st.toast("Cleared all processed emails")
                st.rerun()

inject_css(st.session_state.theme)
mock_mode = st.session_state.mock_mode
can_run = mock_mode or bool(st.session_state.api_key)

# ── Header ─────────────────────────────────────────────────────────────────────
_mode_chip = "🧪 Demo Mode — no API key needed" if mock_mode else f"🟢 {st.session_state.provider_choice} · {st.session_state.model}"
st.markdown(f"""
<h1 style='font-family:Syne;font-size:2.4rem;font-weight:800;
    background:linear-gradient(135deg,var(--accent1),var(--accent2));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0;'>
    Autonomous Email Intelligence Agent
</h1>
<p style='color:var(--text-dim);margin-top:4px;'>Reads · Detects · Drafts · Routes · Approves — responsibly.
&nbsp; <span class="status-chip">{_mode_chip}</span></p>
""", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["✉️ Compose", "📦 Batch", "📊 Dashboard"])

# ── Tab 1: Compose & process a single email ──────────────────────────────────
with tab1:
    col_form, col_out = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### Compose / Paste Email")
        sender = st.text_input("From", key="compose_from", placeholder="sender@example.com")
        subject = st.text_input("Subject", key="compose_subject", placeholder="Email subject…")
        body = st.text_area("Body", key="compose_body", height=180, placeholder="Paste email body here…")

        uploaded_txt = st.file_uploader("Or load a .txt file as the body", type=["txt"], key="compose_file")
        if uploaded_txt is not None and st.session_state.compose_file_loaded != uploaded_txt.name:
            st.session_state.compose_body = uploaded_txt.getvalue().decode("utf-8", errors="ignore")
            st.session_state.compose_file_loaded = uploaded_txt.name
            st.rerun()

        st.markdown("**Quick-fill:**")
        qcols = st.columns(4)
        labels = ["Meeting", "Complaint", "Sales", "Spam"]
        for i, (qc, lb) in enumerate(zip(qcols, labels)):
            with qc:
                st.button(lb, width="stretch", key=f"qf{i}", on_click=fill_sample, args=(SAMPLE_EMAILS[i],))

        bcol1, bcol2 = st.columns([3, 1])
        with bcol1:
            go = st.button("🚀 Process Email", type="primary", width="stretch",
                            disabled=not (sender and subject and body and can_run))
        with bcol2:
            st.button("🧹 Clear", width="stretch", on_click=clear_form)

        if not can_run:
            st.caption("Enter an API key in the sidebar, or switch on 🧪 Demo Mode to try this without one.")

    with col_out:
        st.markdown("### Result")
        if go:
            email = {"from": sender, "subject": subject, "body": body}
            client_obj = None if mock_mode else make_client(st.session_state.api_key, st.session_state.base_url)
            result, err = process_full(client_obj, st.session_state.model, email, mock=mock_mode, expanded=True)
            if err:
                st.error(friendly_error(err))
            else:
                st.session_state.processed_emails.append(result)
                st.session_state.last_compose_result = result["id"]

        last_id = st.session_state.last_compose_result
        last_result = next((r for r in st.session_state.processed_emails if r["id"] == last_id), None) if last_id else None
        if last_result:
            render_result_card(last_result)
            render_draft_block(last_result, context="compose")
        elif not go:
            st.info("Process an email to see the analysis, draft reply, and routing decision here.")


# ── Tab 2: Batch processing ───────────────────────────────────────────────────
with tab2:
    st.markdown("### Batch Processing")
    source = st.radio("Source", ["Sample batch (4 emails)", "Upload CSV"], horizontal=True, key="batch_source")

    if source.startswith("Sample"):
        for s in SAMPLE_EMAILS:
            with st.expander(f"📧 {s['subject']} — {s['from']}"):
                st.text(s["body"])
        batch_box = st.container()
        if st.button("⚡ Process All 4 Emails", type="primary", disabled=not can_run, key="batch_run_sample"):
            run_batch(SAMPLE_EMAILS, mock_mode, batch_box)
        if not can_run:
            st.caption("Enter an API key in the sidebar, or switch on 🧪 Demo Mode to try this without one.")
    else:
        st.caption("Columns needed: **from**, **subject**, **body** (aliases like sender, title, message also work).")
        st.download_button("⬇️ Download CSV template", data=CSV_TEMPLATE, file_name="emails_template.csv", mime="text/csv")
        up = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
        if up is not None:
            try:
                parsed = parse_email_csv(up)
            except ValueError as ve:
                st.error(str(ve))
                parsed = []
            if parsed:
                st.write(f"Parsed **{len(parsed)}** email(s):")
                st.dataframe(parsed, width="stretch", height=220)
                batch_box2 = st.container()
                if st.button(f"⚡ Process {len(parsed)} Email(s)", type="primary", disabled=not can_run, key="batch_run_csv"):
                    run_batch(parsed, mock_mode, batch_box2)
                if not can_run:
                    st.caption("Enter an API key in the sidebar, or switch on 🧪 Demo Mode to try this without one.")
            elif up is not None:
                st.warning("No usable rows found in that file.")


# ── Tab 3: Dashboard ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Dashboard")
    emails = st.session_state.processed_emails
    if not emails:
        st.info("No emails processed yet — head to **✉️ Compose** or **📦 Batch** to get started.")
    else:
        auto = sum(1 for e in emails if e["routing"]["auto_approve"])
        pending = sum(1 for e in emails if e["routing"]["approval_status"] == "pending-human-review")
        urgent = sum(1 for e in emails if e["routing"]["priority"] in ("urgent", "high"))
        avg_conf = sum(e["analysis"].get("confidence", 0) for e in emails) / len(emails)

        kpis = [
            ("Processed", len(emails), "var(--accent1)"),
            ("Auto-Approved", auto, "#4ade80"),
            ("Needs Review", pending, "#fbbf24"),
            ("High/Urgent", urgent, "#f87171"),
            ("Avg. Confidence", f"{avg_conf:.0%}", "var(--accent2)"),
        ]
        kcols = st.columns(5)
        for c, (label, val, color) in zip(kcols, kpis):
            with c:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:{color}">{val}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            st.caption("Intent distribution")
            st.bar_chart(counts_for(emails, lambda r: r["analysis"].get("intent", "information_request"), INTENT_CATEGORIES.keys()), height=180)
        with ch2:
            st.caption("Sentiment")
            st.bar_chart(counts_for(emails, lambda r: r["analysis"].get("sentiment", "neutral"), ["positive", "neutral", "negative", "urgent"]), height=180)
        with ch3:
            st.caption("Priority")
            st.bar_chart(counts_for(emails, lambda r: r["routing"].get("priority", "none"), ["urgent", "high", "medium", "low", "none"]), height=180)

        st.markdown("---")
        with st.expander("🔎 Filter & search", expanded=False):
            fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
            with fc1:
                q = st.text_input("Search", placeholder="subject, sender, or summary…", label_visibility="collapsed")
            with fc2:
                f_intent = st.multiselect("Intent", list(INTENT_CATEGORIES), placeholder="Intent", label_visibility="collapsed")
            with fc3:
                f_status = st.multiselect("Status", ["auto-approved", "pending-human-review", "approved", "rejected"],
                                           placeholder="Status", label_visibility="collapsed")
            with fc4:
                sort_by = st.selectbox("Sort", ["Newest first", "Oldest first", "Priority"], label_visibility="collapsed")

        filtered = list(emails)
        if q:
            ql = q.lower()
            filtered = [r for r in filtered if ql in r["email"].get("subject", "").lower()
                        or ql in r["email"].get("from", "").lower()
                        or ql in r["analysis"].get("summary", "").lower()]
        if f_intent:
            filtered = [r for r in filtered if r["analysis"].get("intent") in f_intent]
        if f_status:
            filtered = [r for r in filtered if r["routing"].get("approval_status") in f_status]

        if sort_by == "Priority":
            _order = {"urgent": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
            filtered = sorted(filtered, key=lambda r: _order.get(r["routing"].get("priority", "none"), 9))
        elif sort_by == "Newest first":
            filtered = list(reversed(filtered))

        pending_items = [r for r in emails if r["routing"]["approval_status"] == "pending-human-review"]
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            if st.button(f"✅ Approve all pending ({len(pending_items)})", disabled=not pending_items, width="stretch"):
                for r in pending_items:
                    r["routing"]["approval_status"] = "approved"
                st.toast(f"Approved {len(pending_items)} email(s)")
                st.rerun()
        with bc2:
            if st.button(f"❌ Reject all pending ({len(pending_items)})", disabled=not pending_items, width="stretch"):
                for r in pending_items:
                    r["routing"]["approval_status"] = "rejected"
                st.toast(f"Rejected {len(pending_items)} email(s)")
                st.rerun()
        with bc3:
            st.download_button("⬇️ Export JSON", data=export_json_bytes(emails), file_name="processed_emails.json",
                                mime="application/json", width="stretch")
        with bc4:
            st.download_button("⬇️ Export CSV", data=export_csv_bytes(emails), file_name="processed_emails.csv",
                                mime="text/csv", width="stretch")

        st.caption(f"Showing {len(filtered)} of {len(emails)} email(s)")
        status_icon = {"pending-human-review": "🟡", "auto-approved": "🟢", "approved": "✅", "rejected": "❌"}
        for result in filtered:
            email = result["email"]
            r = result["routing"]
            icon = status_icon.get(r["approval_status"], "⚪")
            with st.expander(f"{icon} {email.get('subject', '(no subject)')} — {email.get('from', '')}  ·  {result.get('timestamp', '')}"):
                cda, cdb = st.columns([1, 1], gap="large")
                with cda:
                    render_result_card(result)
                with cdb:
                    render_draft_block(result, context="dash")
                    rerun_disabled = not can_run
                    if st.button("🔁 Re-run analysis", key=f"dash_rerun_{result['id']}", width="stretch", disabled=rerun_disabled):
                        cobj = None if mock_mode else make_client(st.session_state.api_key, st.session_state.base_url)
                        new_result, err = process_full(cobj, st.session_state.model, email, mock=mock_mode, expanded=False)
                        if err:
                            st.error(friendly_error(err))
                        else:
                            result["analysis"] = new_result["analysis"]
                            result["draft"] = new_result["draft"]
                            result["routing"] = new_result["routing"]
                            result["mock"] = mock_mode
                            st.toast("Re-analyzed")
                        st.rerun()
