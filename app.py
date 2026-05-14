import streamlit as st
from openai import OpenAI
import json
from datetime import datetime

st.set_page_config(
    page_title="Email Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main, .stApp { background: #0a0a0f; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.email-card {
    background: #111118; border: 1px solid #1e1e2e;
    border-radius: 12px; padding: 20px; margin: 10px 0;
}
.intent-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.badge-meeting_request     { background:#1e3a5f; color:#60a5fa; }
.badge-support_request     { background:#3b1f1f; color:#f87171; }
.badge-sales_inquiry       { background:#1a3a1a; color:#4ade80; }
.badge-complaint           { background:#3b1f1f; color:#fb923c; }
.badge-follow_up           { background:#2a1f3b; color:#c084fc; }
.badge-information_request { background:#1f2e3b; color:#38bdf8; }
.badge-urgent              { background:#3b1f1f; color:#ef4444; }
.badge-spam                { background:#1f1f1f; color:#6b7280; }
.priority-urgent  { color:#ef4444; }
.priority-high    { color:#f97316; }
.priority-medium  { color:#eab308; }
.priority-low     { color:#22c55e; }
.priority-none    { color:#6b7280; }
.status-auto-approved        { color:#4ade80; }
.status-pending-human-review { color:#fbbf24; }
.status-approved             { color:#34d399; }
.status-rejected             { color:#f87171; }
.draft-box {
    background:#0d1117; border:1px solid #21262d; border-left:3px solid #4f46e5;
    border-radius:8px; padding:16px; font-family:'JetBrains Mono',monospace;
    font-size:13px; color:#c9d1d9; white-space:pre-wrap;
}
.metric-card { background:#111118; border:1px solid #1e1e2e; border-radius:12px; padding:20px; text-align:center; }
.metric-value { font-size:36px; font-weight:800; color:#818cf8; }
.metric-label { font-size:12px; color:#6b7280; text-transform:uppercase; letter-spacing:0.1em; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
INTENT_CATEGORIES = {
    "meeting_request":     "Schedule or reschedule a meeting",
    "support_request":     "Technical or customer support needed",
    "sales_inquiry":       "Potential sales lead or product question",
    "complaint":           "Customer complaint requiring escalation",
    "follow_up":           "Follow-up on previous conversation",
    "information_request": "Requesting information or documentation",
    "urgent":              "Time-sensitive matter",
    "spam":                "Unsolicited or irrelevant email",
}
ROUTING_RULES = {
    "meeting_request":     {"team":"📅 Calendar/Admin",   "auto_approve":True,  "priority":"medium"},
    "support_request":     {"team":"🛠️ Support Team",     "auto_approve":False, "priority":"high"},
    "sales_inquiry":       {"team":"💼 Sales Team",       "auto_approve":True,  "priority":"medium"},
    "complaint":           {"team":"🚨 Customer Success", "auto_approve":False, "priority":"urgent"},
    "follow_up":           {"team":"🔁 Original Handler", "auto_approve":True,  "priority":"low"},
    "information_request": {"team":"📋 General",          "auto_approve":True,  "priority":"low"},
    "urgent":              {"team":"⚡ Management",        "auto_approve":False, "priority":"urgent"},
    "spam":                {"team":"🗑️ None",              "auto_approve":True,  "priority":"none"},
}
SAMPLE_EMAILS = [
    {"from":"sarah.chen@techcorp.com","subject":"Quick sync this week?",
     "body":"Hi, I'd love to schedule a 30-minute call to discuss the Q2 roadmap alignment. Are you available Thursday or Friday afternoon?"},
    {"from":"angry.customer@gmail.com","subject":"STILL no resolution after 2 weeks!!!",
     "body":"This is absolutely unacceptable. I've been waiting 2 weeks for a refund on order #78432 and nobody has gotten back to me. I will be disputing this charge with my bank if this isn't resolved TODAY."},
    {"from":"john.smith@startup.io","subject":"Interested in your enterprise plan",
     "body":"Hello, we're a 50-person startup evaluating tools for our engineering team. Could you send over pricing for your enterprise tier? We're also curious about SSO and SLA guarantees."},
    {"from":"noreply@newsletters.biz","subject":"🔥 HOT DEALS This Week Only!!!",
     "body":"You've been selected for our EXCLUSIVE offer! Click here to claim your FREE gift. Limited time only."},
]

if "processed_emails" not in st.session_state:
    st.session_state.processed_emails = []


# ── AI helpers ─────────────────────────────────────────────────────────────────
def make_client(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)

def _chat(client, model, system, user, max_tokens=600):
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
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
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.split("```")[0]
    return json.loads(raw.strip())

def draft_reply(client, model, email, analysis):
    routing = ROUTING_RULES.get(analysis["intent"], {})
    system = "You are a professional email assistant. Write concise, warm, human-sounding replies."
    user = f"""Draft a reply. Rules: under 150 words, warm & professional, sign as "The Team", no placeholders.
If spam: one polite unsubscribe-acknowledged line only.
EMAIL — From: {email['from']} | Subject: {email['subject']}
Body: {email['body']}
CONTEXT — Intent: {analysis['intent']} | Sentiment: {analysis['sentiment']} | Team: {routing.get('team','General')}"""
    return _chat(client, model, system, user, 400)

def route_email(analysis):
    intent = analysis.get("intent", "information_request")
    routing = dict(ROUTING_RULES.get(intent, ROUTING_RULES["information_request"]))
    if analysis.get("requires_human_approval"):
        routing["auto_approve"] = False
    routing["approval_status"] = "auto-approved" if routing["auto_approve"] else "pending-human-review"
    return routing

def process_full(client, model, email, log_ph):
    result = {"email": email, "timestamp": datetime.now().strftime("%H:%M:%S")}
    with log_ph:
        st.markdown("🔍 Analyzing intent…")
    result["analysis"] = analyze_email(client, model, email)
    with log_ph:
        st.markdown("✍️ Drafting reply…")
    result["draft"] = draft_reply(client, model, email, result["analysis"])
    with log_ph:
        st.markdown("🔀 Routing & queueing…")
    result["routing"] = route_email(result["analysis"])
    with log_ph:
        st.success("✅ Done!")
    return result


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Email Agent")
    st.markdown("---")

    api_key  = st.text_input("API Key", type="password", placeholder="sk-...")
    base_url = st.text_input("Base URL", value="https://api.openai.com/v1",
                              help="Change to use Groq, OpenRouter, Ollama, etc.")
    model    = st.text_input("Model", value="gpt-4o-mini",
                              help="e.g. gpt-4o, llama3-70b-8192, mistral-7b-instruct")

    st.markdown("---")
    st.markdown("**Provider examples**")
    st.code("""OpenAI
  https://api.openai.com/v1
  gpt-4o-mini

Groq (free tier)
  https://api.groq.com/openai/v1
  llama3-70b-8192

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("1. 🔍 Intent Detection\n2. ✍️ Reply Drafting\n3. 🔀 Smart Routing\n4. ✅ Approval Flow")

    if st.session_state.processed_emails:
        st.markdown("---")
        emails = st.session_state.processed_emails
        auto    = sum(1 for e in emails if e["routing"]["auto_approve"])
        pending = len(emails) - auto
        st.markdown(f"**Processed:** {len(emails)}  |  **Auto:** {auto}  |  **Review:** {pending}")
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.processed_emails = []
            st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Syne;font-size:2.5rem;font-weight:800;
    background:linear-gradient(135deg,#818cf8,#c084fc);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0;'>
    Autonomous Email Intelligence Agent
</h1>
<p style='color:#6b7280;margin-top:4px;'>Reads · Detects · Drafts · Routes · Approves — responsibly.</p>
""", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📧 Process Email", "📦 Demo Batch", "📊 Dashboard"])

# ── Tab 1: Process single email ────────────────────────────────────────────────
with tab1:
    col_form, col_out = st.columns([1,1], gap="large")

    with col_form:
        st.markdown("### Compose / Paste Email")
        sender  = st.text_input("From",    placeholder="sender@example.com")
        subject = st.text_input("Subject", placeholder="Email subject…")
        body    = st.text_area("Body",     height=180, placeholder="Paste email body here…")

        st.markdown("**Quick-fill:**")
        qcols = st.columns(4)
        labels = ["Meeting","Complaint","Sales","Spam"]
        for i,(qc,lb) in enumerate(zip(qcols,labels)):
            with qc:
                if st.button(lb, use_container_width=True, key=f"qf{i}"):
                    s = SAMPLE_EMAILS[i]
                    st.session_state["_f"] = s["from"]
                    st.session_state["_s"] = s["subject"]
                    st.session_state["_b"] = s["body"]
                    st.rerun()
        if "_f" in st.session_state:
            sender  = st.session_state.pop("_f")
            subject = st.session_state.pop("_s")
            body    = st.session_state.pop("_b")

        go = st.button("🚀 Process Email", type="primary", use_container_width=True,
                        disabled=not (sender and subject and body and api_key))

    with col_out:
        st.markdown("### Result")
        log_ph  = st.empty()
        res_ph  = st.empty()

    if go:
        if not api_key:
            st.error("Enter your API key in the sidebar.")
        else:
            email = {"from": sender, "subject": subject, "body": body}
            try:
                oai = make_client(api_key, base_url)
                result = process_full(oai, model, email, log_ph)
                st.session_state.processed_emails.append(result)
                a = result["analysis"]
                r = result["routing"]
                intent = a.get("intent","unknown")
                with res_ph.container():
                    st.markdown(f"""
                    <div class="email-card">
                        <span class="intent-badge badge-{intent}">{intent.replace('_',' ')}</span>
                        <p style='color:#9ca3af;margin:10px 0 4px'>Sentiment: <b style='color:#e5e7eb'>{a['sentiment']}</b></p>
                        <p style='color:#9ca3af;margin:4px 0'>Confidence: <b style='color:#e5e7eb'>{a['confidence']:.0%}</b></p>
                        <p style='color:#9ca3af;margin:4px 0'>Routes to: <b style='color:#e5e7eb'>{r['team']}</b></p>
                        <p style='color:#9ca3af;margin:4px 0'>Status: <span class="status-{r['approval_status'].replace(' ','-')}">{r['approval_status']}</span></p>
                        <hr style='border-color:#1e1e2e;margin:12px 0'>
                        <p style='color:#6b7280;font-size:12px;margin-bottom:4px'>SUMMARY</p>
                        <p style='color:#c9d1d9;font-size:14px'>{a['summary']}</p>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("**Draft Reply:**")
                    st.markdown(f'<div class="draft-box">{result["draft"]}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")


# ── Tab 2: Batch ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Run Demo Batch — 4 sample emails")
    for s in SAMPLE_EMAILS:
        with st.expander(f"📧 {s['subject']} — {s['from']}"):
            st.text(s["body"])

    if st.button("⚡ Process All 4 Emails", type="primary", disabled=not api_key):
        if not api_key:
            st.error("Add API key in the sidebar.")
        else:
            oai  = make_client(api_key, base_url)
            prog = st.progress(0)
            for i, email in enumerate(SAMPLE_EMAILS):
                msg = st.empty()
                msg.info(f"Processing {i+1}/4: {email['subject']}")
                try:
                    lph = st.empty()
                    result = process_full(oai, model, email, lph)
                    st.session_state.processed_emails.append(result)
                except Exception as e:
                    st.error(f"Failed: {e}")
                prog.progress((i+1)/4)
            st.success("✅ Batch complete!")
            st.balloons()


# ── Tab 3: Dashboard ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Dashboard")
    emails = st.session_state.processed_emails
    if not emails:
        st.info("No emails processed yet.")
    else:
        auto    = sum(1 for e in emails if e["routing"]["auto_approve"])
        pending = sum(1 for e in emails if e["routing"]["approval_status"]=="pending-human-review")
        urgent  = sum(1 for e in emails if e["routing"]["priority"] in ["urgent","high"])

        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(emails)}</div><div class="metric-label">Processed</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#4ade80">{auto}</div><div class="metric-label">Auto-Approved</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#fbbf24">{pending}</div><div class="metric-label">Needs Review</div></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f87171">{urgent}</div><div class="metric-label">High Priority</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        for idx, result in enumerate(reversed(emails)):
            email   = result["email"]
            a       = result["analysis"]
            r       = result["routing"]
            intent  = a.get("intent","unknown")
            with st.expander(f"📧 {email['subject']} — {email['from']}  ·  {result.get('timestamp','')}"):
                ca, cb = st.columns([1,1])
                with ca:
                    st.markdown(f"""
                    <div class="email-card">
                        <span class="intent-badge badge-{intent}">{intent.replace('_',' ')}</span>
                        <p style='color:#9ca3af;margin:8px 0 4px'>Sentiment: <b style='color:#e5e7eb'>{a['sentiment']}</b></p>
                        <p style='color:#9ca3af;margin:4px 0'>Confidence: <b style='color:#e5e7eb'>{a['confidence']:.0%}</b></p>
                        <p style='color:#9ca3af;margin:4px 0'>Team: <b style='color:#e5e7eb'>{r['team']}</b></p>
                        <p style='color:#9ca3af;margin:4px 0'>Priority: <span class="priority-{r['priority']}">{r['priority'].upper()}</span></p>
                        <p style='color:#9ca3af;margin:4px 0'>Status: <span class="status-{r['approval_status'].replace(' ','-')}">{r['approval_status']}</span></p>
                        <hr style='border-color:#1e1e2e;margin:10px 0'>
                        <p style='color:#c9d1d9;font-size:13px'>{a['summary']}</p>
                    </div>""", unsafe_allow_html=True)
                with cb:
                    st.markdown("**Draft Reply:**")
                    st.markdown(f'<div class="draft-box">{result["draft"]}</div>', unsafe_allow_html=True)
                    if r["approval_status"] == "pending-human-review":
                        st.markdown("**Human Review Required:**")
                        ca2, cb2 = st.columns(2)
                        with ca2:
                            if st.button("✅ Approve", key=f"ap_{idx}", use_container_width=True):
                                result["routing"]["approval_status"] = "approved"
                                st.rerun()
                        with cb2:
                            if st.button("❌ Reject", key=f"rj_{idx}", use_container_width=True):
                                result["routing"]["approval_status"] = "rejected"
                                st.rerun()
