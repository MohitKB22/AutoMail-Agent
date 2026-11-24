"""
Autonomous Email Intelligence Agent — Streamlit Dashboard
Works with ANY OpenAI-compatible provider (OpenAI, Groq, OpenRouter, Together, Ollama, ...).

The analysis / drafting / routing logic and the Demo Mode simulator live in
agent.py and are shared with the CLI and the FastAPI backend.
"""

import csv
import html
import io
import json
import re
import uuid
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
from openai import OpenAI

from agent import (
    INTENT_CATEGORIES,
    SAMPLE_EMAILS,
    SENTIMENTS,
    analyze_email,
    draft_reply,
    mock_analyze,
    mock_draft_reply,
    normalize_intent,
    route_email,
)

st.set_page_config(
    page_title="Email Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ────────────────────────────────────────────────────────────
# The dark palette mirrors .streamlit/config.toml (Streamlit's native theme), so
# native widgets and the custom HTML cards always match. Light mode re-skins the
# native widgets from these tokens via CSS (see LIGHT_OVERRIDES below).
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

# Streamlit's own theme can't be switched per session from Python, so in light
# mode we override the native widgets (which follow the dark base theme).
LIGHT_OVERRIDES = """
.stApp, section[data-testid="stSidebar"] { color: var(--text); }
header[data-testid="stHeader"] { background: var(--bg) !important; }
header[data-testid="stHeader"] :is(button, a, span, svg) { color: var(--text-muted) !important; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background: var(--surface2) !important; }
.stApp :is(h1, h2, h3, h4, h5, h6):not([style]) { color: var(--text) !important; }
.stApp :is([data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-testid="stText"],
           [data-testid="stCheckbox"], [data-testid="stRadio"], label) { color: var(--text); }
.stApp :is([data-testid="stWidgetLabel"], label) :is(p, span, div) { color: var(--text); }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color: var(--text-muted) !important; }
.stApp [data-testid="stTooltipIcon"] svg, .stApp [data-testid="stTooltipHoverTarget"] svg { color: var(--text-dim) !important; stroke: var(--text-dim); }
.stApp [data-baseweb="tab-list"] [role="tab"]:not([aria-selected="true"]) p { color: var(--text-muted) !important; }
.stApp [data-baseweb="tab-border"] { background: var(--border) !important; }
.stApp :is([data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div,
           [data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
           [data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"]) {
    background-color: var(--surface) !important; border-color: var(--border) !important;
}
.stApp :is(input, textarea) { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; caret-color: var(--text); }
.stApp :is(input, textarea)::placeholder { color: var(--text-dim) !important; -webkit-text-fill-color: var(--text-dim) !important; }
.stApp :is(input, textarea):disabled { color: var(--text-dim) !important; -webkit-text-fill-color: var(--text-dim) !important; }
.stApp [data-baseweb="select"] :is(div, span, svg) { color: var(--text) !important; }
.stApp [data-baseweb="tag"] { background: var(--surface2) !important; }
.stApp :is(input, textarea) ~ div button svg, .stApp [data-testid="stNumberInput"] button { color: var(--text-muted) !important; }
.stApp :is(button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-tertiary"],
           [data-testid="stPopover"] button, [data-testid="stDownloadButton"] button,
           [data-testid="stFileUploaderDropzone"] button) {
    background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important;
}
.stApp :is(button[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button):disabled { color: var(--text-dim) !important; opacity: .7; }
.stApp [data-testid="stFileUploaderDropzone"] { background: var(--surface) !important; border: 1px dashed var(--border) !important; }
.stApp [data-testid="stFileUploaderDropzone"] *, .stApp [data-testid="stFileUploaderFile"] * { color: var(--text-muted) !important; }
.stApp [data-testid="stFileChip"] { background: var(--surface2) !important; border: 1px solid var(--border); }
.stApp [data-testid="stFileChip"] > div:first-child { background: var(--border) !important; }
.stApp [data-testid="stRadioOption"]:not([data-selected="true"]) > div:first-child,
.stApp [data-testid="stRadioOption"]:not([data-selected="true"]) > * > div:first-child { background: var(--surface) !important; }
.stApp [data-testid="stExpander"] details { background: var(--surface) !important; border-color: var(--border) !important; }
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * { color: var(--text) !important; }
.stApp [data-testid="stExpander"] summary { background: var(--surface) !important; }
.stApp [data-testid="stExpander"] summary:hover { background: var(--surface2) !important; }
.stApp :is([data-testid="stCode"], [data-testid="stCodeBlock"]) :is(pre, code) { background: var(--surface2) !important; color: var(--code-text) !important; }
.stApp [data-testid="stAlertContainer"] :is(p, div, span) { color: var(--text) !important; }
.stApp [data-testid="stProgress"] > div > div { background: var(--border) !important; }
.stApp [data-testid="stElementToolbarButtonContainer"] { background: var(--surface) !important; }
.stApp [data-testid="stElementToolbarButtonContainer"] :is(button, svg) { color: var(--text-muted) !important; }
.stApp hr { border-color: var(--border) !important; background: var(--border) !important; }
div[data-baseweb="popover"] :is(ul, [role="listbox"], [data-testid="stPopoverBody"]),
div[data-testid="stPopoverBody"], [role="listbox"] { background: var(--surface) !important; color: var(--text) !important; }
div[data-baseweb="popover"] :is(li, li *, p, span), [role="listbox"] [role="option"], [role="listbox"] [role="option"] *,
div[data-testid="stPopoverBody"] :is(p, span) { color: var(--text) !important; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"],
[role="listbox"] [role="option"]:is(:hover, [data-focused], [aria-selected="true"]) { background: var(--surface2) !important; }
"""


def inject_css(theme_name: str) -> None:
    t = THEMES.get(theme_name, THEMES["dark"])
    css_vars = "\n".join(f"  --{k.replace('_', '-')}: {v};" for k, v in t.items())
    overrides = LIGHT_OVERRIDES if theme_name == "light" else ""
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
{overrides}
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
.stApp .card-row {{ color:var(--text-muted); margin:8px 0 0; font-size:13px; display:flex; justify-content:space-between; gap:8px; }}
.stApp .card-row b, .stApp .card-row .card-val {{ color:var(--text); }}
.card-hr {{ border:none; border-top:1px solid var(--border); margin:12px 0; }}
.stApp .card-label {{ color:var(--text-dim); font-size:11px; letter-spacing:.08em; margin-bottom:4px; text-transform:uppercase; }}
.stApp .card-summary {{ color:var(--code-text); font-size:14px; margin:0; line-height:1.5; overflow-wrap:anywhere; }}
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

.stApp .priority-urgent {{ color:#ef4444; font-weight:600; }}
.stApp .priority-high {{ color:#f97316; font-weight:600; }}
.stApp .priority-medium {{ color:#ca8a04; font-weight:600; }}
.stApp .priority-low {{ color:#16a34a; font-weight:600; }}
.stApp .priority-none {{ color:#6b7280; font-weight:600; }}

.stApp .status-auto-approved {{ color:#22c55e; font-weight:600; }}
.stApp .status-pending-human-review {{ color:#f59e0b; font-weight:600; }}
.stApp .status-approved {{ color:#10b981; font-weight:600; }}
.stApp .status-rejected {{ color:#ef4444; font-weight:600; }}

.metric-card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:18px; text-align:center; }}
.metric-value {{ font-size:30px; font-weight:800; }}
.stApp .metric-label {{ font-size:11px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.1em; margin-top:2px; }}

.status-chip {{
    display:inline-block; padding:4px 12px; border-radius:8px; font-size:12px;
    font-family:'JetBrains Mono',monospace; background:var(--surface2); border:1px solid var(--border2); color:var(--text-muted);
}}

div[data-testid="stCodeBlock"] pre {{ font-size: 13px !important; }}
.stApp button[kind="primary"], .stApp button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
    border: none !important; color: #ffffff !important;
}}
.stApp button[data-testid="stBaseButton-primary"]:disabled {{ opacity: .45; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
</style>
""",
        unsafe_allow_html=True,
    )


# ── Domain constants ─────────────────────────────────────────────────────────
# Same routing table as agent.ROUTING_RULES, with icons for display.
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
PRIORITIES = ["urgent", "high", "medium", "low", "none"]
STATUSES = ["auto-approved", "pending-human-review", "approved", "rejected"]
PROVIDER_PRESETS = {
    "OpenAI": {"base_url": "https://api.openai.com/v1", "models": ["gpt-4o-mini", "gpt-4o"]},
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]},
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "models": ["mistralai/mistral-7b-instruct"]},
    "Together AI": {"base_url": "https://api.together.xyz/v1", "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"]},
    "Ollama (local)": {"base_url": "http://localhost:11434/v1", "models": ["llama3"]},
    "Custom": {"base_url": "", "models": []},
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
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Pipeline orchestration ───────────────────────────────────────────────────
def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key or "none", base_url=base_url)


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if isinstance(e, json.JSONDecodeError) or "json" in low:
        return "🧩 The model didn't return valid JSON — try a different or larger model."
    if "401" in msg or "authentic" in low or "api key" in low or "api_key" in low:
        return "🔑 Authentication failed — double-check your API key in the sidebar."
    if "404" in msg or "model_not_found" in low or "does not exist" in low:
        return "🤖 Model not found — check the model name for this provider."
    if "429" in msg or "rate limit" in low:
        return "⏳ Rate limited by the provider — wait a moment and try again."
    if "connect" in low or "timed out" in low or "timeout" in low or "name resolution" in low:
        return "🌐 Couldn't reach the provider — check the Base URL and your network."
    return f"⚠️ {msg}"


def process_full(client, model, email, mock=False, expanded=True):
    result = {"id": uuid.uuid4().hex[:8], "email": email, "timestamp": datetime.now().strftime("%H:%M:%S")}
    with st.status("Processing email…", expanded=expanded) as status:
        try:
            st.write("🔍 Analyzing intent & sentiment…")
            analysis = mock_analyze(email) if mock else analyze_email(email, client=client, model=model)
            result["analysis"] = analysis
            st.write(f"→ **{analysis['intent'].replace('_', ' ')}** · {analysis['sentiment']} · {analysis['confidence']:.0%} confidence")

            st.write("✍️ Drafting reply…")
            draft = mock_draft_reply(email, analysis) if mock else draft_reply(email, analysis, client=client, model=model)
            result["draft"] = draft
            st.write(f"→ {len(draft.split())} words drafted")

            st.write("🔀 Routing & queueing…")
            routing = route_email(analysis, rules=ROUTING_RULES)
            result["routing"] = routing
            st.write(f"→ {routing['team']} · priority **{routing['priority']}** · {routing['approval_status']}")

            status.update(label="✅ Done", state="complete")
            result["mock"] = mock
            return result, None
        except Exception as e:  # noqa: BLE001
            status.update(label="❌ Failed", state="error")
            return None, e


def parse_email_csv(uploaded_file):
    # utf-8-sig strips the BOM Excel adds to "CSV UTF-8" exports (otherwise the
    # first header reads "﻿from" and the "from" column is never found).
    text = uploaded_file.getvalue().decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("Couldn't read any columns from this CSV.")
    cols = {c.lower().strip(): c for c in reader.fieldnames if c}

    def find(*aliases):
        for a in aliases:
            if a in cols:
                return cols[a]
        return None

    f_from, f_subj, f_body = find("from", "sender", "email"), find("subject", "title"), find("body", "message", "content", "text")
    missing = [n for n, v in [("from", f_from), ("subject", f_subj), ("body", f_body)] if v is None]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Found: {', '.join(c for c in reader.fieldnames if c)}")
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
def _txt(value) -> str:
    """HTML-escape untrusted text (email content / model output) and keep it on one line.

    A blank line inside st.markdown HTML ends the HTML block and the rest of the
    card would render as a raw-HTML code block, so newlines are collapsed.
    """
    return html.escape(" ".join(str(value if value is not None else "").split()))


def render_result_card(result):
    a, r = result["analysis"], result["routing"]
    intent = normalize_intent(a.get("intent"))
    priority = r.get("priority") if r.get("priority") in PRIORITIES else "none"
    status = r.get("approval_status") or "—"
    status_cls = re.sub(r"[^a-z-]", "", status.lower())
    sim_tag = ' <span class="sim-tag">SIMULATED</span>' if result.get("mock") else ""
    try:
        confidence = f"{float(a.get('confidence', 0)):.0%}"
    except (TypeError, ValueError):
        confidence = "—"

    ents = a.get("key_entities") or []
    entities_html = ""
    if ents:
        chips = "".join(f'<span class="entity-chip">{_txt(e)}</span>' for e in ents[:6])
        entities_html = f'<div class="entity-row">{chips}</div>'

    # Built as a single line on purpose (see _txt).
    card = "".join([
        f'<div class="email-card priority-border-{priority}">',
        f'<div class="card-top"><span class="intent-badge badge-{intent}">{_txt(intent.replace("_", " "))}</span>{sim_tag}</div>',
        f'<p class="card-row">Sentiment <b>{_txt(a.get("sentiment", "—"))}</b></p>',
        f'<p class="card-row">Confidence <b>{confidence}</b></p>',
        f'<p class="card-row">Routes to <b>{_txt(r.get("team", "—"))}</b></p>',
        f'<p class="card-row">Priority <span class="priority-{priority}">{priority.upper()}</span></p>',
        f'<p class="card-row">Status <span class="status-{status_cls}">{_txt(status)}</span></p>',
        entities_html,
        '<hr class="card-hr">',
        '<p class="card-label">Summary</p>',
        f'<p class="card-summary">{_txt(a.get("summary", "—"))}</p>',
        '</div>',
    ])
    st.markdown(card, unsafe_allow_html=True)


def render_draft_block(result, context):
    rid = result["id"]
    st.markdown('<p class="card-label" style="margin-top:14px;">Draft reply</p>', unsafe_allow_html=True)
    st.code(result.get("draft", ""), language=None, wrap_lines=True)

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
                        result["draft"] = draft_reply(result["email"], result["analysis"],
                                                      client=oai, model=st.session_state.model)
                except Exception as e:  # noqa: BLE001
                    # Don't rerun here, or the error disappears before anyone can read it.
                    st.error(friendly_error(e))
                else:
                    st.session_state.pop(f"{context}_ta_{rid}", None)
                    st.toast("Draft regenerated")
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
            f'Current status: <span class="status-chip">{_txt(status)}</span></p>',
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
        container.success(f"✅ Processed {ok} of {len(emails_list)} email(s). See them in **📊 Dashboard**.")
        container.balloons()
    if failed:
        with container.expander(f"⚠️ {len(failed)} failed"):
            for subj, msg in failed:
                st.write(f"**{subj}** — {msg}")


def bar_chart(counts: dict, color: str):
    """Altair bar chart styled from the active theme tokens (st.bar_chart follows
    Streamlit's base theme only, so its axes were unreadable in light mode)."""
    t = THEMES[st.session_state.theme]
    df = pd.DataFrame({"label": [str(k).replace("_", " ") for k in counts], "count": list(counts.values())})
    chart = (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=-35, labelLimit=110)),
            y=alt.Y("count:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d")),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("count:Q", title="emails")],
        )
        .properties(height=200, width="container", background="transparent")
        .configure_axis(labelColor=t["text_muted"], gridColor=t["border"], domainColor=t["border"],
                        tickColor=t["border"], labelFont="sans-serif", labelFontSize=10)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, theme=None)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def apply_preset():
    preset = PROVIDER_PRESETS[st.session_state.provider_choice]
    st.session_state.base_url = preset["base_url"]
    if preset["models"]:
        st.session_state.model = preset["models"][0]
    st.session_state.conn_status = None


def fill_sample(sample):
    st.session_state.compose_from = sample["from"]
    st.session_state.compose_subject = sample["subject"]
    st.session_state.compose_body = sample["body"]


def clear_form():
    st.session_state.compose_from = ""
    st.session_state.compose_subject = ""
    st.session_state.compose_body = ""


def load_txt_body():
    # Runs as an on_change callback, i.e. before the Body widget is created — assigning
    # to its key after it's instantiated raises StreamlitAPIException.
    uploaded = st.session_state.get("compose_file")
    if uploaded is not None:
        st.session_state.compose_body = uploaded.getvalue().decode("utf-8-sig", errors="ignore")


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
    st.text_input("API Key", type="password", key="api_key", placeholder="sk-...", disabled=mock_mode,
                  help="Kept only in this browser session's memory — never stored or logged.")

    if st.button("🔌 Test connection", width="stretch", disabled=mock_mode):
        try:
            test_client = make_client(st.session_state.api_key, st.session_state.base_url)
            ping = [{"role": "user", "content": "ping"}]
            try:
                test_client.chat.completions.create(model=st.session_state.model, max_tokens=5, messages=ping)
            except Exception as e:  # noqa: BLE001
                if "max_tokens" not in str(e):
                    raise
                # Reasoning models (o-series, gpt-5…) only accept max_completion_tokens.
                test_client.chat.completions.create(model=st.session_state.model, max_completion_tokens=16, messages=ping)
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
        _auto = sum(1 for e in _emails if e["routing"]["approval_status"] == "auto-approved")
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
_mode_chip = ("🧪 Demo Mode — no API key needed" if mock_mode
              else f"🟢 {st.session_state.provider_choice} · {st.session_state.model}")
st.markdown(f"""
<h1 style='font-family:Syne;font-size:2.4rem;font-weight:800;
    background:linear-gradient(135deg,var(--accent1),var(--accent2));
    -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0;'>
    Autonomous Email Intelligence Agent
</h1>
<p style='color:var(--text-dim);margin-top:4px;'>Reads · Detects · Drafts · Routes · Approves — responsibly.
&nbsp; <span class="status-chip">{_txt(_mode_chip)}</span></p>
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

        st.file_uploader("Or load a .txt file as the body", type=["txt"], key="compose_file", on_change=load_txt_body)

        st.markdown("**Quick-fill:**")
        qcols = st.columns(4)
        labels = ["Meeting", "Complaint", "Sales", "Spam"]
        for i, (qc, lb) in enumerate(zip(qcols, labels)):
            with qc:
                st.button(lb, width="stretch", key=f"qf{i}", on_click=fill_sample, args=(SAMPLE_EMAILS[i],))

        bcol1, bcol2 = st.columns([3, 1])
        with bcol1:
            go = st.button("🚀 Process Email", type="primary", width="stretch",
                           disabled=not (sender.strip() and subject.strip() and body.strip() and can_run))
        with bcol2:
            st.button("🧹 Clear", width="stretch", on_click=clear_form)

        if not can_run:
            st.caption("Enter an API key in the sidebar, or switch on 🧪 Demo Mode to try this without one.")

    with col_out:
        st.markdown("### Result")
        if go:
            email = {"from": sender.strip(), "subject": subject.strip(), "body": body.strip()}
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
    source = st.radio("Source", [f"Sample batch ({len(SAMPLE_EMAILS)} emails)", "Upload CSV"],
                      horizontal=True, key="batch_source")

    if source.startswith("Sample"):
        for s in SAMPLE_EMAILS:
            with st.expander(f"📧 {s['subject']} — {s['from']}"):
                st.text(s["body"])
        batch_box = st.container()
        if st.button(f"⚡ Process All {len(SAMPLE_EMAILS)} Emails", type="primary", disabled=not can_run,
                     key="batch_run_sample"):
            run_batch(SAMPLE_EMAILS, mock_mode, batch_box)
        if not can_run:
            st.caption("Enter an API key in the sidebar, or switch on 🧪 Demo Mode to try this without one.")
    else:
        st.caption("Columns needed: **from**, **subject**, **body** (aliases like sender, title, message also work).")
        st.download_button("⬇️ Download CSV template", data=CSV_TEMPLATE, file_name="emails_template.csv", mime="text/csv")
        up = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
        if up is not None:
            parsed, parse_error = [], None
            try:
                parsed = parse_email_csv(up)
            except ValueError as ve:
                parse_error = str(ve)
            if parse_error:
                st.error(parse_error)
            elif not parsed:
                st.warning("No usable rows found in that file.")
            else:
                st.write(f"Parsed **{len(parsed)}** email(s):")
                st.dataframe(parsed, width="stretch", height=220)
                batch_box2 = st.container()
                if st.button(f"⚡ Process {len(parsed)} Email(s)", type="primary", disabled=not can_run, key="batch_run_csv"):
                    run_batch(parsed, mock_mode, batch_box2)
                if not can_run:
                    st.caption("Enter an API key in the sidebar, or switch on 🧪 Demo Mode to try this without one.")


# ── Tab 3: Dashboard ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Dashboard")
    emails = st.session_state.processed_emails
    if not emails:
        st.info("No emails processed yet — head to **✉️ Compose** or **📦 Batch** to get started.")
    else:
        # Counted by current status so the numbers always add up after manual
        # approve / reject / send-back-to-review actions.
        auto = sum(1 for e in emails if e["routing"]["approval_status"] == "auto-approved")
        pending = sum(1 for e in emails if e["routing"]["approval_status"] == "pending-human-review")
        urgent = sum(1 for e in emails if e["routing"]["priority"] in ("urgent", "high"))
        avg_conf = sum(float(e["analysis"].get("confidence", 0) or 0) for e in emails) / len(emails)

        kpis = [
            ("Processed", len(emails), "var(--accent1)"),
            ("Auto-Approved", auto, "#22c55e"),
            ("Needs Review", pending, "#f59e0b"),
            ("High/Urgent", urgent, "#ef4444"),
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
        _t = THEMES[st.session_state.theme]
        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            st.caption("Intent distribution")
            bar_chart(counts_for(emails, lambda r: r["analysis"].get("intent", "information_request"), INTENT_CATEGORIES.keys()), _t["accent1"])
        with ch2:
            st.caption("Sentiment")
            bar_chart(counts_for(emails, lambda r: r["analysis"].get("sentiment", "neutral"), SENTIMENTS), _t["accent2"])
        with ch3:
            st.caption("Priority")
            bar_chart(counts_for(emails, lambda r: r["routing"].get("priority", "none"), PRIORITIES), "#f97316")

        st.markdown("---")
        with st.expander("🔎 Filter & search", expanded=False):
            fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
            with fc1:
                q = st.text_input("Search", placeholder="subject, sender, or summary…", label_visibility="collapsed")
            with fc2:
                f_intent = st.multiselect("Intent", list(INTENT_CATEGORIES), placeholder="Intent", label_visibility="collapsed")
            with fc3:
                f_status = st.multiselect("Status", STATUSES, placeholder="Status", label_visibility="collapsed")
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
            _order = {p: i for i, p in enumerate(PRIORITIES)}
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
                    if st.button("🔁 Re-run analysis", key=f"dash_rerun_{result['id']}", width="stretch", disabled=not can_run):
                        cobj = None if mock_mode else make_client(st.session_state.api_key, st.session_state.base_url)
                        new_result, err = process_full(cobj, st.session_state.model, email, mock=mock_mode, expanded=False)
                        if err:
                            # No rerun on failure, so the message stays visible.
                            st.error(friendly_error(err))
                        else:
                            result["analysis"] = new_result["analysis"]
                            result["draft"] = new_result["draft"]
                            result["routing"] = new_result["routing"]
                            result["mock"] = mock_mode
                            st.session_state.pop(f"dash_ta_{result['id']}", None)
                            st.session_state.pop(f"compose_ta_{result['id']}", None)
                            st.toast("Re-analyzed")
                            st.rerun()
