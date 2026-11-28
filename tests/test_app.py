"""Headless UI tests for the Streamlit dashboard (streamlit.testing AppTest)."""
import io
import os
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def fresh():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def ok(at):
    assert not at.exception, [e.value for e in at.exception]
    return at


def card_html(at):
    return [m.value for m in at.markdown if 'class="email-card' in m.value]


def test_quick_fill_and_process_demo():
    at = fresh()
    ok(at.button(key="qf1").click().run())
    ok([b for b in at.button if "Process Email" in b.label][0].click().run())
    assert len(at.session_state.processed_emails) == 1
    assert at.session_state.processed_emails[0]["analysis"]["intent"] == "complaint"


def test_card_escapes_html_and_has_no_blank_lines():
    at = fresh()
    at.text_input(key="compose_from").input("x@y.com")
    at.text_input(key="compose_subject").input("question about my order")
    at.text_area(key="compose_body").input('my order <img src=x onerror="alert(1)"> hasn\'t arrived.\n\nhelp?')
    ok(at.run())
    ok([b for b in at.button if "Process Email" in b.label][0].click().run())
    html_blocks = card_html(at)
    assert html_blocks, "result card not rendered"
    card = html_blocks[0]
    assert "<img" not in card and "&lt;img" in card
    assert "\n" not in card  # a blank line would turn the rest of the card into a raw-HTML code block


def test_txt_upload_callback_sets_body():
    # AppTest can't drive st.file_uploader, so exercise the callback directly.
    import types
    fake_st = types.SimpleNamespace(session_state=types.SimpleNamespace(
        get=lambda k: SimpleNamespace(getvalue=lambda: "﻿Body from file".encode("utf-8")) if k == "compose_file" else None))
    src = open(APP).read()
    ns = {}
    start = src.index("def load_txt_body")
    end = src.index("\n\n\n", start)
    exec(src[start:end], {"st": fake_st}, ns)
    ns["load_txt_body"]()
    assert fake_st.session_state.compose_body == "Body from file"


def test_batch_then_dashboard_actions():
    at = fresh()
    ok(at.button(key="batch_run_sample").click().run())
    emails = at.session_state.processed_emails
    assert len(emails) == 4
    # Re-run analysis, regenerate, edit, approve — all inside dashboard expanders.
    ok([b for b in at.button if (b.key or "").startswith("dash_rerun_")][0].click().run())
    ok([b for b in at.button if (b.key or "").startswith("dash_regen_")][0].click().run())
    rid = at.session_state.processed_emails[0]["id"]
    at.text_area(key=f"dash_ta_{rid}").input("EDITED")
    ok(at.button(key=f"dash_save_{rid}").click().run())
    assert at.session_state.processed_emails[0]["draft"] == "EDITED"
    ok([b for b in at.button if (b.key or "").startswith("dash_approve_")][0].click().run())
    statuses = sorted(e["routing"]["approval_status"] for e in at.session_state.processed_emails)
    assert statuses == ["approved", "auto-approved", "auto-approved", "auto-approved"]


def test_kpis_add_up_after_send_back_to_review():
    at = fresh()
    ok(at.button(key="batch_run_sample").click().run())
    ok([b for b in at.button if (b.key or "").startswith("dash_repend_")][0].click().run())
    import re
    kpi = {}
    for m in at.markdown:
        hit = re.search(r'class="metric-value"[^>]*>([^<]*)</div><div class="metric-label">([^<]*)<', m.value)
        if hit:
            kpi[hit.group(2)] = hit.group(1)
    assert kpi["Processed"] == "4"
    assert int(kpi["Auto-Approved"]) + int(kpi["Needs Review"]) == 4


def test_light_mode_and_provider_switch():
    at = fresh()
    ok(at.toggle(key="_light_toggle").set_value(True).run())
    assert at.session_state.theme == "light"
    ok(at.toggle(key="mock_mode").set_value(False).run())
    ok(at.selectbox(key="provider_choice").set_value("Groq").run())
    assert at.session_state.base_url == "https://api.groq.com/openai/v1"
    assert at.session_state.model == "llama-3.3-70b-versatile"
    # Without a key, processing is disabled rather than erroring.
    assert [b for b in at.button if "Process Email" in b.label][0].disabled


def test_model_name_is_escaped_in_header():
    at = fresh()
    at.toggle(key="mock_mode").set_value(False).run()
    at.text_input(key="model").input("<script>x</script>")
    ok(at.run())
    assert any("&lt;script&gt;" in m.value for m in at.markdown)
    assert not any("<script>x" in m.value for m in at.markdown)


def test_csv_parser_handles_excel_bom():
    src = open(APP).read()
    start = src.index("def parse_email_csv")
    end = src.index("\n\n\n", start)
    import csv as _csv, io as _io
    ns = {}
    exec(src[start:end], {"csv": _csv, "io": _io}, ns)
    data = "﻿From,Subject,Body\r\na@b.com,Hi,Pricing please\r\n,,\r\n".encode("utf-8")
    rows = ns["parse_email_csv"](SimpleNamespace(getvalue=lambda: data))
    assert rows == [{"from": "a@b.com", "subject": "Hi", "body": "Pricing please"}]
