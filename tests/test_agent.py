import json

import httpx
import pytest
from openai import BadRequestError

import agent
from agent import (
    SAMPLE_EMAILS, analyze_email, chat, draft_reply, mock_analyze, normalize_analysis,
    normalize_confidence, normalize_intent, parse_json_object, process_email, route_email,
)

GOOD = {"intent": "complaint", "sentiment": "negative", "summary": "Wants refund.",
        "key_entities": ["#78432"], "requires_human_approval": True, "confidence": 0.93}


@pytest.mark.parametrize("raw", [
    json.dumps(GOOD),
    "```json\n" + json.dumps(GOOD) + "\n```",
    "```\n" + json.dumps(GOOD) + "\n```",
    "Here is the analysis:\n" + json.dumps(GOOD) + "\nLet me know if you need more.",
    "Sure! ```JSON\n" + json.dumps(GOOD) + "``` hope that helps",
])
def test_parse_json_object_tolerates_wrappers(raw):
    assert parse_json_object(raw) == GOOD


def test_parse_json_object_rejects_non_objects():
    with pytest.raises(ValueError):
        parse_json_object("[1, 2, 3]")
    with pytest.raises(json.JSONDecodeError):
        parse_json_object("no json here")


@pytest.mark.parametrize("value,expected", [
    ("meeting_request", "meeting_request"), ("Meeting Request", "meeting_request"),
    ("FOLLOW-UP", "follow_up"), ("followup", "follow_up"), ("sales", "sales_inquiry"),
    ("totally-unknown", "information_request"), (None, "information_request"),
])
def test_normalize_intent(value, expected):
    assert normalize_intent(value) == expected


@pytest.mark.parametrize("value,expected", [
    (0.87, 0.87), ("0.87", 0.87), ("87%", 0.87), (87, 0.87),
    ("high", 0.5), (None, 0.5), (-3, 0.0), (True, 0.5), (250, 1.0),
])
def test_normalize_confidence(value, expected):
    assert normalize_confidence(value) == expected


def test_normalize_analysis_fills_and_cleans():
    out = normalize_analysis({"intent": "Urgent", "sentiment": "VERY ANGRY", "confidence": "90%",
                              "key_entities": "ACME", "requires_human_approval": "false"})
    assert out == {"intent": "urgent", "sentiment": "neutral", "summary": "No summary provided.",
                   "key_entities": ["ACME"], "requires_human_approval": False, "confidence": 0.9}
    assert normalize_analysis(None)["intent"] == "information_request"


def test_route_email_forces_review_when_model_asks():
    r = route_email({"intent": "meeting_request", "requires_human_approval": True})
    assert r["auto_approve"] is False and r["approval_status"] == "pending-human-review"
    assert route_email({"intent": "spam"})["approval_status"] == "auto-approved"
    assert route_email({"intent": "weird"})["team"] == "General"


@pytest.mark.parametrize("idx,intent", [(0, "meeting_request"), (1, "complaint"), (2, "sales_inquiry"), (3, "spam")])
def test_mock_classifies_samples(idx, intent):
    assert mock_analyze(SAMPLE_EMAILS[idx])["intent"] == intent


def test_mock_uses_whole_words():
    # Previously "plan" matched "explanation" and "call" matched "recall" → sales/meeting.
    a = mock_analyze({"from": "x@y.com", "subject": "re: docs", "body": "I recall the explanation in the docs."})
    assert a["intent"] == "information_request"


def test_mock_confidence_is_deterministic_across_processes():
    # built-in hash() is salted per process (PYTHONHASHSEED); crc32 is stable across restarts.
    import os, subprocess, sys
    code = "import agent; print(agent.mock_analyze(agent.SAMPLE_EMAILS[0])['confidence'])"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip() for seed in ("1", "2", "3")}
    assert len(outs) == 1 and 0.74 <= float(outs.pop()) <= 0.94


def test_mock_entities_skip_filler_words():
    ents = mock_analyze(SAMPLE_EMAILS[1])["key_entities"]
    assert "#78432" in ents
    assert not {"STILL", "This", "TODAY"} & set(ents)
    assert {"Q2", "Thursday", "Friday"} <= set(mock_analyze(SAMPLE_EMAILS[0])["key_entities"])


def test_analyze_email_handles_messy_model_output(fake_client):
    client, calls = fake_client('Analysis below:\n```json\n{"intent":"Sales Inquiry","sentiment":"Positive","summary":"Wants pricing","key_entities":[],"requires_human_approval":"no","confidence":"80%"}\n```')
    a = analyze_email(SAMPLE_EMAILS[2], client=client, model="m")
    assert a["intent"] == "sales_inquiry" and a["sentiment"] == "positive" and a["confidence"] == 0.8
    assert a["requires_human_approval"] is False
    assert calls.calls[0]["model"] == "m"


def test_chat_retries_with_max_completion_tokens(fake_client):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    err = BadRequestError("Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                          response=httpx.Response(400, request=req), body=None)
    client, calls = fake_client(err, "hello")
    assert chat(client, "o3-mini", "s", "u", max_tokens=100) == "hello"
    assert "max_tokens" in calls.calls[0] and "max_completion_tokens" in calls.calls[1]


def test_chat_rejects_empty_content(fake_client):
    client, _ = fake_client(None)
    with pytest.raises(ValueError, match="empty response"):
        chat(client, "m", "s", "u")


def test_draft_reply_survives_missing_analysis_keys(fake_client):
    client, calls = fake_client("Thanks! — The Team")
    assert draft_reply(SAMPLE_EMAILS[0], {}, client=client, model="m") == "Thanks! — The Team"
    assert "information_request" in calls.calls[0]["messages"][1]["content"]


def test_process_email_mock_end_to_end():
    r = process_email(SAMPLE_EMAILS[1], mock=True, verbose=False)
    assert r["routing"]["approval_status"] == "pending-human-review"
    assert "Customer Success" in r["draft_reply"]


def test_missing_api_key_is_explicit(monkeypatch):
    monkeypatch.setattr(agent, "API_KEY", "")
    with pytest.raises(agent.MissingAPIKeyError):
        analyze_email(SAMPLE_EMAILS[0])
