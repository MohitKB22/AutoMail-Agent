import pytest
from fastapi.testclient import TestClient

import agent
import api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(agent, "API_KEY", "")
    api.email_store.clear()
    return TestClient(api.app)


def test_root_and_samples(client):
    assert client.get("/").json()["status"] == "running"
    assert len(client.get("/emails/samples").json()["emails"]) == 4


def test_process_mock_and_approve_flow(client):
    r = client.post("/emails/process", json={"from": "a@b.com", "subject": "Refund NOW!!", "body": "This is unacceptable, I want a refund today.", "mock": True})
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert res["analysis"]["intent"] == "complaint"
    assert res["routing"]["approval_status"] == "pending-human-review"
    ok = client.post("/emails/approve", json={"email_id": res["id"], "approved": True})
    assert ok.json()["approval_status"] == "approved"
    assert client.get("/emails/processed").json()["emails"][0]["routing"]["approval_status"] == "approved"


def test_from_alias_accepted(client):
    r = client.post("/emails/process", json={"from_": "z@z.com", "subject": "hi", "body": "pricing?", "mock": True})
    assert r.json()["result"]["email"]["from"] == "z@z.com"


def test_same_id_upserts_instead_of_duplicating(client):
    for _ in range(2):
        client.post("/emails/process", json={"id": "001", "subject": "s", "body": "b", "mock": True})
    assert client.get("/emails/processed").json()["count"] == 1


def test_generated_ids_are_unique(client):
    ids = {client.post("/emails/process", json={"subject": f"s{i}", "body": "b", "mock": True}).json()["result"]["id"] for i in range(5)}
    assert len(ids) == 5


def test_empty_email_rejected(client):
    assert client.post("/emails/process", json={"from": "a@b.com"}).status_code == 422


def test_missing_key_returns_503_with_hint(client):
    r = client.post("/emails/process", json={"subject": "s", "body": "b"})
    assert r.status_code == 503 and "mock" in r.json()["detail"]


def test_approve_unknown_and_bad_payload(client):
    assert client.post("/emails/approve", json={"email_id": "nope", "approved": True}).status_code == 404
    assert client.post("/emails/approve", json={"approved": True}).status_code == 422
