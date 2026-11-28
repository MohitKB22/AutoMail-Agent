"""
FastAPI backend — uses the openai library, works with any OpenAI-compatible provider.

Run:  python api.py            (or: uvicorn api:app --host 0.0.0.0 --port 8000)
Docs: http://localhost:8000/docs

Set API_KEY / BASE_URL / MODEL (env or .env) for a real model, or send
"mock": true in the request body to use the local Demo Mode simulator.
"""
from __future__ import annotations

import os
import sys
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402
from agent import SAMPLE_EMAILS, MissingAPIKeyError, process_email  # noqa: E402

app = FastAPI(title="Email Intelligence Agent API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# id → processed result (insertion-ordered). In-memory only: resets on restart.
email_store: dict[str, dict] = {}


class EmailIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = None
    from_: str = Field("", alias="from")
    subject: str = ""
    body: str = ""
    mock: bool = Field(False, description="Use the local Demo Mode simulator instead of a real model.")


class ApprovalIn(BaseModel):
    email_id: str
    approved: bool


@app.get("/")
def root():
    return {
        "status": "running",
        "model_configured": bool(agent.API_KEY),
        "model": agent.MODEL if agent.API_KEY else None,
        "docs": "/docs",
    }


@app.get("/emails/samples")
def get_samples():
    return {"emails": SAMPLE_EMAILS}


@app.post("/emails/process")
def process_single(email_in: EmailIn):
    if not (email_in.subject.strip() or email_in.body.strip()):
        raise HTTPException(422, "Provide at least a subject or a body.")
    email = {
        "id": email_in.id or uuid.uuid4().hex[:8],
        "from": email_in.from_,
        "subject": email_in.subject,
        "body": email_in.body,
    }
    try:
        result = process_email(email, mock=email_in.mock, verbose=False)
    except MissingAPIKeyError as e:
        raise HTTPException(503, f'{e} Or send "mock": true to use the simulator.') from e
    except Exception as e:  # noqa: BLE001 — provider/network/parse failures
        raise HTTPException(502, f"Model provider error: {e}") from e
    result["id"] = email["id"]
    email_store[email["id"]] = result  # re-processing the same id replaces the old result
    return {"success": True, "result": result}


@app.get("/emails/processed")
def get_processed():
    emails = list(email_store.values())
    return {"emails": emails, "count": len(emails)}


@app.post("/emails/approve")
def approve(data: ApprovalIn):
    result = email_store.get(data.email_id)
    if result is None:
        raise HTTPException(404, "Email not found")
    result["routing"]["approval_status"] = "approved" if data.approved else "rejected"
    return {"success": True, "id": data.email_id, "approval_status": result["routing"]["approval_status"]}


if __name__ == "__main__":
    import uvicorn

    # reload/workers need an import string, not the app object (otherwise uvicorn exits immediately).
    uvicorn.run(
        "api:app",
        app_dir=os.path.dirname(os.path.abspath(__file__)),
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "").lower() in ("1", "true", "yes"),
    )
