"""
FastAPI backend — uses openai library, works with any OpenAI-compatible provider.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import analyze_email, draft_reply, route_email, process_email, SAMPLE_EMAILS

app = FastAPI(title="Email Intelligence Agent API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

email_store = []

@app.get("/")
def root(): return {"status": "running"}

@app.get("/emails/samples")
def get_samples(): return {"emails": SAMPLE_EMAILS}

@app.post("/emails/process")
def process_single(email_data: dict):
    try:
        email = {
            "id": email_data.get("id", f"e{len(email_store)+1:03d}"),
            "from": email_data.get("from") or email_data.get("from_", ""),
            "subject": email_data.get("subject", ""),
            "body": email_data.get("body", ""),
        }
        result = process_email(email)
        result["id"] = email["id"]
        email_store.append(result)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/emails/processed")
def get_processed(): return {"emails": email_store, "count": len(email_store)}

@app.post("/emails/approve")
def approve(data: dict):
    for e in email_store:
        if e.get("id") == data.get("email_id"):
            e["routing"]["approval_status"] = "approved" if data.get("approved") else "rejected"
            return {"success": True}
    raise HTTPException(404, "Email not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
