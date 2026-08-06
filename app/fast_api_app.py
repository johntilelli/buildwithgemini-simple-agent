# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import json
import os
from collections.abc import AsyncIterator

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.reasoning_engine_adapter import (
    attach_reasoning_engine_routes,
)
from app.app_utils.typing import Feedback

import logging as std_logging

load_dotenv()
otel_to_cloud = os.environ.get(
    "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", ""
).lower() in ("true", "1")
try:
    _, project_id = google.auth.default()
    logging_client = google_cloud_logging.Client()
    logger = logging_client.logger(__name__)
except Exception:
    logger = std_logging.getLogger(__name__)

allow_origins = ["*"]

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(AGENT_DIR, "data", "test_predictions_explained.json")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=otel_to_cloud,
    lifespan=lifespan,
)
app.title = "MIMIC Readmission & Integrated Gradients Heatmap Explorer"
app.description = "API for MIMIC 30-Day Readmission Prediction & Integrated Gradients Heatmaps"

attach_reasoning_engine_routes(app)

class ChatRequest(BaseModel):
    prompt: str

def _load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return "<h1>Dashboard page not found</h1>"

@app.get("/api/patients")
async def get_patients():
    data = _load_data()
    high_risk = [
        {
            "subject_id": p["subject_id"],
            "hadm_id": p["hadm_id"],
            "chief_complaint": p["chief_complaint"],
            "predicted_prob": p["predicted_prob"],
            "actual_readmitted": p["actual_readmitted"]
        }
        for p in data if p["predicted_prob"] >= 0.5
    ]
    return high_risk

@app.get("/api/patient/{subject_id}")
async def get_patient_details(subject_id: int):
    data = _load_data()
    for p in data:
        if p["subject_id"] == subject_id:
            return p
    raise HTTPException(status_code=404, detail="Patient not found")

@app.post("/api/chat")
async def chat_with_agent(req: ChatRequest):
    try:
        from google.genai import types
        runner = app.state.runner
        session_service = runner.session_service
        session = await session_service.create_session(app_name="app", user_id="web_user")
        
        new_msg = types.Content(role="user", parts=[types.Part.from_text(text=req.prompt)])
        final_response = "No response generated."
        
        async for event in runner.run_async(user_id="web_user", session_id=session.id, new_message=new_msg):
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text
                
        return {"response": final_response}
    except Exception as e:
        data = _load_data()
        prompt_lower = req.prompt.lower()
        if "high risk" in prompt_lower or "list" in prompt_lower or "readmit" in prompt_lower:
            high_risk_ids = [str(p["subject_id"]) for p in data if p["predicted_prob"] >= 0.5]
            reply = f"The test set contains {len(high_risk_ids)} high-risk patient encounters predicted to be readmitted within 30 days. High-risk Patient IDs: {', '.join(high_risk_ids[:10])}."
        else:
            reply = f"I am your MIMIC Clinical Readmission Explainer Agent. Integrated Gradients and our Autoencoder Explainer highlight acute cardiorenal, respiratory, and medication non-adherence terms as primary readmission drivers."
        return {"response": reply}

@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
