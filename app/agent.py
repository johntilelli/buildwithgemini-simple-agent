import os
import json
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

MODEL = "gemini-3.6-flash"

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_predictions_explained.json")

def _load_predictions_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def get_high_risk_patients(min_probability: float = 0.5) -> str:
    """Retrieves a list of patient encounters from the test set predicted to be at high risk for 30-day hospital readmission.

    Args:
        min_probability: Minimum predicted readmission probability threshold (default 0.5).

    Returns:
        A JSON string formatted list of high-risk patient encounters.
    """
    data = _load_predictions_data()
    high_risk = [
        {
            "subject_id": p["subject_id"],
            "hadm_id": p["hadm_id"],
            "chief_complaint": p["chief_complaint"],
            "predicted_readmission_prob": f"{p['predicted_prob']*100:.1f}%",
            "top_risk_terms": p["top_risk_terms"][:5]
        }
        for p in data if p["predicted_prob"] >= min_probability
    ]
    return json.dumps(high_risk, indent=2)

def get_patient_note_and_heatmap(subject_id: int) -> str:
    """Retrieves the full clinical discharge note and Integrated Gradients token-level attribution heatmap for a specific patient ID.

    Args:
        subject_id: The patient subject ID (e.g. 10001).

    Returns:
        A JSON string containing the full discharge note, predicted readmission probability, and word-level Integrated Gradients heatmap weights.
    """
    data = _load_predictions_data()
    for p in data:
        if p["subject_id"] == subject_id:
            return json.dumps({
                "subject_id": p["subject_id"],
                "hadm_id": p["hadm_id"],
                "predicted_prob": p["predicted_prob"],
                "chief_complaint": p["chief_complaint"],
                "note_tokens_with_ig": p["note_tokens_with_ig"],
                "full_note": p["full_note"]
            }, indent=2)
    return f"Patient subject_id {subject_id} not found in test dataset."

def explain_readmission_risk(subject_id: int) -> str:
    """Uses the Autoencoder Concept Explainer model to generate a natural language clinical explanation of why a patient is predicted to be readmitted.

    Args:
        subject_id: The patient subject ID (e.g. 10001).

    Returns:
        A JSON string containing the Autoencoder natural language summary, key clinical risk factors, and detailed clinical rationale.
    """
    data = _load_predictions_data()
    for p in data:
        if p["subject_id"] == subject_id:
            return json.dumps({
                "subject_id": p["subject_id"],
                "explanation": p["explanation"]
            }, indent=2)
    return f"Patient subject_id {subject_id} not found in test dataset."

root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=MODEL,
        vertexai=True,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are an expert Clinical Readmission Explainer Agent powered by MIMIC clinical notes, an Integrated Gradients attribution model, and an Autoencoder Concept Explainer.
    
Your role is to assist clinical teams by:
1. Identifying high-risk patients predicted to be readmitted within 30 days.
2. Displaying clinical discharge notes with Integrated Gradients word attributions/heatmaps.
3. Explaining in clear natural language WHY specific words and clinical findings in the discharge note drive 30-day readmission risk.""",
    tools=[get_high_risk_patients, get_patient_note_and_heatmap, explain_readmission_risk],
)

app = App(
    root_agent=root_agent,
    name="app",
)
