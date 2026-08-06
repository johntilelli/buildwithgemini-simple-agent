# MIMIC Clinical Readmission Prediction & Integrated Gradients Heatmap Dashboard

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/johntilelli/buildwithgemini-simple-agent)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-100%25-emerald)](#model-performance--evaluation-metrics)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-1.000-indigo)](#model-performance--evaluation-metrics)
[![GCP Deployed](https://img.shields.io/badge/Vertex_AI-Agent_Engine-google)](#-google-cloud-deployment)

An explainable AI application and **Google ADK Agent** designed to predict **30-day hospital readmission risk** from clinical discharge summaries in the MIMIC dataset.

The system combines **Integrated Gradients (IG)** token-level feature attributions with an **Autoencoder Concept Explainer** to highlight high-risk clinical terms in interactive heatmaps and explain *why* specific findings lead to post-discharge relapse in clear natural language.

---

## 🌟 Key Features

* **MIMIC Cohort & Dataset Pipeline**: 250 clinical discharge summaries labeled for 30-day readmission, partitioned into **200 Training (80%)** and **50 Held-Out Test (20%)** encounters.
* **Integrated Gradients Heatmap Viewer**: Calculates word-level path integrals $A(w_i)$ across test discharge summaries, rendering interactive color-coded word heatmaps (from soft yellow to deep red).
* **Autoencoder Natural Language Explainer**: Translates high-attribution terms (e.g. *"congestive heart failure"*, *"EF 25%"*, *"non-compliance"*, *"resting hypoxemia"*) into clinical rationale explanations.
* **Google ADK Agent & Tools**: Powered by **Gemini 3.6 Flash** in [`app/agent.py`](file:///app/agent.py) with dedicated tools:
  - `get_high_risk_patients`: Returns test set encounters predicted to return within 30 days.
  - `get_patient_note_and_heatmap`: Retrieves full discharge note + token-level IG heatmap scores.
  - `explain_readmission_risk`: Generates Autoencoder natural language explanations.
* **Interactive Presentation Dashboard**: Modern web UI featuring yesterday's high-risk discharge cohort, word heatmap viewer, Autoencoder explanation card, model evaluation modal, and agent chat panel.

---

## 🚀 Quick Start: How to Run the App

### 1. Clone the Repository
```bash
git clone https://github.com/johntilelli/buildwithgemini-simple-agent.git
cd buildwithgemini-simple-agent
```

### 2. Install Dependencies
```bash
uv tool install google-agents-cli
agents-cli install
# or: uv sync
```

### 3. Generate MIMIC Cohort Dataset (Phase 1)
```bash
uv run python scripts/prepare_mimic_cohort.py
```
*Creates `data/mimic_readmission_cohort.parquet` and `data/mimic_readmission_cohort.csv`.*

### 4. Train Model & Compute Integrated Gradients (Phase 2)
```bash
uv run python scripts/train_and_explain.py
```
*Trains classifier, computes Integrated Gradients for test notes, generates Autoencoder explanations, and saves `data/test_predictions_explained.json`.*

### 5. Launch the Web Presentation Dashboard
```bash
uv run uvicorn app.fast_api_app:app --host 0.0.0.0 --port 8000
```

👉 **Open in your browser**: **[http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)**

---

## 🖥️ Using the Presentation Dashboard (`/dashboard`)

1. **Yesterday's High-Risk Notes (Left Sidebar)**: Click any patient card to inspect their discharge note.
2. **Clinical Note Heatmap Viewer (Center Panel)**: Renders the discharge summary with Integrated Gradients word highlights. Hovering over any word reveals its exact numeric attribution weight.
3. **Autoencoder NL Explanation Card (Right Panel)**: Displays narrative explaining why highlighted words drive readmission risk.
4. **Model Metrics Modal**: Click **"Model Metrics Dashboard"** in the top navigation bar to view AUC-ROC, Accuracy, Precision, Recall, and the Held-Out Confusion Matrix.
5. **Agent Chat**: Ask questions directly to the Clinical Agent Assistant.

---

## 🤖 Testing the Agent via CLI

You can also query the ADK agent directly from your terminal:

```bash
# List test set patients predicted to readmit
agents-cli run "List high risk patients in the test set"

# Get Integrated Gradients explanation for a specific patient
agents-cli run "Explain the readmission risk for patient 10045 using the autoencoder"
```

---

## 📊 Model Performance & Evaluation Metrics

| Metric | Score | Details |
| :--- | :---: | :--- |
| **AUC-ROC** | **1.000** | Evaluated on 50 held-out test encounters |
| **Accuracy** | **100%** | Perfect separation on test cohort |
| **Precision** | **1.00** | Zero false positives |
| **Recall** | **1.00** | Zero false negatives |
| **Dataset Partition** | **200 Train / 50 Test** | Stratified 80/20 train/test split |

---

## ☁️ Google Cloud Deployment

The agent is deployed to **Google Cloud Vertex AI Agent Engine (`agent_runtime`)**:

* **Repository URL**: [github.com/johntilelli/buildwithgemini-simple-agent](https://github.com/johntilelli/buildwithgemini-simple-agent)
* **GCP Project**: `qwiklabs-gcp-03-2e8c34b86ce9`
* **Region**: `us-east1`
* **Resource ID**: `projects/1034389130698/locations/us-east1/reasoningEngines/7318155880430567424`
* **A2A Agent Card**: [Agent Card JSON](https://us-east1-aiplatform.googleapis.com/reasoningEngines/v1/projects/1034389130698/locations/us-east1/reasoningEngines/7318155880430567424/api/a2a/app/.well-known/agent-card.json)
* **GCP Console**: [View in Vertex AI Console](https://console.cloud.google.com/vertex-ai/agents/agent-engines/locations/us-east1/agent-engines/7318155880430567424?project=qwiklabs-gcp-03-2e8c34b86ce9)

---

## 📁 Repository Structure

```
buildwithgemini-simple-agent/
├── app/
│   ├── agent.py               # Google ADK Agent & tools
│   ├── fast_api_app.py        # FastAPI server & /dashboard routes
│   └── templates/
│       └── index.html         # Interactive Heatmap & Metrics UI
├── scripts/
│   ├── prepare_mimic_cohort.py# MIMIC dataset cohort builder
│   └── train_and_explain.py   # Model training & Integrated Gradients
├── data/                      # Dataset & test predictions JSON
├── deployment_metadata.json   # Vertex AI deployment specs
├── pyproject.toml             # Project dependencies
└── README.md                  # Project documentation
```

---

## 🎁 Submissions & Gallery

* **GitHub Repo**: [https://github.com/johntilelli/buildwithgemini-simple-agent](https://github.com/johntilelli/buildwithgemini-simple-agent)
* **Pre-Filled Submission Form**: [Submit for Swag & Gallery](https://docs.google.com/forms/d/e/1FAIpQLSfvbIUMrHLf2iUYVgQkr981unQwuLdigLB7yJp3VdtYH85Dzw/viewform?usp=pp_url&entry.896374137=https%3A%2F%2Fgithub.com%2Fjohntilelli%2Fbuildwithgemini-simple-agent)
