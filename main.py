# FastAPI web framework
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
# Pydantic for request validation
from pydantic import BaseModel

# OS utilities (for checking model folder existence)
import os

# HuggingFace pipeline for NLP inference
from transformers import pipeline


# -----------------------------
# Configuration
# -----------------------------

# Path to your locally saved HuggingFace model
MODEL_DIR = "./food_sentiment_model"

# Mapping from model output IDs to human-readable labels
LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Create FastAPI app instance
app = FastAPI(title="Food Review Sentiment App")

# Global classifier object
# Loaded once on startup and reused for all requests
_classifier = None


# -----------------------------
# Request schema
# -----------------------------

class PredictIn(BaseModel):
    """
    Schema for the POST /predict endpoint.
    Expects a JSON body like:
    {
        "text": "The food was amazing"
    }
    """
    text: str


# -----------------------------
# Model loading
# -----------------------------

def load_model():
    """
    Loads the HuggingFace sentiment model into memory.
    This runs only once when the server starts.
    """
    global _classifier

    # Ensure the model directory exists
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

    # Create a HuggingFace pipeline for text classification
    _classifier = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        return_all_scores=False,   # Only return the best label
    )


# Automatically run when FastAPI starts
@app.on_event("startup")
def startup():
    """
    Startup hook for FastAPI.
    Loads the ML model into memory before any request is served.
    """
    load_model()


# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.
    Used to verify that:
    - The API is alive
    - The model is loaded
    """
    return {
        "ok": True,
        "model_loaded": _classifier is not None
    }


@app.post("/predict")
def predict(inp: PredictIn):
    """
    Runs sentiment prediction on input text.

    Input:
        { "text": "Food was great!" }

    Output:
        {
          "label": "positive",
          "confidence": 0.97
        }
    """
    # Safety check
    if _classifier is None:
        return {"error": "Model not loaded"}

    # Clean and validate input text
    text = (inp.text or "").strip()
    if not text:
        return {"error": "Empty text"}

    # Run inference
    # Example output:
    # [{'label': 'LABEL_2', 'score': 0.98}]
    out = _classifier(text)[0]

    # Convert LABEL_2 → 2
    label_id = int(out["label"].split("_")[-1])

    # Return human-readable label
    return {
        "label": LABELS.get(label_id, out["label"]),
        "confidence": float(out["score"])
    }


# -----------------------------
# Frontend (Simple HTML UI)
# -----------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    """
    Serves a small single-page web UI.
    The page:
    - Lets the user type a food review
    - Sends it to /predict
    - Displays label + confidence
    """
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Food Review Sentiment</title>
  <style>
    /* Basic dark UI styling */
    body {
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      margin: 0;
      background: #0b1020;
      color: #e8ecff;
    }
    .wrap { max-width: 900px; margin: 0 auto; padding: 28px 18px; }
    .card {
      background: #121a33;
      border: 1px solid #22305d;
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    textarea {
      width: 100%;
      min-height: 140px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid #2a3a72;
      background: #0e1530;
      color: #e8ecff;
    }
    button {
      padding: 10px 14px;
      border-radius: 12px;
      border: 1px solid #2a3a72;
      background: #17214a;
      color: #e8ecff;
      cursor: pointer;
    }
    .pill {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid #2a3a72;
      background: #0e1530;
    }
    .err { color: #ffb4b4; }
    .ok { color: #b6ffcc; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Food Review Sentiment</h1>
      <p>Type a review and hit <b>Predict</b>. Backend: <code>/predict</code> Have fun!</p>

      <!-- Text input -->
      <textarea id="txt" placeholder="e.g. The laksa was amazing but the service was slow."></textarea>

      <!-- Buttons + status -->
      <div>
        <button id="btn">Predict</button>
        <button id="demo">Load sample</button>
        <span class="pill">Status: <span id="status">Ready</span></span>
      </div>

      <!-- Results -->
      <div class="pill">
        Label: <b id="label">—</b>
        Confidence: <b id="conf">—</b>
      </div>

      <div id="msg"></div>
    </div>
  </div>

<script>
  // DOM references
  const txt = document.getElementById('txt');
  const btn = document.getElementById('btn');
  const demo = document.getElementById('demo');
  const statusEl = document.getElementById('status');
  const labelEl = document.getElementById('label');
  const confEl = document.getElementById('conf');
  const msgEl = document.getElementById('msg');

  // Fill demo text
  demo.addEventListener('click', () => {
    txt.value = "Food was great and portion was generous. Would come back again!";
  });

  // Update status pill
  function setStatus(text, cls="") {
    statusEl.textContent = text;
    statusEl.className = cls;
  }

  // Call backend prediction
  btn.addEventListener('click', async () => {
    const text = (txt.value || "").trim();
    if (!text) {
      msgEl.textContent = "Please enter some text.";
      msgEl.className = "err";
      return;
    }

    setStatus("Predicting...");
    labelEl.textContent = "—";
    confEl.textContent = "—";

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "Request failed");

      labelEl.textContent = data.label;
      confEl.textContent = (data.confidence * 100).toFixed(2) + "%";
      setStatus("Done", "ok");
    } catch (e) {
      setStatus("Error", "err");
      msgEl.textContent = e.message;
      msgEl.className = "err";
    }
  });
</script>
</body>
</html>
"""
