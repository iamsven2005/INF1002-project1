from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os

from transformers import pipeline

MODEL_DIR = "./food_sentiment_model"
LABELS = {0: "negative", 1: "neutral", 2: "positive"}

app = FastAPI(title="Food Review Sentiment App")

_classifier = None

class PredictIn(BaseModel):
    text: str

def load_model():
    global _classifier
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

    _classifier = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        return_all_scores=False,
    )

@app.on_event("startup")
def startup():
    # Load once at server start
    load_model()

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _classifier is not None}

@app.post("/predict")
def predict(inp: PredictIn):
    if _classifier is None:
        return {"error": "Model not loaded"}

    text = (inp.text or "").strip()
    if not text:
        return {"error": "Empty text"}

    out = _classifier(text)[0]  # {'label': 'LABEL_2', 'score': 0.98}
    label_id = int(out["label"].split("_")[-1])
    return {
        "label": LABELS.get(label_id, out["label"]),
        "confidence": float(out["score"])
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Food Review Sentiment</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #0b1020; color: #e8ecff; }
    .wrap { max-width: 900px; margin: 0 auto; padding: 28px 18px; }
    .card { background: #121a33; border: 1px solid #22305d; border-radius: 14px; padding: 18px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
    h1 { margin: 0 0 8px; font-size: 22px; }
    p { margin: 0 0 14px; color: #b9c2ff; }
    textarea { width: 100%; min-height: 140px; padding: 12px; border-radius: 12px; border: 1px solid #2a3a72; background: #0e1530; color: #e8ecff; outline: none; resize: vertical; }
    textarea:focus { border-color: #5c7cfa; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
    button { padding: 10px 14px; border-radius: 12px; border: 1px solid #2a3a72; background: #17214a; color: #e8ecff; cursor: pointer; }
    button:hover { border-color: #5c7cfa; }
    .pill { display: inline-flex; gap: 8px; align-items: center; padding: 8px 10px; border-radius: 999px; border: 1px solid #2a3a72; background: #0e1530; color: #e8ecff; }
    .muted { color: #b9c2ff; }
    .err { color: #ffb4b4; }
    .ok { color: #b6ffcc; }
    .sp { height: 12px; }
    code { background: rgba(255,255,255,.06); padding: 2px 6px; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Food Review Sentiment</h1>
      <p>Type a review and hit <b>Predict</b>. Backend: <code>/predict</code> (FastAPI + Transformers).</p>

      <textarea id="txt" placeholder="e.g. The laksa was amazing, but the service was slow."></textarea>

      <div class="row">
        <button id="btn">Predict</button>
        <button id="demo">Load sample</button>
        <span class="pill"><span class="muted">Status:</span> <span id="status">Ready</span></span>
      </div>

      <div class="sp"></div>
      <div class="pill" style="width: fit-content;">
        <span class="muted">Label:</span> <b id="label">—</b>
        <span class="muted">Confidence:</span> <b id="conf">—</b>
      </div>

      <div class="sp"></div>
      <div id="msg" class="muted"></div>
    </div>
  </div>

<script>
  const txt = document.getElementById('txt');
  const btn = document.getElementById('btn');
  const demo = document.getElementById('demo');
  const statusEl = document.getElementById('status');
  const labelEl = document.getElementById('label');
  const confEl = document.getElementById('conf');
  const msgEl = document.getElementById('msg');

  demo.addEventListener('click', () => {
    txt.value = "Food was great and portion was generous. Would come back again!";
    msgEl.textContent = "";
  });

  function setStatus(text, cls="") {
    statusEl.textContent = text;
    statusEl.className = cls;
  }

  btn.addEventListener('click', async () => {
    const text = (txt.value || "").trim();
    if (!text) {
      msgEl.textContent = "Please enter some text.";
      msgEl.className = "err";
      return;
    }

    setStatus("Predicting...", "muted");
    msgEl.textContent = "";
    labelEl.textContent = "—";
    confEl.textContent = "—";

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || "Request failed");
      }

      labelEl.textContent = data.label;
      confEl.textContent = (data.confidence * 100).toFixed(2) + "%";
      setStatus("Done", "ok");
    } catch (e) {
      setStatus("Error", "err");
      msgEl.textContent = String(e.message || e);
      msgEl.className = "err";
    }
  });
</script>
</body>
</html>
    """
