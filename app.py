"""Food Review Sentiment Analysis API

A FastAPI application that classifies food reviews into positive, neutral, or negative sentiments.
Features:
- Real-time sentiment prediction on user input
- Span-level sentiment analysis for detailed insights
- Frontend UI with PII detection and sensitivity warnings
- Pre-trained HuggingFace transformer model
"""

# === External Dependencies ===
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline

# === Standard Library ===
import os
import re
from typing import List, Dict


# === Configuration Constants ===
# Path to locally saved HuggingFace model directory
MODEL_DIR = "./food_sentiment_model"

# Mapping from model output label IDs to human-readable sentiment labels
LABELS = {
    0: "negative",    # Negative sentiment
    1: "neutral",     # Neutral/mixed sentiment
    2: "positive"     # Positive sentiment
}

# === FastAPI Application ===
app = FastAPI(
    title="Food Review Sentiment App",
    description="Classifies food reviews by sentiment (positive/neutral/negative)",
    version="1.0.0"
)

# Global classifier object - loaded once on startup and reused for all requests
_classifier = None


# === Request/Response Schemas ===
class PredictIn(BaseModel):
    """Request schema for /predict and /predict_spans endpoints.
    
    Expects a JSON body containing the review text to classify.
    
    Example:
        {"text": "The food was amazing but service was slow."}
    """
    text: str

# === Text Processing Utilities ===
# Regex pattern to identify sentence boundaries and contrasting conjunctions
# Matches: "but", "however", "although", "yet" (case-insensitive) and punctuation (,.!?;)
SPLIT_RE = re.compile(r'(\bbut\b|\bhowever\b|\balthough\b|\byet\b|[,.!?;])', re.IGNORECASE)

def split_with_offsets(text: str) -> List[Dict]:
    """Split text into segments by sentence boundaries while preserving character offsets.
    
    This enables span-level sentiment analysis - analyzing sentiment of individual
    clauses rather than the entire text. Useful for detecting contrasting sentiments
    (e.g., "Great food but terrible service").
    
    Args:
        text: Input text to segment
    
    Returns:
        List of dictionaries with keys:
            - text: The segment content (trimmed)
            - start: Character position in original text
            - end: End position in original text
    """
    parts = []
    idx = 0
    start = 0

    # Split text while keeping separators ("but", "however", punctuation, etc.)
    tokens = SPLIT_RE.split(text)
    buf = ""              # Accumulates non-separator tokens
    buf_start = 0         # Start position of current buffer
    cur_pos = 0           # Current position in text

    for t in tokens:
        # Skip empty or None tokens from regex split
        if t is None or t == "":
            continue

        # Find the token's position in the original text
        pos = text.lower().find(t.lower(), cur_pos)
        if pos == -1:
            pos = cur_pos
        cur_pos = pos + len(t)

        # Check if current token is a separator (conjunction or punctuation)
        if SPLIT_RE.fullmatch(t):
            # Separator found - flush accumulated buffer as a complete segment
            seg = buf.strip()
            if seg:
                # Calculate segment position accounting for leading whitespace
                seg_start = buf_start + (len(buf) - len(buf.lstrip()))
                seg_end = seg_start + len(seg)
                parts.append({"text": seg, "start": seg_start, "end": seg_end})
            buf = ""
            buf_start = cur_pos
        else:
            # Accumulate non-separator tokens into buffer
            if buf == "":
                buf_start = pos
            buf += t

    # Flush any remaining buffer at end of text
    seg = buf.strip()
    if seg:
        seg_start = buf_start + (len(buf) - len(buf.lstrip()))
        seg_end = seg_start + len(seg)
        parts.append({"text": seg, "start": seg_start, "end": seg_end})

    return parts


# === Word Segmentation Utility ===
# Uses HuggingFace tokenizer to intelligently re-space text

def segment_text(text: str) -> str:
    """Re-insert spaces into text that has had them removed using tokenizer.
    
    Uses the HuggingFace tokenizer's encode/decode mechanism which is robust
    for handling unknown words through subword tokenization.
    For example: "thisisapen" → "this is a pen"
    
    Args:
        text: Input text without spaces
    
    Returns:
        Text with spaces re-inserted using tokenizer vocabulary
    """
    segmentations = segment_text_all(text)
    
    # Return the first segmentation found, or original text
    return segmentations[0] if segmentations else text


def segment_text_all(text: str) -> List[str]:
    """Find valid segmentations using HuggingFace tokenizer.
    
    Encodes text to tokens and decodes back with proper spacing.
    The tokenizer handles subword tokens (unknown words) gracefully.
    
    Args:
        text: Input text without spaces
    
    Returns:
        List containing the tokenizer-decoded segmentation
        Returns empty list if text cannot be processed
    """
    if _classifier is None:
        return []
    
    text = (text or "").strip()
    if not text:
        return []
    
    try:
        # Get the tokenizer from the classifier pipeline
        tokenizer = _classifier.tokenizer
        
        # Encode the text (converts to token IDs)
        # The tokenizer will split text into subwords intelligently
        encoded = tokenizer.encode(text)
        
        # Decode back (converts token IDs back to text with proper spacing)
        # skip_special_tokens=True removes [CLS], [SEP], etc.
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        
        # Return the decoded text as a single-element list
        # This maintains API compatibility
        if decoded and decoded.strip():
            return [decoded.strip()]
        else:
            return []
    
    except Exception as e:
        # If tokenizer fails, return empty list
        return []


# === Model Management ===
def load_model():
    """Load the HuggingFace sentiment model into memory.
    
    Loads the pre-trained transformer model from MODEL_DIR and initializes
    a text classification pipeline. Runs once at server startup.
    
    Raises:
        FileNotFoundError: If the model directory doesn't exist
    """
    global _classifier

    # Ensure the model directory exists before attempting to load
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

    # Initialize HuggingFace pipeline for text classification
    # return_all_scores=False: Only return the highest-confidence label
    _classifier = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        return_all_scores=False,   # Return only best prediction
    )


@app.on_event("startup")
def startup():
    """FastAPI startup event handler.
    
    Automatically called when the server starts. Loads the ML model
    into memory before any request is served.
    """
    load_model()


# === API Endpoints ===
@app.get("/health")
def health():
    """Health check endpoint.
    
    Verifies that:
    - The API server is running
    - The ML model has been loaded into memory
    
    Returns:
        dict: {"ok": bool, "model_loaded": bool}
    """
    return {
        "ok": True,
        "model_loaded": _classifier is not None
    }


@app.post("/segment")
def segment(inp: PredictIn):
    """Re-insert spaces into text that has had them removed.
    
    Uses the HuggingFace tokenizer's encode/decode mechanism to intelligently
    add spacing to text. The tokenizer handles unknown words through subword
    tokenization, making it robust for any input text.
    
    Args:
        inp: PredictIn object with text field (no spaces)
    
    Returns:
        dict: {
            "original": str,
            "segmentations": List[str] (tokenizer-decoded result),
            "count": int (should be 1)
        }
        
    Example:
        Request:  {"text": "thisisapen"}
        Response: {
            "original": "thisisapen",
            "segmentations": ["this is a pen"],
            "count": 1
        }
    """
    text = (inp.text or "").strip()
    if not text:
        return {"error": "Empty text"}
    
    segmentations = segment_text_all(text)
    return {
        "original": text,
        "segmentations": segmentations,
        "count": len(segmentations)
    }


@app.post("/predict_spans")
def predict_spans(inp: PredictIn):
    """Detailed sentiment prediction with span-level analysis.
    
    Splits the input text into segments at sentence boundaries and
    predicts sentiment for each segment individually, in addition to
    overall sentiment. Useful for detecting mixed sentiments.
    
    Args:
        inp: PredictIn object with text field
    
    Returns:
        dict with overall sentiment and per-span predictions with positions
    """
    # Check if model is initialized
    if _classifier is None:
        return {"error": "Model not loaded"}

    # Validate and clean input
    text = (inp.text or "").strip()
    if not text:
        return {"error": "Empty text"}

    # Split text into meaningful segments by sentence boundaries
    spans = split_with_offsets(text)

    # Predict sentiment for each span segment
    results = []
    for s in spans:
        # Get prediction for this span
        out = _classifier(s["text"])[0]
        label_id = int(out["label"].split("_")[-1])
        
        # Store result with position offsets for frontend highlighting
        results.append({
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "label": LABELS.get(label_id, out["label"]),
            "confidence": float(out["score"]),
        })

    # Also predict overall sentiment for the entire text
    overall = _classifier(text)[0]
    overall_id = int(overall["label"].split("_")[-1])

    return {
        "overall": {
            "label": LABELS.get(overall_id, overall["label"]),
            "confidence": float(overall["score"])
        },
        "spans": results
    }


@app.post("/predict")
def predict(inp: PredictIn):
    """Simple overall sentiment prediction.
    
    Predicts the sentiment of the entire input text in one go.
    Returns the most likely sentiment label and confidence score.

    Args:
        inp: PredictIn object with text field

    Returns:
        dict: {
            "label": "positive" | "neutral" | "negative",
            "confidence": float (0-1)
        }
        
    Example:
        Request:  {"text": "The food was amazing!"}
        Response: {"label": "positive", "confidence": 0.97}
    """
    # Check if model is initialized
    if _classifier is None:
        return {"error": "Model not loaded"}

    # Clean and validate input text
    text = (inp.text or "").strip()
    if not text:
        return {"error": "Empty text"}

    # Run inference on the input text
    # Model output format: [{'label': 'LABEL_2', 'score': 0.98}]
    out = _classifier(text)[0]

    # Extract numeric label ID from format "LABEL_2" → 2
    label_id = int(out["label"].split("_")[-1])

    # Return human-readable sentiment label with confidence score
    return {
        "label": LABELS.get(label_id, out["label"]),
        "confidence": float(out["score"])
    }


# === Frontend UI ===
@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the interactive frontend UI.
    
    Provides a single-page web application that:
    - Accepts food review text input from the user
    - Sends text to /predict_spans for analysis
    - Displays sentiment predictions with interactive highlights
    - Detects and warns about sensitive information (PII)
    - Shows overall and per-segment sentiment analysis
    """
    return r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Food Review Sentiment</title>
  <style>
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
    .pos.selected { outline: 2px solid rgba(255,255,255,0.5); }

    .pos { padding: 2px 6px; border-radius: 8px; }
    .pos.positive { background: rgba(0,255,0,0.15); border: 1px solid rgba(0,255,0,0.35); }
    .pos.neutral  { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18); }
    .pos.negative { background: rgba(255,0,0,0.15); border: 1px solid rgba(255,0,0,0.35); }
  </style>
  <style>
  .warnbox {
    margin-top: 10px;
    padding: 10px 12px;
    border-radius: 12px;
    border: 1px solid rgba(255, 180, 180, 0.45);
    background: rgba(255, 0, 0, 0.10);
  }
  .warnbox.ok {
    border-color: rgba(182, 255, 204, 0.35);
    background: rgba(0, 255, 0, 0.08);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    margin: 6px 6px 0 0;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.06);
    font-size: 12px;
  }
  .chip b { font-weight: 700; }

  /* Optional highlight preview */
  #preview {
    margin-top: 10px;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    white-space: pre-wrap;
    line-height: 1.7;
  }
  mark.pii {
    padding: 2px 4px;
    border-radius: 6px;
    background: rgba(255, 180, 180, 0.25);
    border: 1px solid rgba(255, 180, 180, 0.35);
    color: inherit;
  }

  /* History Items Styling */
  .history-item {
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid rgba(100, 150, 255, 0.3);
    background: rgba(100, 150, 255, 0.1);
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 13px;
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .history-item:hover {
    background: rgba(100, 150, 255, 0.2);
    border-color: rgba(100, 150, 255, 0.6);
    transform: translateY(-2px);
  }

  #historySection {
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding-top: 16px;
  }

  #history:empty::after {
    content: "No history yet. Make predictions to see them here.";
    color: rgba(232, 236, 255, 0.5);
    font-size: 12px;
  }

  /* Segmentation Option Styling */
  .segment-option {
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 6px;
    border: 1px solid rgba(150, 180, 255, 0.4);
    background: rgba(150, 180, 255, 0.05);
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 13px;
  }

  .segment-option:hover {
    background: rgba(150, 180, 255, 0.15);
    border-color: rgba(150, 180, 255, 0.7);
  }

  .segment-option.selected {
    background: rgba(150, 180, 255, 0.25);
    border-color: rgba(150, 180, 255, 0.9);
  }

  .segment-badge {
    display: inline-block;
    background: rgba(150, 180, 255, 0.3);
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 11px;
    margin-left: 8px;
  }
</style>

</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Food Review Sentiment</h1>
      <p>Type a review and hit <b>Predict</b>. Backend: <code>/predict_spans</code></p>
      <div id="piiBox" class="warnbox ok">
        <b id="piiTitle">No sensitive info detected</b>
        <div id="piiChips"></div>
      </div>

      <div id="preview" style="display:none;"></div>

      <textarea id="txt" placeholder="e.g. The laksa was amazing but the service was slow."></textarea>

      <div style="margin-top:12px;">
        <button id="btn">Predict</button>
        <button id="demo">Load sample</button>
        <button id="segmentBtn">Segment (remove spaces)</button>
        <span class="pill">Status: <span id="status">Ready</span></span>
      </div>

      <!-- Segmentation Results Section -->
      <div id="segmentSection" style="margin-top:12px; display:none;">
        <div style="padding:10px; border-radius:8px; border:1px solid rgba(150,180,255,0.3); background:rgba(150,180,255,0.08);">
          <b>Segmentation Results:</b>
          <div id="segmentResults" style="margin-top:8px;"></div>
        </div>
      </div>

      <div style="margin-top:12px;" class="pill">
        Overall: <b id="overallLabel">—</b>
        Confidence: <b id="overallConf">—</b>
      </div>
<div id="chunkInfo" class="pill" style="margin-top:12px; display:none;"></div>

      <div id="hl" style="margin-top:12px; line-height:1.8;"></div>
      <div id="msg" style="margin-top:12px;"></div>

      <!-- Sentiment Extremes Section -->
      <div id="extremesSection" style="margin-top:20px; display:none;">
        <h3 style="margin-top:0; margin-bottom:12px;">Sentiment Extremes</h3>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
          <div>
            <h4 style="margin-top:0; color:#b6ffcc;">Most Positive</h4>
            <div id="mostPositive" style="padding:10px; border-radius:8px; border:1px solid rgba(0,255,0,0.3); background:rgba(0,255,0,0.08);"></div>
          </div>
          <div>
            <h4 style="margin-top:0; color:#ffb4b4;">Most Negative</h4>
            <div id="mostNegative" style="padding:10px; border-radius:8px; border:1px solid rgba(255,0,0,0.3); background:rgba(255,0,0,0.08);"></div>
          </div>
        </div>
      </div>
      
      <!-- Prediction History Section -->
      <div id="historySection" style="margin-top:20px;">
        <h3 style="margin-top:0; margin-bottom:10px;">History</h3>
        <div id="history" style="display:flex; flex-wrap:wrap; gap:8px;"></div>
      </div>
    </div>
  </div>

<script>
  // ===== DOM Element References =====
  // Cache DOM elements for efficient access throughout the script
  const txt = document.getElementById('txt');
  const btn = document.getElementById('btn');
  const demo = document.getElementById('demo');
  const statusEl = document.getElementById('status');
  const overallLabelEl = document.getElementById('overallLabel');
  const overallConfEl = document.getElementById('overallConf');
  const msgEl = document.getElementById('msg');
  const hlEl = document.getElementById('hl');
  const chunkInfoEl = document.getElementById('chunkInfo');
  const historyEl = document.getElementById('history');
  const extremesSection = document.getElementById('extremesSection');
  const mostPositiveEl = document.getElementById('mostPositive');
  const mostNegativeEl = document.getElementById('mostNegative');
  const segmentBtn = document.getElementById('segmentBtn');
  const segmentSection = document.getElementById('segmentSection');
  const segmentResults = document.getElementById('segmentResults');

  // ===== Prediction History Management =====
  // Array to store previous predictions for quick access and re-analysis
  let predictionHistory = [];

  // ===== Event Listeners: Highlight Selection =====
  // Allow user to click on highlighted spans to view detailed segment information
  hlEl.addEventListener('click', (e) => {
    const span = e.target.closest('.pos');
    if (!span) return;

    // Remove previous selection
    hlEl.querySelectorAll('.pos.selected').forEach(x => x.classList.remove('selected'));
    span.classList.add('selected');

    // Extract and display segment details
    const label = span.dataset.label;
    const conf = span.dataset.conf;
    const chunk = span.dataset.chunk;

    chunkInfoEl.style.display = "inline-flex";
    chunkInfoEl.innerHTML = `Chunk: <b>${chunk}</b> | Sentiment: <b>${label}</b> | Confidence: <b>${conf}%</b>`;
  });

  // ===== Utility Functions =====
  /**
   * Update the status message displayed to the user
   * @param {string} text - Status message
   * @param {string} cls - CSS class for styling ("ok" or "err")
   */
  function setStatus(text, cls="") {
    statusEl.textContent = text;
    statusEl.className = cls;
  }

  /**
   * Display the most positive and most negative chunks
   * Sorts spans by sentiment and shows the extremes
   * @param {Array} spans - Array of span predictions
   */
  function displaySentimentExtremes(spans) {
    if (!spans || spans.length === 0) {
      extremesSection.style.display = "none";
      return;
    }

    // Find most positive (highest confidence positive)
    const positiveSpans = spans.filter(s => s.label === "positive").sort((a, b) => b.confidence - a.confidence);
    
    // Find most negative (highest confidence negative)
    const negativeSpans = spans.filter(s => s.label === "negative").sort((a, b) => b.confidence - a.confidence);

    let hasExtremes = false;

    // Display most positive
    if (positiveSpans.length > 0) {
      const best = positiveSpans[0];
      mostPositiveEl.innerHTML = `
        <div style="margin-bottom:8px;">
          <strong>${escapeHtml(best.text)}</strong>
        </div>
        <div style="font-size:12px; color:rgba(232,236,255,0.7);">
          Confidence: <strong>${(best.confidence * 100).toFixed(1)}%</strong>
        </div>
      `;
      hasExtremes = true;
    } else {
      mostPositiveEl.textContent = "No positive sentiments detected.";
    }

    // Display most negative
    if (negativeSpans.length > 0) {
      const worst = negativeSpans[0];
      mostNegativeEl.innerHTML = `
        <div style="margin-bottom:8px;">
          <strong>${escapeHtml(worst.text)}</strong>
        </div>
        <div style="font-size:12px; color:rgba(232,236,255,0.7);">
          Confidence: <strong>${(worst.confidence * 100).toFixed(1)}%</strong>
        </div>
      `;
      hasExtremes = true;
    } else {
      mostNegativeEl.textContent = "No negative sentiments detected.";
    }

    // Show extremes section if we found any
    extremesSection.style.display = hasExtremes ? "block" : "none";
  }

  /**
   * Add a prediction to the history
   * @param {string} text - The predicted text
   */
  function addToHistory(text) {
    // Avoid adding duplicate consecutive entries
    if (predictionHistory.length > 0 && predictionHistory[predictionHistory.length - 1].text === text) {
      return;
    }
    
    predictionHistory.push({ text: text, timestamp: new Date() });
    
    // Limit history to last 20 items to avoid memory issues
    if (predictionHistory.length > 20) {
      predictionHistory.shift();
    }
    
    renderHistory();
  }

  /**
   * Render the prediction history as clickable items
   * Allows users to quickly re-analyze previous texts
   */
  function renderHistory() {
    historyEl.innerHTML = "";
    
    // Display history items in reverse order (most recent first)
    for (let i = predictionHistory.length - 1; i >= 0; i--) {
      const item = predictionHistory[i];
      const historyItem = document.createElement("div");
      historyItem.className = "history-item";
      historyItem.textContent = item.text.length > 50 
        ? item.text.substring(0, 50) + "..." 
        : item.text;
      historyItem.title = item.text; // Full text on hover
      
      // Click to load history item back into text area
      historyItem.addEventListener("click", () => {
        txt.value = item.text;
        updatePIIUI();
        txt.focus();
      });
      
      historyEl.appendChild(historyItem);
    }
  }

  /**
   * Escape HTML special characters to prevent XSS
   * @param {string} s - String to escape
   * @returns {string} Escaped HTML string
   */
  function escapeHtml(s) {
    return (s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }

  /**
   * Render highlighted spans with sentiment labels
   * Builds HTML showing original text with colored sentiment predictions
   * @param {string} fullText - Original input text
   * @param {Array} spans - Array of span predictions with label, start, end, confidence
   * @returns {string} HTML string with styled spans
   */
  function renderHighlights(fullText, spans) {
    spans.sort((a,b) => a.start - b.start);
    let out = "";
    let i = 0;

    for (const s of spans) {
      // Add text before this span
      out += escapeHtml(fullText.slice(i, s.start));

      // Create colored span for this segment
      const chunk = fullText.slice(s.start, s.end);
      const pct = (s.confidence * 100).toFixed(2);

      out += `<span class="pos ${s.label}"
                    data-label="${escapeHtml(s.label)}"
                    data-conf="${pct}"
                    data-chunk="${escapeHtml(chunk)}"
                    title="Click for details">${escapeHtml(chunk)}</span>`;
      i = s.end;
    }

    // Add remaining text after last span
    out += escapeHtml(fullText.slice(i));
    return out;
  }

  // ===== Event Listeners: Sample Text Button =====
  // Load demo text for easy testing
  demo.addEventListener('click', () => {
    txt.value = "The food is really good but it is a bit salty.";
    updatePIIUI();
  });

  // ===== Event Listeners: Segment Button =====
  // Handle word segmentation for text without spaces
  segmentBtn.addEventListener('click', async () => {
    const text = (txt.value || "").trim();
    segmentSection.style.display = "none";
    segmentResults.innerHTML = "";

    if (!text) {
      msgEl.textContent = "Please enter some text to segment.";
      msgEl.className = "err";
      return;
    }

    setStatus("Segmenting...");
    try {
      // Send text to backend for segmentation
      const res = await fetch('/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "Request failed");

      // Display segmentation results
      if (data.segmentations && data.segmentations.length > 0) {
        displaySegmentationResults(data.original, data.segmentations);
        setStatus("Done", "ok");
      } else {
        msgEl.textContent = "No valid segmentation found for this text.";
        msgEl.className = "err";
        setStatus("No match", "err");
      }
    } catch (e) {
      setStatus("Error", "err");
      msgEl.textContent = e.message;
      msgEl.className = "err";
    }
  });

  /**
   * Display all possible segmentation options
   * Allows user to click on a segmentation to load it into the text area
   * @param {string} original - Original text without spaces
   * @param {Array} segmentations - Array of possible segmentations
   */
  function displaySegmentationResults(original, segmentations) {
    segmentResults.innerHTML = "";
    
    if (segmentations.length === 0) {
      segmentResults.innerHTML = "<p style='color:rgba(232,236,255,0.5);'>No valid segmentations found.</p>";
      return;
    }

    // Show count
    const countDiv = document.createElement("div");
    countDiv.style.cssText = "margin-bottom:12px; color:rgba(232,236,255,0.7); font-size:12px;";
    countDiv.textContent = `Found ${segmentations.length} possible segmentation${segmentations.length !== 1 ? 's' : ''}:`;
    segmentResults.appendChild(countDiv);

    // Display each segmentation
    for (let i = 0; i < segmentations.length; i++) {
      const seg = segmentations[i];
      const wordCount = seg.split(' ').length;
      
      const option = document.createElement("div");
      option.className = "segment-option";
      option.innerHTML = `
        <strong>${escapeHtml(seg)}</strong>
        <span class="segment-badge">${wordCount} word${wordCount !== 1 ? 's' : ''}</span>
      `;
      
      // Click to load this segmentation into text area
      option.addEventListener("click", () => {
        txt.value = seg;
        updatePIIUI();
        
        // Highlight selected option
        document.querySelectorAll('.segment-option').forEach(opt => opt.classList.remove('selected'));
        option.classList.add('selected');
      });
      
      segmentResults.appendChild(option);
    }

    segmentSection.style.display = "block";
  }

  // ===== Event Listeners: Predict Button =====
  // Main prediction handler - validates input, checks for PII, and calls API
  btn.addEventListener('click', async () => {
    const text = (txt.value || "").trim();
    msgEl.textContent = "";
    msgEl.className = "";
    hlEl.innerHTML = "";
    overallLabelEl.textContent = "—";
    overallConfEl.textContent = "—";
    
    // Check for sensitive information before sending to backend
    const piiMatches = findMatches(text);
    if (piiMatches.length) {
      msgEl.textContent = "Please remove sensitive info before submitting.";
      msgEl.className = "err";
      setStatus("Blocked", "err");
      return;
    }

    if (!text) {
      msgEl.textContent = "Please enter some text.";
      msgEl.className = "err";
      return;
    }

    setStatus("Predicting...");
    try {
      // Send text to backend for sentiment prediction
      const res = await fetch('/predict_spans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "Request failed");

      // Display overall sentiment
      overallLabelEl.textContent = data.overall.label;
      overallConfEl.textContent = (data.overall.confidence * 100).toFixed(2) + "%";

      // Display highlighted spans with sentiment colors
      hlEl.innerHTML = renderHighlights(text, data.spans);
      setStatus("Done", "ok");
      
      // Display sentiment extremes (most positive and most negative)
      displaySentimentExtremes(data.spans);
      
      // Add this prediction to history
      addToHistory(text);
    } catch (e) {
      setStatus("Error", "err");
      msgEl.textContent = e.message;
      msgEl.className = "err";
    }
  });

  // ===== PII Detection =====
  // Offline patterns for detecting sensitive information
  // These patterns help prevent submission of personal data
  const piiBox = document.getElementById("piiBox");
  const piiTitle = document.getElementById("piiTitle");
  const piiChips = document.getElementById("piiChips");
  const preview = document.getElementById("preview");

  // Regex patterns to detect common types of sensitive information
  const SENSITIVE_PATTERNS = [
    {
      type: "PHONE",
      label: "Phone number",
      re: /\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b/g
    },
    {
      type: "EMAIL",
      label: "Email address",
      re: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi
    },
    {
      type: "CREDIT_CARD",
      label: "Credit card-like number",
      re: /\b(?:\d[ -]*?){13,16}\b/g
    },
    {
      type: "NRIC_SG",
      label: "SG NRIC/FIN",
      re: /\b[STFG]\d{7}[A-Z]\b/gi
    },
    {
      type: "ADDRESS_HINT",
      label: "Address hint",
      re: /\b(blk|block|street|st|road|rd|ave|avenue|lane|ln|drive|dr|unit|#)\b/gi
    }
  ];

  // Keyword patterns for banking and security-related mentions
  const SENSITIVE_KEYWORDS = [
    { type: "PASSWORD", label: "Password mention", re: /\b(password|passcode|otp|2fa|pin)\b/gi },
    { type: "BANK", label: "Banking mention", re: /\b(bank account|acct|account number|routing|swift)\b/gi },
  ];

  /**
   * Find all sensitive information matches in the text
   * Uses regex patterns to detect PII
   * @param {string} text - Text to scan
   * @returns {Array} Array of match objects with position and type info
   */
  function findMatches(text) {
    const matches = [];

    // Check regex patterns
    for (const p of SENSITIVE_PATTERNS) {
      let m;
      while ((m = p.re.exec(text)) !== null) {
        matches.push({
          type: p.type,
          label: p.label,
          value: m[0],
          start: m.index,
          end: m.index + m[0].length
        });
        // Prevent infinite loops for 0-length matches
        if (m.index === p.re.lastIndex) p.re.lastIndex++;
      }
    }

    // Check keyword patterns
    for (const k of SENSITIVE_KEYWORDS) {
      let m;
      while ((m = k.re.exec(text)) !== null) {
        matches.push({
          type: k.type,
          label: k.label,
          value: m[0],
          start: m.index,
          end: m.index + m[0].length
        });
        if (m.index === k.re.lastIndex) k.re.lastIndex++;
      }
    }

    // Sort matches and handle overlaps
    matches.sort((a,b) => a.start - b.start || b.end - a.end);

    return matches;
  }

  /**
   * Render a preview of text with PII matches highlighted
   * Shows sensitive information markers in the preview section
   * @param {string} text - Original text to preview
   * @param {Array} matches - Array of PII match objects
   * @returns {string} HTML string with marked PII sections
   */
  function renderPreview(text, matches) {
    if (!matches.length) return escapeHtml(text);

    // Remove overlaps - keep first match when overlaps occur
    const filtered = [];
    let lastEnd = -1;
    for (const m of matches) {
      if (m.start >= lastEnd) {
        filtered.push(m);
        lastEnd = m.end;
      }
    }

    // Build HTML with marked PII sections
    let out = "";
    let i = 0;
    for (const m of filtered) {
      out += escapeHtml(text.slice(i, m.start));
      out += `<mark class="pii" title="${escapeHtml(m.label)}">${escapeHtml(text.slice(m.start, m.end))}</mark>`;
      i = m.end;
    }
    out += escapeHtml(text.slice(i));
    return out;
  }

  /**
   * Update the PII warning UI based on text content
   * Shows warnings if sensitive information is detected
   */
  function updatePIIUI() {
    const text = (txt.value || "");
    const matches = findMatches(text);

    // Count matches by type for summary display
    const counts = {};
    for (const m of matches) counts[m.type] = (counts[m.type] || 0) + 1;

    piiChips.innerHTML = "";
    if (!matches.length) {
      // No sensitive info found - show green OK status
      piiBox.classList.add("ok");
      piiTitle.textContent = "No sensitive info detected";
      preview.style.display = "none";
      preview.innerHTML = "";
      return;
    }

    // Sensitive info detected - show warning
    piiBox.classList.remove("ok");
    piiTitle.textContent = "Possible sensitive info detected";
    for (const [type, count] of Object.entries(counts)) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerHTML = `<b>${escapeHtml(type)}</b> × ${count}`;
      piiChips.appendChild(chip);
    }

    // Show preview with highlights (optional)
    preview.style.display = "block";
    preview.innerHTML = renderPreview(text, matches);
  }

  // Monitor text input with debouncing to avoid excessive checks
  let piiTimer = null;
  txt.addEventListener("input", () => {
    clearTimeout(piiTimer);
    piiTimer = setTimeout(updatePIIUI, 120);
  });

  // Initialize PII UI on page load
  updatePIIUI();


</script>
</body>
</html>

"""
