"""Food Review Sentiment Analysis API

A FastAPI application that classifies food reviews into positive, neutral, or negative sentiments.
Features:
- Real-time sentiment prediction on user input with Server-Sent Events (SSE) streaming
- Span-level sentiment analysis for detailed insights
- Progressive result display as spans are analyzed
- Frontend UI with PII detection and sensitivity warnings
- Pre-trained HuggingFace transformer model
"""

# === External Dependencies ===
import csv
import io
import json
import re
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

# === Standard Library ===
from typing import List, Dict

# === Internal Modules ===
from config import LABELS
from text_utils import split_with_offsets
from segmentation import segment_text_all
from model import load_model, get_classifier, is_model_loaded

# === Concurrency and DB Modules ===
from starlette.concurrency import run_in_threadpool
from concurrency import tracker, now_ms
from db import init_db, log_prediction, fetch_predictions, fetch_latency_vs_length, fetch_word_frequencies
# === FastAPI Application ===
app = FastAPI(
    title="Food Review Sentiment App",
    description="Classifies food reviews by sentiment (positive/neutral/negative)",
    version="1.0.0",
    root_path="/inf",
    docs_url="/docs",
    openapi_url="/openapi.json",
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


# === Request/Response Schemas ===
class PredictIn(BaseModel):
    """Request schema for  /predict_spans endpoints.
    
    Expects a JSON body containing the review text to classify.
    
    Example:
        {"text": "The food was amazing but service was slow."}
    """
    text: str

@app.middleware("http")
async def inflight_http_middleware(request, call_next):
    await tracker.http_enter()
    try:
        response = await call_next(request)
        return response
    finally:
        await tracker.http_exit()
@app.on_event("startup")
def startup():
    load_model()
    init_db()

# === PII (Personally Identifiable Information) Detection Patterns ===
# These regex patterns identify common types of sensitive information
# to prevent them from being suggested in autocomplete or stored inappropriately

# Email pattern: Matches standard email addresses (e.g., user@example.com)
# Case-insensitive to catch variations like User@Example.COM
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)

# Phone number pattern: Matches various phone formats including:
# - International format: +1-555-123-4567
# - With parentheses: (555) 123-4567
# - With spaces/dashes: 555 123 4567 or 555-123-4567
# - Plain format: 5551234567
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b")

# Credit card pattern: Matches sequences of 13-16 digits with optional spaces/dashes
# Catches common card formats (Visa, Mastercard, Amex, etc.)
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

# Singapore NRIC/FIN pattern: Matches Singapore National Registration Identity Card numbers
# Format: S/T/F/G followed by 7 digits and a checksum letter (e.g., S1234567A)
# S/T for citizens, F/G for foreigners/permanent residents
_NRIC_RE = re.compile(r"\b[STFG]\d{7}[A-Z]\b", re.I)

def _looks_sensitive(token: str) -> bool:
    """Check if a token contains sensitive information (PII).
    
    This server-side validation prevents sensitive data from being:
    - Suggested in autocomplete results
    - Stored in word frequency databases
    - Exposed in any way through the API
    
    Args:
        token: The word/text to check for sensitive patterns
    
    Returns:
        bool: True if the token appears to contain PII, False otherwise
    
    Security Note:
        This is a first line of defense. Users should not enter real PII,
        but this helps prevent accidental exposure if they do.
    """
    # Treat empty/null tokens as potentially sensitive (err on side of caution)
    if not token:
        return True
    
    t = token.strip()
    
    # Check against all PII patterns
    if _EMAIL_RE.search(t): return True      # Email addresses
    if _PHONE_RE.search(t): return True      # Phone numbers
    if _CC_RE.search(t): return True         # Credit card numbers
    if _NRIC_RE.search(t): return True       # Singapore NRIC/FIN
    
    # No sensitive patterns detected
    return False

@app.get("/autocomplete")
async def autocomplete(
    q: str = Query("", min_length=1, max_length=40),
    limit: int = Query(8, ge=1, le=20),
    window: int = Query(500, ge=50, le=5000),
):
    """
    Autocomplete suggestions based on historical word frequencies
    computed from recent predictions stored in DB.
    
    Args:
        q: Query prefix to match (user's partial input)
        limit: Maximum number of suggestions to return (1-20)
        window: Number of recent predictions to analyze (50-5000)
    
    Returns:
        JSON with suggestions list, sorted by frequency and length
    """
    # Normalize the query prefix to lowercase for case-insensitive matching
    prefix = (q or "").strip().lower()
    if not prefix:
        return {"suggestions": []}

    # Fetch word frequencies from recent predictions (runs in thread pool to avoid blocking)
    # Returns dict[word] = count for the most common words
    word_freq = await run_in_threadpool(lambda: fetch_word_frequencies(limit=window))
    if not word_freq:
        return {"suggestions": []}

    # Filter words matching the prefix and apply safety checks
    items = []
    for w, c in word_freq.items():
        # Skip empty or null words
        if not w:
            continue
        ww = str(w).strip()
        
        # Skip single-character words (too short for meaningful suggestions)
        if len(ww) < 2:
            continue
        
        # Skip words that appear to contain sensitive information (PII)
        # This prevents autocomplete from suggesting emails, phone numbers, etc.
        # that may have been entered in previous reviews
        if _looks_sensitive(ww):
            continue
        
        # Include only words that start with the user's prefix
        if ww.lower().startswith(prefix):
            items.append((ww, int(c)))

    # Sort by frequency (descending), then by word length (shorter first), then alphabetically
    # This prioritizes common, concise words for better UX
    items.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
    
    # Return top N suggestions (extract just the words, not frequencies)
    suggestions = [w for (w, _) in items[:limit]]

    return {"suggestions": suggestions}

@app.get("/db/latency_vs_length")
async def db_latency_vs_length(limit: int = Query(500, ge=10, le=10000)):
    # latest N points; frontend will sort by text_len for plotting
    return {"items": fetch_latency_vs_length(limit=limit)}

@app.get("/word_cloud")
async def word_cloud(limit: int = Query(500, ge=10, le=5000)):
    """Generate word frequency data for word cloud visualization.
    
    Analyzes recent predictions and returns word frequencies suitable
    for rendering a word cloud visualization on the frontend.
    
    Args:
        limit: Maximum number of predictions to analyze (default: 500)
    
    Returns:
        dict: {
            "words": List[{"text": str, "frequency": int, "size": float}]
        }
    """
    word_freq = await run_in_threadpool(lambda: fetch_word_frequencies(limit=limit))
    
    if not word_freq:
        return {"words": []}
    
    # Convert to sorted list and compute sizing
    items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Get top words (limit for performance)
    top_words = items[:100]
    
    if not top_words:
        return {"words": []}
    
    # Normalize frequencies to 0.5-2.0 for sizing
    min_freq = min(f for _, f in top_words)
    max_freq = max(f for _, f in top_words)
    freq_range = max_freq - min_freq or 1
    
    words = []
    for word, freq in top_words:
        normalized_size = 0.5 + 1.5 * ((freq - min_freq) / freq_range)
        words.append({
            "text": word,
            "frequency": freq,
            "size": normalized_size
        })
    
    return {"words": words}

@app.get("/db/predictions")
async def db_predictions(limit: int = Query(50, ge=1, le=500)):
    # returns latest N rows
    return {"items": fetch_predictions(limit=limit)}

@app.get("/db/predictions.csv")
async def db_predictions_csv(limit: int = Query(500, ge=1, le=5000)):
    items = fetch_predictions(limit=limit)

    # Convert to CSV in-memory
    buf = io.StringIO()
    if items:
        fieldnames = list(items[0].keys())
    else:
        fieldnames = [
            "id","ts_ms","endpoint","text_len","text_sha256","label","confidence",
            "latency_ms","http_in_flight","predict_in_flight","predict_waiting"
        ]

    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in items:
        writer.writerow(row)

    buf.seek(0)

    headers = {
        "Content-Disposition": 'attachment; filename="predictions.csv"'
    }
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)


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
        "model_loaded": is_model_loaded()
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

@app.get("/server_load")
async def server_load():
    snap = await tracker.snapshot()
    return {
        "http_in_flight": snap.http_in_flight,
        "predict_in_flight": snap.predict_in_flight,
        "predict_waiting": snap.predict_waiting,
    }

@app.get("/predict_spans_stream")
async def predict_spans_stream(text: str = Query("", min_length=1)):
    """
    Server-Sent Events (SSE) streaming endpoint for real-time predictions.
    
    Streams span-by-span sentiment predictions as they are computed,
    allowing the frontend to display results progressively for better UX.
    
    Args:
        text: The review text to analyze (query parameter)
    
    Returns:
        StreamingResponse with text/event-stream content type
    
    Events:
        - start: Initial event with total span count
        - span: Individual span prediction result (sent for each span)
        - done: Final event with overall prediction and metadata
        - error: Sent if an error occurs during processing
    
    Example:
        GET /predict_spans_stream?text=The%20food%20was%20great
    """
    classifier = get_classifier()
    if classifier is None:
        # Return error as SSE event
        async def error_gen():
            yield f"event: error\ndata: {{\"error\": \"Model not loaded\"}}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    text = (text or "").strip()
    if not text:
        # Return error as SSE event
        async def error_gen():
            yield f"event: error\ndata: {{\"error\": \"Empty text\"}}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    async def gen():
        """Generator function that yields SSE-formatted prediction events."""
        req_start = now_ms()
        
        # Acquire prediction semaphore to track concurrency
        await tracker.predict_acquire()
        try:
            # Split text into semantic spans (sentences/clauses)
            spans = split_with_offsets(text)

            # Send start event with span count so frontend can prepare UI
            yield f"event: start\ndata: {json.dumps({'n_spans': len(spans)})}\n\n"

            # Process and stream each span as it completes
            results = []
            for i, s in enumerate(spans):
                # Run prediction in thread pool to avoid blocking async loop
                out = await run_in_threadpool(
                    lambda t=s["text"]: classifier(t, truncation=True, max_length=512)[0]
                )
                
                # Parse label and build result
                label_id = int(out["label"].split("_")[-1])
                span_result = {
                    "i": i,                                      # Span index
                    "start": s["start"],                         # Character start position
                    "end": s["end"],                             # Character end position
                    "text": s["text"],                           # Span text content
                    "label": LABELS.get(label_id, out["label"]),# Sentiment label
                    "confidence": float(out["score"]),          # Confidence score
                }
                results.append(span_result)
                
                # Stream this span immediately to frontend
                yield f"event: span\ndata: {json.dumps(span_result)}\n\n"

            # Compute overall sentiment for the entire text
            overall = await run_in_threadpool(
                lambda: classifier(text, truncation=True, max_length=512)[0]
            )
            overall_id = int(overall["label"].split("_")[-1])
            overall_label = LABELS.get(overall_id, overall["label"])
            overall_conf = float(overall["score"])

            # Calculate metrics
            snap = await tracker.snapshot()
            latency_ms = now_ms() - req_start

            # Log to database (in background thread)
            await run_in_threadpool(
                lambda: log_prediction(
                    ts_ms=req_start,
                    endpoint="/predict_spans_stream",
                    text=text,
                    label=overall_label,
                    confidence=overall_conf,
                    spans=results,
                    latency_ms=latency_ms,
                    http_in_flight=snap.http_in_flight,
                    predict_in_flight=snap.predict_in_flight,
                    predict_waiting=snap.predict_waiting,
                )
            )

            # Send final event with overall results and metadata
            done_data = {
                "overall": {
                    "label": overall_label,
                    "confidence": overall_conf
                },
                "meta": {
                    "latency_ms": latency_ms,
                    "http_in_flight": snap.http_in_flight,
                    "predict_in_flight": snap.predict_in_flight,
                    "predict_waiting": snap.predict_waiting,
                }
            }
            yield f"event: done\ndata: {json.dumps(done_data)}\n\n"
            
        except Exception as e:
            # Stream error event if something goes wrong
            yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            # Always release the semaphore
            await tracker.predict_release()

    # Return streaming response with proper headers
    return StreamingResponse(
        gen(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind proxy
        }
    )


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
      <p>Type a review and hit <b>Predict</b>. Results stream in real-time. Backend: <code>/predict_spans_stream</code></p>
      <div id="piiBox" class="warnbox ok">
        <b id="piiTitle">No sensitive info detected</b>
        <div id="piiChips"></div>
      </div>

      <div id="preview" style="display:none;"></div>

      <div style="position:relative;">
        <textarea id="txt" placeholder="e.g. The laksa was amazing but the service was slow."></textarea>
        <div id="acBox" style="
          position:absolute;
          left:0; right:0;
          top:100%;
          margin-top:6px;
          background:#0e1530;
          border:1px solid #2a3a72;
          border-radius:12px;
          display:none;
          z-index:9999;
          overflow:hidden;
        "></div>
      </div>

      <div style="margin-top:12px;">
        <button id="btn">Predict</button>
        <button id="demo">Load sample</button>
        <button id="segmentBtn">Segment (remove spaces)</button>
        <button id="dbBtn">DB History</button>
        <button id="wordCloudBtn">Word Cloud</button>

        <span class="pill">Status: <span id="status">Ready</span></span>
        <span class="pill">Server: <span id="serverLoad">—</span></span>
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
<!-- DB Modal -->
<div id="dbModal" style="display:none; position:fixed; inset:0; z-index:9999;">
  <div id="dbBackdrop" style="position:absolute; inset:0; background:rgba(0,0,0,0.55);"></div>

  <div style="position:relative; max-width:980px; margin:6vh auto; background:#121a33; border:1px solid #22305d; border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,.35);">
    <div style="display:flex; justify-content:space-between; align-items:center; padding:14px 16px; border-bottom:1px solid rgba(255,255,255,0.10);">
      <div>
        <b>SQLite Prediction History</b>
        <div style="font-size:12px; color:rgba(232,236,255,0.7);">Latest entries from <code>predictions.sqlite</code></div>
      </div>
      <div style="display:flex; gap:8px; align-items:center;">
        <input id="dbSearch" placeholder="Search (endpoint/label/id/hash)…"
  style="width:260px; border-radius:10px; padding:8px 10px; border:1px solid #2a3a72; background:#0e1530; color:#e8ecff;" />

        <select id="dbLimit" style="border-radius:10px; padding:8px; border:1px solid #2a3a72; background:#0e1530; color:#e8ecff;">
          <option value="25">25</option>
          <option value="50" selected>50</option>
          <option value="100">100</option>
          <option value="200">200</option>
        </select>
        <button id="dbRefresh">Refresh</button>
        <button id="dbClose">Close</button>
        <button id="dbExport">Export CSV</button>

      </div>
    </div>

    <div style="padding:14px 16px;">
      <div id="dbErr" class="err" style="margin-bottom:10px;"></div>
      <div style="overflow:auto; max-height:65vh; border:1px solid rgba(255,255,255,0.10); border-radius:12px;">
        
        <table style="width:100%; border-collapse:collapse; font-size:12px;">
          <thead style="position:sticky; top:0; background:#0e1530; border-bottom:1px solid rgba(255,255,255,0.10);">
            <tr>
              <th style="text-align:left; padding:10px;">id</th>
              <th style="text-align:left; padding:10px;">time</th>
              <th style="text-align:left; padding:10px;">endpoint</th>
              <th style="text-align:left; padding:10px;">text_len</th>
              <th style="text-align:left; padding:10px;">text_sha256</th>
              <th style="text-align:left; padding:10px;">label</th>
              <th style="text-align:left; padding:10px;">conf</th>
              <th style="text-align:left; padding:10px;">latency_ms</th>
              <th style="text-align:left; padding:10px;">http</th>
              <th style="text-align:left; padding:10px;">predict</th>
              <th style="text-align:left; padding:10px;">queue</th>
            </tr>
          </thead>
          <tbody id="dbTbody"></tbody>
        </table>
        <div style="margin-top:14px; padding:12px; border-radius:12px; border:1px solid rgba(255,255,255,0.10); background:rgba(255,255,255,0.04);">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;">
    <b>Latency vs Text Length</b>
    <div style="display:flex; gap:8px; align-items:center;">
      <button id="dbPlotBtn">Plot</button>
      <span style="font-size:12px; color:rgba(232,236,255,0.7);">Uses latest rows from DB</span>
    </div>
  </div>

  <canvas id="latCanvas" width="900" height="260"
    style="margin-top:10px; width:100%; height:260px; border-radius:10px; border:1px solid rgba(255,255,255,0.10); background:rgba(14,21,48,0.85);">
  </canvas>

  <div id="latLegend" style="margin-top:8px; font-size:12px; color:rgba(232,236,255,0.7);"></div>
</div>

      </div>

      <div style="margin-top:10px; font-size:12px; color:rgba(232,236,255,0.65);">
        Note: raw text isn’t stored (only length + SHA256) to reduce PII risk.
      </div>
    </div>
  </div>
</div>

<!-- Word Cloud Modal -->
<div id="wordCloudModal" style="display:none; position:fixed; inset:0; z-index:9999;">
  <div id="wcBackdrop" style="position:absolute; inset:0; background:rgba(0,0,0,0.55);"></div>

  <div style="position:relative; max-width:900px; margin:8vh auto; background:#121a33; border:1px solid #22305d; border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,.35);">
    <div style="display:flex; justify-content:space-between; align-items:center; padding:14px 16px; border-bottom:1px solid rgba(255,255,255,0.10);">
      <div>
        <b>Word Cloud</b>
        <div style="font-size:12px; color:rgba(232,236,255,0.7);">Word frequency analysis from recent predictions</div>
      </div>
      <div style="display:flex; gap:8px; align-items:center;">
        <select id="wcLimit" style="border-radius:10px; padding:8px; border:1px solid #2a3a72; background:#0e1530; color:#e8ecff;">
          <option value="100">100 predictions</option>
          <option value="250" selected>250 predictions</option>
          <option value="500">500 predictions</option>
          <option value="1000">1000 predictions</option>
        </select>
        <button id="wcRefresh">Refresh</button>
        <button id="wcClose">Close</button>
      </div>
    </div>

    <div style="padding:14px 16px;">
      <div id="wcErr" class="err" style="margin-bottom:10px;"></div>
      <div style="text-align:center;">
        <canvas id="wordCloudCanvas" width="860" height="440"
          style="width:100%; height:440px; border-radius:10px; border:1px solid rgba(255,255,255,0.10); background:rgba(14,21,48,0.85);">
        </canvas>
      </div>
      
      <div style="margin-top:12px; padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.10); background:rgba(255,255,255,0.04);">
        <div id="wcStats" style="font-size:12px; color:rgba(232,236,255,0.7); line-height:1.6;">
          <div>Loading statistics...</div>
        </div>
      </div>

      <div style="margin-top:10px; font-size:12px; color:rgba(232,236,255,0.65);">
        Word sizes are proportional to frequency. Larger words appear more frequently in recent predictions.
      </div>
    </div>
  </div>
</div>

<script>
  // ===== DOM Element References =====
  // Cache DOM elements for efficient access throughout the script
  const dbExport = document.getElementById('dbExport');
  const dbPlotBtn = document.getElementById('dbPlotBtn');
  const latCanvas = document.getElementById('latCanvas');
  const latLegend = document.getElementById('latLegend');
  const dbSearch = document.getElementById('dbSearch');

  const wordCloudBtn = document.getElementById('wordCloudBtn');
  const wordCloudModal = document.getElementById('wordCloudModal');
  const wcBackdrop = document.getElementById('wcBackdrop');
  const wcClose = document.getElementById('wcClose');
  const wcRefresh = document.getElementById('wcRefresh');
  const wcLimit = document.getElementById('wcLimit');
  const wordCloudCanvas = document.getElementById('wordCloudCanvas');
  const wcErr = document.getElementById('wcErr');
  const wcStats = document.getElementById('wcStats');

  const serverLoadEl = document.getElementById('serverLoad');
  const dbBtn = document.getElementById('dbBtn');
  const dbModal = document.getElementById('dbModal');
  const dbBackdrop = document.getElementById('dbBackdrop');
  const dbClose = document.getElementById('dbClose');
  const dbRefresh = document.getElementById('dbRefresh');
  const dbLimit = document.getElementById('dbLimit');
  const dbTbody = document.getElementById('dbTbody');
  const dbErr = document.getElementById('dbErr');

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
  let dbRowsCache = [];

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

function fmtTime(tsMs) {
  try {
    return new Date(tsMs).toLocaleString();
  } catch {
    return String(tsMs);
  }
}

function rowToSearchString(row) {
  // Choose fields that make sense to search
  // Include predicted label in text form for better searchability
  const labelText = {
    0: "negative",
    1: "neutral",
    2: "positive"
  }[row.label] || row.label;
  
  return [
    row.id,
    row.ts_ms,
    row.endpoint,
    row.text_len,
    row.text_sha256,
    row.label,
    labelText,           // <- Add predicted label text (negative/neutral/positive)
    row.confidence,
    row.latency_ms,
    row.http_in_flight,
    row.predict_in_flight,
    row.predict_waiting,
  ].join(" ").toLowerCase();
}

 function renderDbTable() {
    const q = (dbSearch.value || "").trim().toLowerCase();
    const rows = !q ? dbRowsCache : dbRowsCache.filter(r => rowToSearchString(r).includes(q));

    if (!rows.length) {
      dbTbody.innerHTML = `<tr><td colspan="12" style="padding:12px; color:rgba(232,236,255,0.7);">No matches.</td></tr>`;
      return;
    }

    dbTbody.innerHTML = rows.map(row => `
      <tr style="border-top:1px solid rgba(255,255,255,0.06);">
        <td style="padding:10px;">${row.id}</td>
        <td style="padding:10px; white-space:nowrap;">${escapeHtml(fmtTime(row.ts_ms))}</td>
        <td style="padding:10px;">${escapeHtml(row.endpoint)}</td>
        <td style="padding:10px;">${row.text_len}</td>
        <td style="padding:10px; font-family: ui-monospace, monospace;">
          ${escapeHtml(String(row.text_sha256).slice(0, 16))}…
        </td>
        <td style="padding:10px;">${escapeHtml(row.label)}</td>
        <td style="padding:10px;">${(Number(row.confidence) * 100).toFixed(1)}%</td>
        <td style="padding:10px;">${row.latency_ms}</td>
        <td style="padding:10px;">${row.http_in_flight}</td>
        <td style="padding:10px;">${row.predict_in_flight}</td>
        <td style="padding:10px;">${row.predict_waiting}</td>
        <td style="padding:10px; max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
          ${escapeHtml(row.text_preview || "")}
        </td>
      </tr>
    `).join("");
  }


  async function openDbModal() {
    dbModal.style.display = "block";
    dbSearch.value = "";
    await loadDbRows();     // <- important
    renderDbTable();        // <- so search works immediately
  }

function closeDbModal() {
  dbModal.style.display = "none";
}

async function plotLatencyVsLength() {
  latLegend.textContent = "Loading…";
  try {
    const limit = Number(dbLimit.value || 50);
    const res = await fetch(`./db/latency_vs_length?limit=${Math.max(50, limit)}`, { cache: 'no-store' });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || "Failed to fetch graph data");

    const items = (data.items || []).map(x => ({
      text_len: Number(x.text_len),
      latency_ms: Number(x.latency_ms),
      ts_ms: Number(x.ts_ms),
    })).filter(p => Number.isFinite(p.text_len) && Number.isFinite(p.latency_ms));

    drawLineChart(latCanvas, items);

    if (!items.length) {
      latLegend.textContent = "No data available yet.";
      return;
    }

    const avg = items.reduce((s,p)=>s+p.latency_ms,0)/items.length;
    const max = Math.max(...items.map(p=>p.latency_ms));
    latLegend.textContent = `Points: ${items.length} | Avg: ${avg.toFixed(1)} ms | Max: ${max} ms`;
  } catch (e) {
    latLegend.textContent = `Error: ${e.message || e}`;
  }
}

dbPlotBtn.addEventListener('click', plotLatencyVsLength);



async function loadDbRows() {
    dbErr.textContent = "";
    dbTbody.innerHTML = `<tr><td colspan="12" style="padding:12px; color:rgba(232,236,255,0.7);">Loading…</td></tr>`;

    try {
      const limit = Number(dbLimit.value || 50);
      const res = await fetch(`./db/predictions?limit=${limit}`, { cache: 'no-store' });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Failed to load DB rows");

      const items = data.items || [];
      dbRowsCache = items;          // <- THIS is what your search needs

      renderDbTable();              // <- render using cache + search query

    } catch (e) {
      dbErr.textContent = e.message || String(e);
      dbTbody.innerHTML = "";
      dbRowsCache = [];
    }
  }

dbExport.addEventListener('click', () => {
  const limit = Number(dbLimit.value || 50);
  // Navigating to the CSV endpoint triggers a download
  window.location.href = `/db/predictions.csv?limit=${limit}`;
});

  async function refreshServerLoad() {
  try {
    const res = await fetch('./server_load', { cache: 'no-store' });
    const data = await res.json();

    // Format: HTTP in-flight, predictions in-flight, waiting queue
    serverLoadEl.textContent =
      `HTTP: ${data.http_in_flight} | Predict: ${data.predict_in_flight} | Queue: ${data.predict_waiting}`;
  } catch (e) {
    serverLoadEl.textContent = "Unavailable";
  }
}

  // ===== Word Cloud Functions =====
  /**
   * Open the word cloud modal and load data
   */
  function openWordCloudModal() {
    wordCloudModal.style.display = "block";
    loadWordCloud();
  }

  /**
   * Close the word cloud modal
   */
  function closeWordCloudModal() {
    wordCloudModal.style.display = "none";
  }

  /**
   * Load word cloud data from backend and render visualization
   */
  async function loadWordCloud() {
    wcErr.textContent = "";
    wcStats.innerHTML = '<div>Loading...</div>';
    
    try {
      const limit = Number(wcLimit.value || 250);
      const res = await fetch(`./word_cloud?limit=${limit}`, { cache: 'no-store' });
      const data = await res.json();
      
      if (!res.ok) throw new Error(data?.error || "Failed to load word cloud");
      
      const words = data.words || [];
      
      if (words.length === 0) {
        wcStats.innerHTML = '<div style="color:rgba(232,236,255,0.5);">No words found. Make predictions first.</div>';
        return;
      }
      
      // Render word cloud on canvas
      renderWordCloud(wordCloudCanvas, words);
      
      // Display statistics
      const totalWords = words.reduce((sum, w) => sum + w.frequency, 0);
      const avgFreq = (totalWords / words.length).toFixed(2);
      
      wcStats.innerHTML = `
        <div><b>Total unique words:</b> ${words.length}</div>
        <div><b>Total word occurrences:</b> ${totalWords}</div>
        <div><b>Average frequency:</b> ${avgFreq}</div>
        <div style="margin-top:8px;"><b>Top 5 words:</b></div>
        <div style="margin-left:10px;">
          ${words.slice(0, 5).map(w => `• ${w.text}: ${w.frequency} times`).join('<br>')}
        </div>
      `;
      
    } catch (e) {
      wcErr.textContent = e.message || String(e);
      wcStats.innerHTML = '';
    }
  }

  /**
   * Render word cloud on canvas using frequency-based sizing and random positioning
   * @param {HTMLCanvasElement} canvas - Canvas element to draw on
   * @param {Array} words - Array of word objects with {text, frequency, size}
   */
  function renderWordCloud(canvas, words) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = "rgba(14,21,48,0.85)";
    ctx.fillRect(0, 0, W, H);
    
    if (!words.length) return;
    
    // Create layout array for collision detection
    const layout = [];
    
    // Sort words by frequency (largest first) for better placement
    const sortedWords = [...words].sort((a, b) => b.frequency - a.frequency);
    
    // Color palette (cycling through colors)
    const colors = [
      "rgba(182,255,204,0.9)",    // green (positive-like)
      "rgba(100,150,255,0.9)",    // blue
      "rgba(255,200,100,0.9)",    // orange
      "rgba(255,150,150,0.9)",    // red-ish (negative-like)
      "rgba(200,150,255,0.9)",    // purple
    ];
    
    let colorIndex = 0;
    
    // Place each word
    for (const word of sortedWords) {
      // Font size based on frequency (larger = more frequent)
      const fontSize = Math.floor(16 + word.size * 24);
      ctx.font = `bold ${fontSize}px system-ui`;
      ctx.fillStyle = colors[colorIndex % colors.length];
      
      // Measure text width
      const metrics = ctx.measureText(word.text);
      const textWidth = metrics.width;
      const textHeight = fontSize;
      
      // Try to find a valid position (with max attempts to avoid infinite loop)
      let placed = false;
      let attempts = 0;
      const maxAttempts = 50;
      
      while (!placed && attempts < maxAttempts) {
        // Random position
        const x = Math.random() * (W - textWidth - 20) + 10;
        const y = Math.random() * (H - textHeight - 10) + textHeight;
        
        // Check collision with existing text
        let collision = false;
        for (const existing of layout) {
          if (checkCollision(x, y, textWidth, textHeight, existing)) {
            collision = true;
            break;
          }
        }
        
        if (!collision) {
          // Place text
          ctx.fillText(word.text, x, y);
          layout.push({ x, y, width: textWidth, height: textHeight });
          placed = true;
        }
        
        attempts++;
      }
      
      colorIndex++;
    }
    
    // Draw border
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, W, H);
  }

  /**
   * Check if a new text box collides with an existing one
   * @param {number} x1 - X coordinate of new text
   * @param {number} y1 - Y coordinate of new text
   * @param {number} w1 - Width of new text
   * @param {number} h1 - Height of new text
   * @param {Object} existing - Existing text box {x, y, width, height}
   * @returns {boolean} True if collision detected
   */
  function checkCollision(x1, y1, w1, h1, existing) {
    const padding = 8;
    return !(x1 + w1 + padding < existing.x || 
             x1 - padding > existing.x + existing.width ||
             y1 + h1 + padding < existing.y ||
             y1 - padding > existing.y + existing.height);
  }

  // Event listeners for word cloud
  wordCloudBtn.addEventListener('click', openWordCloudModal);
  wcClose.addEventListener('click', closeWordCloudModal);
  wcBackdrop.addEventListener('click', closeWordCloudModal);
  wcRefresh.addEventListener('click', loadWordCloud);
  wcLimit.addEventListener('change', loadWordCloud);

  // ESC to close word cloud modal
  document.addEventListener('keydown', (e) => {
    if (e.key === "Escape" && wordCloudModal.style.display === "block") closeWordCloudModal();
  });

dbBtn.addEventListener('click', openDbModal);
dbClose.addEventListener('click', closeDbModal);
dbBackdrop.addEventListener('click', closeDbModal);
dbRefresh.addEventListener('click', loadDbRows);
dbLimit.addEventListener('change', loadDbRows);

// ESC to close
document.addEventListener('keydown', (e) => {
  if (e.key === "Escape" && dbModal.style.display === "block") closeDbModal();
});


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
      const res = await fetch('./segment', {
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
  // Main prediction handler - validates input, checks for PII, and calls streaming API
  // Uses Server-Sent Events (SSE) to show results progressively as each span is analyzed
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
    
    // Use SSE for streaming predictions - shows results as they're computed
    // This provides better UX for longer texts with multiple spans
    const es = new EventSource(`/predict_spans_stream?text=${encodeURIComponent(text)}`);
    
    // Array to accumulate span results as they stream in
    const spans = [];
    
    // Event: start - Server indicates prediction has begun
    es.addEventListener("start", (e) => {
      const data = JSON.parse(e.data);
      // Reset display for new prediction
      hlEl.innerHTML = "";
      setStatus(`Analyzing ${data.n_spans} segment${data.n_spans !== 1 ? 's' : ''}...`);
    });
    
    // Event: span - Individual span prediction received
    // This fires multiple times as each segment is analyzed
    es.addEventListener("span", (e) => {
      const s = JSON.parse(e.data);
      spans.push(s);
      
      // Re-render highlights progressively to show real-time results
      // User sees each segment appear with its sentiment color as it's processed
      hlEl.innerHTML = renderHighlights(text, spans);
      setStatus(`Processing segment ${s.i + 1}...`);
    });
    
    // Event: done - All spans processed, overall prediction complete
    es.addEventListener("done", (e) => {
      const data = JSON.parse(e.data);
      
      // Display final overall sentiment with confidence score
      overallLabelEl.textContent = data.overall.label;
      overallConfEl.textContent = (data.overall.confidence * 100).toFixed(2) + "%";
      
      // Update status with completion indicator
      setStatus("Done", "ok");
      
      // Display sentiment extremes (most positive and most negative spans)
      displaySentimentExtremes(spans);
      
      // Add this prediction to history for quick re-access
      addToHistory(text);
      
      // Close the SSE connection to free resources
      es.close();
    });
    
    // Event: error - Something went wrong during prediction
    es.addEventListener("error", (e) => {
      // Try to parse error message if it's a proper error event
      let errorMsg = "Connection failed or server error";
      try {
        if (e.data) {
          const errData = JSON.parse(e.data);
          errorMsg = errData.error || errorMsg;
        }
      } catch {
        // If we can't parse the error data, use generic message
      }
      
      setStatus("Error", "err");
      msgEl.textContent = errorMsg;
      msgEl.className = "err";
      
      // Close the connection
      es.close();
    });
  });

function drawLineChart(canvas, points) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  // Clear
  ctx.clearRect(0, 0, W, H);

  if (!points.length) {
    ctx.fillStyle = "rgba(232,236,255,0.7)";
    ctx.font = "14px system-ui";
    ctx.fillText("No data yet. Make predictions first.", 16, 28);
    return;
  }

  // Sort by text length so the line makes sense
  points.sort((a, b) => a.text_len - b.text_len);

  // Compute bounds
  const minX = Math.min(...points.map(p => p.text_len));
  const maxX = Math.max(...points.map(p => p.text_len));
  const minY = 0;
  const maxY = Math.max(...points.map(p => p.latency_ms)) || 1;

  // Padding
  const padL = 48, padR = 14, padT = 14, padB = 34;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xToPx = (x) => padL + (maxX === minX ? plotW/2 : ((x - minX) / (maxX - minX)) * plotW);
  const yToPx = (y) => padT + (1 - (y - minY) / (maxY - minY)) * plotH;

  // Grid lines
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;

  const gridY = 4;
  for (let i = 0; i <= gridY; i++) {
    const y = padT + (i / gridY) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
  }

  // Axes labels (simple)
  ctx.fillStyle = "rgba(232,236,255,0.75)";
  ctx.font = "12px system-ui";
  ctx.fillText(`${minX} chars`, padL, H - 12);
  ctx.fillText(`${maxX} chars`, padL + plotW - 60, H - 12);
  ctx.fillText(`${maxY} ms`, 8, padT + 10);

  // Line
  ctx.strokeStyle = "rgba(182,255,204,0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();

  points.forEach((p, i) => {
    const x = xToPx(p.text_len);
    const y = yToPx(p.latency_ms);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Points (small dots)
  ctx.fillStyle = "rgba(182,255,204,0.9)";
  for (const p of points) {
    const x = xToPx(p.text_len);
    const y = yToPx(p.latency_ms);
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
}

  // ===== Autocomplete Functionality =====
  // Provides real-time word suggestions based on historical review data
  // to help users type common food review terms faster

  // DOM element and state variables for autocomplete
  const acBox = document.getElementById("acBox");
  let acItems = [];      // Array of current suggestion strings
  let acIndex = -1;      // Index of currently selected suggestion (-1 = none)
  let acTimer = null;    // Debounce timer to limit API calls

  /**
   * Extract the last partial word (token) from the text
   * Used to determine what the user is currently typing
   * @param {string} text - The full text input
   * @returns {string} The last word/token or empty string
   */
  function getLastToken(text) {
    const m = (text || "").match(/([A-Za-z']{1,40})$/);
    return m ? m[1] : "";
  }

  /**
   * Replace the last partial word with the selected suggestion
   * @param {string} text - The full text input
   * @param {string} replacement - The word to replace the last token with
   * @returns {string} The updated text
   */
  function replaceLastToken(text, replacement) {
    return (text || "").replace(/([A-Za-z']{1,40})$/, replacement);
  }

  /**
   * Hide the autocomplete dropdown and reset state
   */
  function hideAc() {
    acBox.style.display = "none";
    acBox.innerHTML = "";
    acItems = [];
    acIndex = -1;
  }

  /**
   * Render the autocomplete dropdown with suggestions
   * @param {string[]} list - Array of suggestion words to display
   */
  function renderAc(list) {
    acItems = list || [];
    acIndex = -1;  // Reset selection when list changes

    // Hide if no suggestions
    if (!acItems.length) return hideAc();

    // Build HTML for each suggestion item
    acBox.innerHTML = acItems.map((w, i) => `
      <div class="ac-item"
        data-i="${i}"
        style="padding:10px 12px; cursor:pointer; border-top:1px solid rgba(255,255,255,0.06);">
        ${escapeHtml(w)}
      </div>
    `).join("");

    // Remove top border on first item for cleaner appearance
    const first = acBox.firstElementChild;
    if (first) first.style.borderTop = "none";

    // Show the dropdown
    acBox.style.display = "block";
  }

  /**
   * Fetch autocomplete suggestions from the server
   * @param {string} prefix - The partial word to get suggestions for
   * @returns {Promise<string[]>} Array of suggestion words
   */
  async function fetchAc(prefix) {
    const res = await fetch(`./autocomplete?q=${encodeURIComponent(prefix)}&limit=8&window=500`, { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || "Autocomplete failed");
    return data.suggestions || [];
  }

  /**
   * Apply the selected autocomplete suggestion to the text input
   * @param {string} word - The complete word to insert
   */
  function applySuggestion(word) {
    // Replace the partial word with the complete suggestion
    txt.value = replaceLastToken(txt.value, word);
    
    // Re-run PII detection on updated text
    updatePIIUI();
    
    // Hide dropdown and return focus to input
    hideAc();
    txt.focus();
  }

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
    // Debounce PII check to avoid excessive processing on every keystroke
    clearTimeout(piiTimer);
    piiTimer = setTimeout(updatePIIUI, 120);

    // Debounce autocomplete to avoid excessive API calls
    // Only trigger after user pauses typing (140ms delay)
    clearTimeout(acTimer);
    acTimer = setTimeout(async () => {
      try {
        // Extract the current word being typed
        const token = getLastToken(txt.value);
        
        // Don't show suggestions for very short tokens (need at least 2 chars)
        if (!token || token.length < 2) return hideAc();

        // Safety check: Don't autocomplete if the token looks like PII
        // This prevents suggesting sensitive data back to the user
        const piiMatches = findMatches(token);
        if (piiMatches.length) return hideAc();

        // Fetch and display suggestions from the server
        const suggestions = await fetchAc(token);
        renderAc(suggestions);
      } catch {
        // On any error (network, server), silently hide autocomplete
        hideAc();
      }
    }, 140);
  });

  // Mouse interaction: click to select a suggestion
  acBox.addEventListener("click", (e) => {
    const row = e.target.closest(".ac-item");
    if (!row) return;
    
    // Get the index from the data attribute
    const i = Number(row.dataset.i);
    if (!Number.isFinite(i) || !acItems[i]) return;
    
    // Apply the clicked suggestion
    applySuggestion(acItems[i]);
  });

  // Keyboard navigation for accessibility and power users
  // Supports: Arrow keys for selection, Enter to apply, Escape to close
  txt.addEventListener("keydown", (e) => {
    // Only handle keys when autocomplete is visible
    if (acBox.style.display !== "block") return;

    // Escape key: Close autocomplete without applying
    if (e.key === "Escape") {
      hideAc();
      return;
    }

    // Arrow Down: Move selection down
    if (e.key === "ArrowDown") {
      e.preventDefault();
      acIndex = Math.min(acIndex + 1, acItems.length - 1);
    } 
    // Arrow Up: Move selection up
    else if (e.key === "ArrowUp") {
      e.preventDefault();
      acIndex = Math.max(acIndex - 1, 0);
    } 
    // Enter key: Apply the currently selected suggestion
    else if (e.key === "Enter" && acIndex >= 0) {
      e.preventDefault();
      applySuggestion(acItems[acIndex]);
      return;
    } 
    // Other keys: Ignore (let normal typing happen)
    else {
      return;
    }

    // Update visual highlight to show the currently selected item
    [...acBox.querySelectorAll(".ac-item")].forEach((el, idx) => {
      el.style.background = (idx === acIndex) ? "rgba(255,255,255,0.08)" : "transparent";
    });
  });

  // Click-away handler: Hide autocomplete when clicking outside the input or dropdown
  document.addEventListener("click", (e) => {
    if (e.target === txt || acBox.contains(e.target)) return;
    hideAc();
  });


  let dbSearchTimer = null;
  dbSearch.addEventListener("input", () => {
    clearTimeout(dbSearchTimer);
    dbSearchTimer = setTimeout(renderDbTable, 80);
  });

  // Initialize PII UI on page load
  updatePIIUI();
  refreshServerLoad();
  setInterval(refreshServerLoad, 750);
  plotLatencyVsLength();

</script>
</body>
</html>

"""