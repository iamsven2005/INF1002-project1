"""Database module for logging and retrieving prediction history.

This module provides a lightweight SQLite-based storage layer for tracking
sentiment analysis predictions. It stores metadata about predictions (length,
hash, latency, concurrency metrics) rather than raw text to minimize PII exposure.

Key Features:
- WAL mode for improved read/write concurrency
- Stores text hash instead of raw content for privacy
- Tracks performance metrics (latency, in-flight requests)
- Provides query interfaces for analytics and history
"""

# Standard library imports for database operations and data handling
import json
import hashlib
import sqlite3
from typing import Any, Optional, Dict, List
from pathlib import Path

# Database file path - uses SQLite for zero-configuration persistence
DB_PATH = Path("predictions.sqlite")

def _connect():
    """Create and configure a SQLite database connection.
    
    Optimizations:
    - WAL (Write-Ahead Logging) mode: Allows concurrent reads during writes,
      significantly improving performance in multi-threaded environments
    - SYNCHRONOUS=NORMAL: Balances durability with performance by reducing
      disk sync frequency (safe for WAL mode)
    - check_same_thread=False: Allows connection sharing across threads
      (use with caution in production)
    
    Returns:
        sqlite3.Connection: Configured database connection
    """
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    
    # Enable WAL mode for better concurrency - readers don't block writers
    conn.execute("PRAGMA journal_mode=WAL;")
    
    # NORMAL synchronous mode: fsync only at critical checkpoints
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    return conn

# Global connection object - initialized once and reused for efficiency
_CONN = None

def init_db():
    """Initialize the database schema and global connection.
    
    Creates the predictions table if it doesn't exist. This function is
    idempotent and safe to call multiple times.
    
    Schema Details:
    - id: Auto-incrementing primary key for unique identification
    - ts_ms: Timestamp in milliseconds (Unix epoch) for temporal analysis
    - endpoint: API endpoint used (/predict or /predict_spans)
    - text_len: Length of input text in characters
    - text_sha256: SHA-256 hash of input text (for deduplication without storing PII)
    - label: Predicted sentiment (positive/neutral/negative)
    - confidence: Model confidence score (0.0 to 1.0)
    - spans_json: JSON-serialized span-level predictions (nullable)
    - latency_ms: Request processing time in milliseconds
    - http_in_flight: Number of concurrent HTTP requests at prediction time
    - predict_in_flight: Number of concurrent model inference calls
    - predict_waiting: Number of requests waiting for model access
    """
    global _CONN
    _CONN = _connect()
    
    # Create predictions table with comprehensive metrics tracking
    _CONN.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_ms INTEGER NOT NULL,
        endpoint TEXT NOT NULL,
        text_len INTEGER NOT NULL,
        text_sha256 TEXT NOT NULL,
        label TEXT NOT NULL,
        confidence REAL NOT NULL,
        spans_json TEXT,
        latency_ms INTEGER NOT NULL,
        http_in_flight INTEGER NOT NULL,
        predict_in_flight INTEGER NOT NULL,
        predict_waiting INTEGER NOT NULL
    );
    """)
    _CONN.commit()
    try:
        _CONN.execute("ALTER TABLE predictions ADD COLUMN text TEXT;")
        _CONN.commit()
    except sqlite3.OperationalError:
        # column already exists
        pass

def _sha256(text: str) -> str:
    """Compute SHA-256 hash of text for privacy-preserving storage.
    
    Instead of storing raw review text (which may contain PII), we store
    a cryptographic hash. This allows deduplication and verification while
    protecting user privacy.
    
    Args:
        text: Input text to hash
    
    Returns:
        Hexadecimal string representation of SHA-256 hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fetch_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent prediction records from the database.
    
    Fetches the most recent predictions in reverse chronological order.
    Useful for displaying prediction history, analytics dashboards, or
    debugging model behavior.
    
    Args:
        limit: Maximum number of records to return (default: 50)
    
    Returns:
        List of dictionaries, each containing prediction metadata:
        - id, ts_ms, endpoint, text_len, text_sha256
        - label, confidence, latency_ms
        - http_in_flight, predict_in_flight, predict_waiting
        
    Note: Does not include spans_json to reduce payload size for listing views
    """
    # Lazy initialization - create database if not yet initialized
    if _CONN is None:
        init_db()

    # Query predictions in descending order (newest first)
    cur = _CONN.execute(
        """
        SELECT
          id, ts_ms, endpoint, text_len, text_sha256,
          label, confidence, latency_ms,
          http_in_flight, predict_in_flight, predict_waiting,           substr(COALESCE(text, ''), 1, 160) AS text_preview
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    
    # Convert SQLite rows to list of dictionaries for easier JSON serialization
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]  # Extract column names from cursor
    return [dict(zip(cols, r)) for r in rows]

def fetch_latency_vs_length(limit: int = 500) -> List[Dict[str, Any]]:
    """Retrieve text length vs. latency data for performance analysis.
    
    Fetches a lightweight dataset correlating input text length with
    processing time. Useful for:
    - Performance monitoring and optimization
    - Identifying bottlenecks with longer inputs
    - Capacity planning and SLA validation
    - Generating performance visualization charts
    
    Args:
        limit: Maximum number of records to return (default: 500)
    
    Returns:
        List of dictionaries with keys: text_len, latency_ms, ts_ms
    """
    # Lazy initialization
    if _CONN is None:
        init_db()

    # Query minimal fields needed for latency analysis
    cur = _CONN.execute(
        """
        SELECT text_len, latency_ms, ts_ms
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    
    # Convert to list of dicts with explicit keys
    rows = cur.fetchall()
    return [{"text_len": r[0], "latency_ms": r[1], "ts_ms": r[2]} for r in rows]

def log_prediction(
    *,
    ts_ms: int,
    endpoint: str,
    text: str,
    label: str,
    confidence: float,
    spans: Optional[Any],
    latency_ms: int,
    http_in_flight: int,
    predict_in_flight: int,
    predict_waiting: int,
):
    """Log a prediction to the database with full metrics.
    
    Stores prediction metadata without raw text content to protect user privacy.
    Instead of storing the actual review text, we store its SHA-256 hash and
    length, which allows deduplication and analysis without PII exposure.
    
    Args:
        ts_ms: Timestamp in milliseconds (Unix epoch time)
        endpoint: API endpoint called (e.g., "/predict" or "/predict_spans")
        text: Input text (only hash and length are stored, not raw content)
        label: Predicted sentiment label ("positive", "neutral", or "negative")
        confidence: Model confidence score between 0.0 and 1.0
        spans: Optional span-level predictions (serialized as JSON if provided)
        latency_ms: Request processing time in milliseconds
        http_in_flight: Number of concurrent HTTP requests at time of prediction
        predict_in_flight: Number of concurrent model inference calls
        predict_waiting: Number of requests waiting in queue for model access
    
    Privacy Note:
        Raw text is NOT stored - only SHA-256 hash and character length are
        persisted. This prevents PII leakage while enabling analytics.
    """
    # Lazy initialization - create database if not yet initialized
    if _CONN is None:
        init_db()

    # Serialize span predictions to JSON if provided
    spans_json = None
    if spans is not None:
        spans_json = json.dumps(spans, ensure_ascii=False)

    # Insert prediction record with all metrics
    _CONN.execute(
        """
        INSERT INTO predictions (
            ts_ms, endpoint, text_len, text_sha256, label, confidence,
            spans_json, latency_ms,
            http_in_flight, predict_in_flight, predict_waiting, text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts_ms,
            endpoint,
            len(text),              # Store length, not raw text
            _sha256(text),          # Store hash for deduplication
            label,
            float(confidence),
            spans_json,
            int(latency_ms),
            int(http_in_flight),
            int(predict_in_flight),
            int(predict_waiting),
            text,
        ),
    )
    
    # Commit immediately to ensure data durability
    _CONN.commit()
def fetch_word_frequencies(limit: int = 500) -> Dict[str, int]:
    """Extract and count word frequencies from all predictions in database.
    
    Analyzes all stored prediction texts to generate word frequency data
    for word cloud visualization. Applies simple filtering (minimum length,
    case normalization) to focus on meaningful content words.
    
    Args:
        limit: Maximum number of prediction records to analyze (default: 500)
    
    Returns:
        Dictionary mapping word strings to their occurrence counts.
        Example: {"delicious": 23, "food": 45, "amazing": 18, ...}
        
    Note: 
        - Filters out words shorter than 3 characters
        - Case-insensitive counting (all words normalized to lowercase)
        - Includes common stop words (future enhancement: could filter these)
    """
    # Lazy initialization - create database if not yet initialized
    if _CONN is None:
        init_db()
    
    # Fetch recent predictions with their text content
    cur = _CONN.execute(
        """
        SELECT text FROM predictions
        WHERE text IS NOT NULL AND text != ''
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    
    word_freq = {}
    
    # Process each prediction text
    for (text,) in cur.fetchall():
        if not text:
            continue
        
        # Split text into words and normalize
        words = text.lower().split()
        
        for word in words:
            # Remove punctuation and filter
            clean_word = ''.join(c for c in word if c.isalnum())
            
            # Skip very short words and empty strings
            if len(clean_word) >= 3:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
    
    return word_freq