"""Concurrency tracking and rate limiting for the sentiment analysis API.

This module provides thread-safe tracking of concurrent requests and implements
a semaphore-based rate limiter to prevent model overload. It tracks three key metrics:
- HTTP requests currently being processed
- Prediction requests actively running model inference
- Prediction requests waiting for available slots

The semaphore limits concurrent model inference to prevent memory exhaustion
and maintain response time SLAs.
"""

import asyncio
import time
from dataclasses import dataclass

@dataclass
class ConcurrencySnapshot:
    """Immutable snapshot of current concurrency state.
    
    Attributes:
        http_in_flight: Number of HTTP requests currently being processed
        predict_in_flight: Number of predictions actively running model inference
        predict_waiting: Number of predictions waiting in queue for model access
    """
    http_in_flight: int
    predict_in_flight: int
    predict_waiting: int

class ConcurrencyTracker:
    """Thread-safe tracker for concurrent request metrics with rate limiting.
    
    Uses asyncio primitives (Lock and Semaphore) to coordinate concurrent access
    to shared counters and limit model inference concurrency. The semaphore prevents
    too many simultaneous predictions, which could exhaust GPU/CPU memory or degrade
    response times.
    
    Typical usage:
        tracker = ConcurrencyTracker(max_predict_concurrency=4)
        
        async def handle_request():
            await tracker.http_enter()
            try:
                await tracker.predict_acquire()
                try:
                    result = run_model(text)
                finally:
                    await tracker.predict_release()
            finally:
                await tracker.http_exit()
    """
    
    def __init__(self, max_predict_concurrency: int = 4):
        """Initialize the concurrency tracker.
        
        Args:
            max_predict_concurrency: Maximum number of simultaneous model inference
                calls allowed. Limits resource usage and maintains consistent latency.
                Default of 4 balances throughput with memory constraints.
        """
        # Counters for different stages of request processing
        self._http_in_flight = 0        # Total HTTP requests being handled
        self._predict_in_flight = 0     # Predictions actively running model
        self._predict_waiting = 0       # Predictions queued, waiting for slot

        # Lock protects counter updates from race conditions
        self._lock = asyncio.Lock()
        
        # Semaphore limits concurrent model inference to prevent overload
        self._sem = asyncio.Semaphore(max_predict_concurrency)

    async def http_enter(self):
        """Record that an HTTP request has started processing.
        
        Should be called at the beginning of each request handler.
        Thread-safe: uses lock to prevent race conditions on counter update.
        """
        async with self._lock:
            self._http_in_flight += 1

    async def http_exit(self):
        """Record that an HTTP request has finished processing.
        
        Should be called in a finally block to ensure cleanup even on errors.
        Thread-safe: uses lock to prevent race conditions on counter update.
        """
        async with self._lock:
            self._http_in_flight -= 1

    async def snapshot(self) -> ConcurrencySnapshot:
        """Capture an atomic snapshot of current concurrency metrics.
        
        Returns a consistent view of all three counters taken while holding
        the lock, ensuring they represent the same moment in time.
        
        Returns:
            ConcurrencySnapshot with current values of all tracked metrics
            
        Use case:
            Logged with each prediction for performance analysis and
            capacity planning. Helps identify bottlenecks and queue buildup.
        """
        async with self._lock:
            return ConcurrencySnapshot(
                http_in_flight=self._http_in_flight,
                predict_in_flight=self._predict_in_flight,
                predict_waiting=self._predict_waiting,
            )

    async def predict_acquire(self):
        """Acquire a slot for model inference, blocking if at capacity.
        
        Three-phase process:
        1. Atomically increment waiting counter (shows queue buildup)
        2. Block on semaphore until slot available (enforces rate limit)
        3. Atomically decrement waiting, increment in_flight (transition to active)
        
        This will block the coroutine if max_predict_concurrency predictions
        are already running. The waiting counter allows monitoring queue depth.
        
        Must be paired with predict_release() in a try/finally block.
        """
        # Phase 1: Mark this request as waiting for a slot
        async with self._lock:
            self._predict_waiting += 1

        # Phase 2: Block until semaphore slot available (may wait here)
        await self._sem.acquire()

        # Phase 3: Got a slot! Move from waiting to in_flight
        async with self._lock:
            self._predict_waiting -= 1
            self._predict_in_flight += 1

    async def predict_release(self):
        """Release a model inference slot, allowing queued requests to proceed.
        
        Decrements the in_flight counter and releases the semaphore slot,
        which will wake up one waiting coroutine (if any) to acquire the slot.
        
        Must be called in a finally block to ensure release even on errors,
        otherwise the semaphore could become permanently exhausted.
        """
        # Decrement in_flight counter
        async with self._lock:
            self._predict_in_flight -= 1
        
        # Release semaphore slot, allowing one waiting request to proceed
        self._sem.release()

# Global tracker instance shared across all requests
# Limits concurrent model inference to 4 to balance throughput with resource constraints
tracker = ConcurrencyTracker(max_predict_concurrency=4)

def now_ms() -> int:
    """Get current timestamp in milliseconds since Unix epoch.
    
    Returns:
        Current time as integer milliseconds (not float seconds)
        
    Use case:
        Consistent timestamp format for database logging and latency calculations.
        Millisecond precision is sufficient for API latency tracking while avoiding
        floating point precision issues.
    """
    return int(time.time() * 1000)


def sliding_window_paragraph_sentiment(spans, window_size, score_fn):
    windows = []
    for i in range(len(spans) - window_size + 1):
        window = spans[i:i + window_size]
        score = sum(score_fn(s["text"]) for s in window)
        windows.append({"start": i, "end": i + window_size - 1, "score": score, "spans": window})
    return windows