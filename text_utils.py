"""Text processing utilities for sentiment analysis."""

import re
from typing import List, Dict


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
