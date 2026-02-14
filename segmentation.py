"""Word segmentation utilities using HuggingFace tokenizer."""

from typing import List
from model import get_classifier
import re
import wordninja

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
    text = (text or "").strip()
    if not text:
        return []

    # Keep punctuation, but segment the alphanumeric runs
    parts = re.findall(r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]+", text)
    out = []
    for p in parts:
        if p.isalpha():
            out.extend(wordninja.split(p))
        else:
            out.append(p)

    # clean spacing before punctuation like "." "," "!" "?"
    s = " ".join(out)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return [s]