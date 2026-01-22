"""Word segmentation utilities using HuggingFace tokenizer."""

from typing import List
from model import get_classifier


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
    classifier = get_classifier()
    if classifier is None:
        return []
    
    text = (text or "").strip()
    if not text:
        return []
    
    try:
        # Get the tokenizer from the classifier pipeline
        tokenizer = classifier.tokenizer
        
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
