"""Model management for loading and using HuggingFace transformer models."""

import os
from config import MODEL_DIR


# Global classifier object - loaded once on startup and reused for all requests
_classifier = None


def load_model():
    """Load the HuggingFace sentiment model into memory.
    
    Loads the pre-trained transformer model from MODEL_DIR and initializes
    a text classification pipeline. Runs once at server startup.
    
    Raises:
        FileNotFoundError: If the model directory doesn't exist
    """
    global _classifier
    from transformers import pipeline

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


def get_classifier():
    """Get the global classifier instance.
    
    Returns:
        The loaded classifier pipeline, or None if not yet loaded
    """
    return _classifier


def is_model_loaded() -> bool:
    """Check if the model has been loaded.
    
    Returns:
        True if model is loaded, False otherwise
    """
    return _classifier is not None
