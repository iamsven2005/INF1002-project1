"""Model management for loading and using HuggingFace transformer models."""

import os
from config import MODEL_DIR

_classifier = None

def load_model():
    global _classifier
    from transformers import pipeline

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

    _classifier = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        return_all_scores=False,
        truncation=True,
        max_length=512,
        batch_size=16,   # ✅ speeds up span processing
    )


def get_classifier():
    return _classifier

def is_model_loaded() -> bool:
    return _classifier is not None
