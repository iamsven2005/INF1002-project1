"""Configuration constants for the Food Review Sentiment Analysis API."""

# Path to locally saved HuggingFace model directory
MODEL_DIR = "./food_sentiment_model"

# Mapping from model output label IDs to human-readable sentiment labels
LABELS = {
    0: "negative",    # Negative sentiment
    1: "neutral",     # Neutral/mixed sentiment
    2: "positive"     # Positive sentiment
}
