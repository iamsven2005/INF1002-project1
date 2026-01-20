# Standard library imports
import os

# Data processing
import pandas as pd
from sklearn.model_selection import train_test_split

# Version handling (not strictly needed here but useful for compatibility checks)
from packaging import version

# HuggingFace / evaluation
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# Configuration
# -----------------------------

# Base model to fine-tune
MODEL_NAME = "distilbert-base-uncased"

# Where the trained model + tokenizer will be saved
OUTPUT_DIR = "./food_sentiment_model"


# -----------------------------
# Label conversion
# -----------------------------

def score_to_label(score: int) -> int:
    """
    Convert a numeric review score into sentiment labels:
        0 → negative
        1 → neutral
        2 → positive

    Example mapping:
        1,2 → negative
        3   → neutral
        4,5 → positive
    """
    if score <= 2:
        return 0
    if score == 3:
        return 1
    return 2


# -----------------------------
# Load & preprocess dataset
# -----------------------------

def load_and_prepare(csv_path: str, text_col: str = "Text", score_col: str = "Score"):
    """
    Loads a CSV file and prepares training / testing datasets.

    Steps:
    1. Load CSV
    2. Remove missing values
    3. Clean text
    4. Convert numeric scores to sentiment labels
    5. Split into train/test sets
    """

    # Load CSV into pandas DataFrame
    df = pd.read_csv(csv_path)

    # Drop rows with missing text or score
    df = df.dropna(subset=[text_col, score_col])

    # Ensure text is string and strip whitespace
    df[text_col] = df[text_col].astype(str).str.strip()

    # Remove empty text rows
    df = df[df[text_col].str.len() > 0]

    # Convert numeric scores into sentiment labels
    df["label"] = df[score_col].astype(int).apply(score_to_label)

    # Optional: reduce dataset size for fast experiments
    # df = df.sample(n=min(len(df), 100000), random_state=42)

    # Split into training and testing sets (80/20 split)
    train_df, test_df = train_test_split(
        df[[text_col, "label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]  # preserve class distribution
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# -----------------------------
# Training pipeline
# -----------------------------

def train(csv_path: str):
    """
    Full training pipeline:
    1. Load and clean data
    2. Tokenize text
    3. Load pretrained DistilBERT
    4. Fine-tune on sentiment classification
    5. Save trained model + tokenizer
    """

    # Prepare dataset
    train_df, test_df = load_and_prepare(csv_path)

    # Convert pandas DataFrames into HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    # Load tokenizer for the base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization function
    def tokenize(batch):
        """
        Tokenizes a batch of texts:
        - Truncates long sequences
        - Pads dynamically later via DataCollator
        """
        return tokenizer(batch["Text"], truncation=True, max_length=256)

    # Apply tokenization to datasets
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["Text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["Text"])

    # Handles dynamic padding in each batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load the pretrained model and configure for 3-class classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    # Load evaluation metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        """
        Compute metrics during evaluation.
        Returns:
        - Accuracy
        - Macro F1 score (balanced across classes)
        """
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # Training configuration
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",              # Evaluate once per epoch
        save_strategy="no",                 # Do not save checkpoints during training
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        fp16=True,                          # Enable if you have a GPU that supports FP16
        report_to="none",                   # Disable external logging (WandB, etc.)
    )

    # Trainer object handles training loop, evaluation, and logging
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save trained model
    # - safe_serialization=True → saves as .safetensors
    # - max_shard_size keeps files small enough for GitHub
    trainer.model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
        max_shard_size="100MB",
    )

    # Save tokenizer files
    tokenizer.save_pretrained(OUTPUT_DIR)

    return {"model_dir": OUTPUT_DIR}


# -----------------------------
# CLI entry point
# -----------------------------

if __name__ == "__main__":
    """
    Run from command line:
        python train.py --csv reviews.csv

    This trains the model and writes:
        ./food_sentiment_model/
    """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to reviews CSV")
    args = p.parse_args()

    print(train(args.csv))
